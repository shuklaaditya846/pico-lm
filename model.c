#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ---- PSRAM bulk float helpers -------------------------------------------

// The PIO SPI program packs the transaction bit-count into a single uint8_t
// (write_command[0] = (4 + count) * 8 in psram_spi.h), which overflows past
// ~27-31 bytes. Larger single calls silently corrupt the PIO transaction and
// hang dma_channel_wait_for_finish_blocking() forever. Chunk everything.
#define PSRAM_MAX_CHUNK 16u

static inline void kv_write(psram_spi_inst_t* spi, uint32_t addr, const float* v, int n) {
    const uint8_t* src = (const uint8_t*)v;
    size_t remaining = (size_t)n * sizeof(float);
    uint32_t a = addr;
    uint8_t verify[PSRAM_MAX_CHUNK];
    while (remaining > 0) {
        size_t chunk = remaining < PSRAM_MAX_CHUNK ? remaining : PSRAM_MAX_CHUNK;
        for (int attempt = 0; attempt < 5; attempt++) {
            psram_write(spi, a, src, chunk);
            psram_read(spi, a, verify, chunk);
            if (memcmp(verify, src, chunk) == 0) break;
#ifdef PSRAM_DEBUG
            printf("PSRAM write mismatch at addr %lu, attempt %d\n", (unsigned long)a, attempt);
#endif
        }
        src += chunk; a += chunk; remaining -= chunk;
    }
}
static inline void kv_read(psram_spi_inst_t* spi, uint32_t addr, float* v, int n) {
    for (int attempt = 0; attempt < 4; attempt++) {
        uint8_t* dst = (uint8_t*)v;
        size_t remaining = (size_t)n * sizeof(float);
        uint32_t a = addr;
        while (remaining > 0) {
            size_t chunk = remaining < PSRAM_MAX_CHUNK ? remaining : PSRAM_MAX_CHUNK;
            psram_read(spi, a, dst, chunk);
            dst += chunk; a += chunk; remaining -= chunk;
        }
        // Sanity check: reject NaN/Inf or implausible magnitudes and retry the
        // read. Real activations in this model stay well under this range;
        // anything wildly outside it means the SPI transfer glitched.
        int ok = 1;
        for (int i = 0; i < n; i++) {
            float val = v[i];
            if (!(val == val) || val > 1e4f || val < -1e4f) { ok = 0; break; }
        }
        if (ok) return;
#ifdef PSRAM_DEBUG
        printf("kv_read: implausible data at addr %lu, retrying (attempt %d)\n", (unsigned long)addr, attempt);
#endif
    }
    // Every attempt looked bad - zero it out rather than let garbage/NaN
    // cascade through the rest of generation forever.
    memset(v, 0, (size_t)n * sizeof(float));
}

// One-shot diagnostic: tells us exactly where a NaN/Inf first appears so we
// know whether the write-verify above actually fixed the root cause, or
// whether something else is overflowing.
static int nan_already_reported = 0;
static void check_finite(const char* where, const float* buf, int n, int layer, int pos) {
    if (nan_already_reported) return;
    for (int i = 0; i < n; i++) {
        if (!(buf[i] == buf[i]) || buf[i] > 1e30f || buf[i] < -1e30f) { // NaN or blown-up value
            printf("[DIAG] non-finite value at %s, layer=%d pos=%d index=%d value=%f\n",
                   where, layer, pos, i, (double)buf[i]);
            nan_already_reported = 1;
            return;
        }
    }
}

// ---- math primitives ------------------------------------------------------

static void rmsnorm(float* o, const float* x, const float* w, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int i = 0; i < size; i++) o[i] = w[i] * (ss * x[i]);
}

static void softmax(float* x, int size) {
    float maxv = x[0];
    for (int i = 1; i < size; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// out[d] = sum_n w[d*n_ + i] * x[i]   (w is row-major, d rows of n_ cols)
static void matmul(float* out, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        const float* row = w + (size_t)i * n;
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
}

// ---- init -------------------------------------------------------------

void transformer_init(Transformer* t, const uint8_t* blob) {
    int32_t h[7];
    memcpy(h, blob, sizeof(h)); // header is always 4-byte aligned by construction

    Config* p = &t->config;
    p->dim = h[0];
    p->hidden_dim = h[1];
    p->n_layers = h[2];
    p->n_heads = h[3];
    p->n_kv_heads = h[4];
    int vocab_raw = h[5];
    p->seq_len = h[6];
    int shared_weights = vocab_raw > 0;
    p->vocab_size = shared_weights ? vocab_raw : -vocab_raw;

    int head_size = p->dim / p->n_heads;
    const float* ptr = (const float*)(blob + sizeof(h));

    TransformerWeights* w = &t->weights;
    w->token_embedding_table = ptr; ptr += (size_t)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;        ptr += (size_t)p->n_layers * p->dim;
    w->wq = ptr;                    ptr += (size_t)p->n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                    ptr += (size_t)p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                    ptr += (size_t)p->n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                    ptr += (size_t)p->n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;        ptr += (size_t)p->n_layers * p->dim;
    w->w1 = ptr;                    ptr += (size_t)p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                    ptr += (size_t)p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                    ptr += (size_t)p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;      ptr += p->dim;
    ptr += (size_t)p->seq_len * head_size / 2; // legacy freq_cis_real placeholder (unused, RoPE computed on the fly)
    ptr += (size_t)p->seq_len * head_size / 2; // legacy freq_cis_imag placeholder
    w->wcls = shared_weights ? w->token_embedding_table : ptr;

    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    RunState* s = &t->state;
    s->x      = malloc((size_t)p->dim * sizeof(float));
    s->xb     = malloc((size_t)p->dim * sizeof(float));
    s->xb2    = malloc((size_t)p->dim * sizeof(float));
    s->hb     = malloc((size_t)p->hidden_dim * sizeof(float));
    s->hb2    = malloc((size_t)p->hidden_dim * sizeof(float));
    s->q      = malloc((size_t)p->dim * sizeof(float));
    s->keyt   = malloc((size_t)kv_dim * sizeof(float));
    s->valt   = malloc((size_t)kv_dim * sizeof(float));
    s->att    = malloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = malloc((size_t)p->vocab_size * sizeof(float));

    s->key_cache_base = 0;
    s->value_cache_base = (uint32_t)((size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float));
}

// ---- forward ------------------------------------------------------------

float* transformer_forward(Transformer* t, psram_spi_inst_t* psram, int token, int pos) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(s->x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + (size_t)l * dim, dim);

        matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
        // k/v go straight into small local buffers before being written to PSRAM
        float* k_local = s->keyt;
        float* v_local = s->valt;
        matmul(k_local, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        matmul(v_local, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);
        check_finite("k_local", k_local, kv_dim, l, pos);
        check_finite("v_local", v_local, kv_dim, l, pos);

        // RoPE, computed on the fly (matches modern llama2.c export format)
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = (i < kv_dim) ? 2 : 1; // rotate q and k, or just q if dims differ (GQA)
            for (int v = 0; v < rotn; v++) {
                float* vec = (v == 0) ? s->q : k_local;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        uint32_t key_addr   = s->key_cache_base   + ((size_t)(l * p->seq_len + pos) * kv_dim) * sizeof(float);
        uint32_t value_addr = s->value_cache_base + ((size_t)(l * p->seq_len + pos) * kv_dim) * sizeof(float);
        kv_write(psram, key_addr, k_local, kv_dim);
        kv_write(psram, value_addr, v_local, kv_dim);

        // Pass 1: attention scores for ALL heads, reading each cached position once
        for (int tstep = 0; tstep <= pos; tstep++) {
            uint32_t kaddr = s->key_cache_base + ((size_t)(l * p->seq_len + tstep) * kv_dim) * sizeof(float);
            kv_read(psram, kaddr, s->keyt, kv_dim);
            for (int hh = 0; hh < p->n_heads; hh++) {
                int kvh = hh / kv_mul;
                const float* qh = s->q + hh * head_size;
                const float* kh = s->keyt + kvh * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += qh[i] * kh[i];
                score /= sqrtf((float)head_size);
                s->att[hh * p->seq_len + tstep] = score;
            }
        }
        for (int hh = 0; hh < p->n_heads; hh++) softmax(s->att + hh * p->seq_len, pos + 1);

        memset(s->xb, 0, dim * sizeof(float));
        // Pass 2: weighted sum over values, again reading each cached position once
        for (int tstep = 0; tstep <= pos; tstep++) {
            uint32_t vaddr = s->value_cache_base + ((size_t)(l * p->seq_len + tstep) * kv_dim) * sizeof(float);
            kv_read(psram, vaddr, s->valt, kv_dim);
            for (int hh = 0; hh < p->n_heads; hh++) {
                int kvh = hh / kv_mul;
                float a = s->att[hh * p->seq_len + tstep];
                float* xbh = s->xb + hh * head_size;
                const float* vh = s->valt + kvh * head_size;
                for (int i = 0; i < head_size; i++) xbh[i] += a * vh[i];
            }
        }

        matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        check_finite("x_after_attn", s->x, dim, l, pos);

        // FFN (SwiGLU)
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (size_t)l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float v = s->hb[i];
            v *= (1.0f / (1.0f + expf(-v))); // SiLU
            s->hb[i] = v * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->w2 + (size_t)l * hidden_dim * dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
        check_finite("x_after_ffn", s->x, dim, l, pos);
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    check_finite("logits", s->logits, p->vocab_size, p->n_layers, pos);
    return s->logits;
}
