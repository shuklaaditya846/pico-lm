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
// The PIO SPI program packs the transaction length into a single byte:
//   write_command[0] = (4 + count) * 8  -> needs 4+count <= 31, so count <= 27
//   read_command[1]  = count * 8        -> needs count <= 31
// Exceeding those silently wraps the byte and corrupts the transaction (which
// then hangs the DMA completion wait forever). Use the largest 4-aligned values
// that stay inside both limits - bigger chunks mean proportionally fewer SPI
// transactions, which is the dominant cost of KV-cache traffic.
#define PSRAM_MAX_WRITE_CHUNK 24u
#define PSRAM_MAX_READ_CHUNK  28u
#define PSRAM_MAX_CHUNK       PSRAM_MAX_WRITE_CHUNK

// ---- DIAGNOSTIC MODE ------------------------------------------------------
// Build with -DKV_CACHE_IN_SRAM to keep the KV cache in SRAM and never touch
// PSRAM at all. The cache needs n_layers*seq_len*kv_dim*4*2 bytes, so seq_len
// is clamped to KV_DEBUG_MAX_SEQ to fit in the RP2350's 520KB of SRAM.
//
// Purpose: split "PSRAM corrupts data" from "the transformer math is wrong".
// If output is fluent in this mode but garbage with PSRAM, it's PSRAM.
#ifdef KV_CACHE_IN_SRAM
#ifndef KV_DEBUG_MAX_SEQ
#define KV_DEBUG_MAX_SEQ 24
#endif
#endif

// Counters so corruption is reported as a number instead of being invisible.
static uint32_t kv_read_retries = 0;
static uint32_t kv_read_failures = 0;
static uint32_t kv_write_retries = 0;

void kv_stats_print(void) {
    printf("[KV] write retries=%lu  read retries=%lu  read HARD FAILURES=%lu\n",
           (unsigned long)kv_write_retries, (unsigned long)kv_read_retries,
           (unsigned long)kv_read_failures);
}

// A trailing checksum is stored after each cached vector, so a corrupted read
// is *detected* rather than guessed at. The previous magnitude-only check could
// only catch exponent-bit corruption; a mantissa bit flip yields a plausible
// small float that passed straight through it. This catches any bit flip.
#ifndef KV_CACHE_IN_SRAM
static uint32_t kv_checksum(const float* v, int n) {
    const uint8_t* b = (const uint8_t*)v;
    uint32_t h = 2166136261u; // FNV-1a
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h ? h : 1u; // never 0, so an all-zero read can't look valid
}
#endif

static inline void kv_write(psram_spi_inst_t* spi, RunState* s, uint32_t addr, const float* v, int n) {
#ifdef KV_CACHE_IN_SRAM
    (void)spi;
    memcpy(s->sram_kv + addr, v, (size_t)n * sizeof(float));
#else
    (void)s;
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
            kv_write_retries++;
        }
        src += chunk; a += chunk; remaining -= chunk;
    }
    // store the checksum immediately after the vector
    uint32_t sum = kv_checksum(v, n);
    uint32_t sum_addr = addr + (uint32_t)((size_t)n * sizeof(float));
    for (int attempt = 0; attempt < 5; attempt++) {
        uint32_t back = 0;
        psram_write(spi, sum_addr, (const uint8_t*)&sum, sizeof(sum));
        psram_read(spi, sum_addr, (uint8_t*)&back, sizeof(back));
        if (back == sum) break;
        kv_write_retries++;
    }
#endif
}

static inline void kv_read(psram_spi_inst_t* spi, RunState* s, uint32_t addr, float* v, int n) {
#ifdef KV_CACHE_IN_SRAM
    (void)spi;
    memcpy(v, s->sram_kv + addr, (size_t)n * sizeof(float));
#else
    (void)s;
    uint32_t sum_addr = addr + (uint32_t)((size_t)n * sizeof(float));
    for (int attempt = 0; attempt < 4; attempt++) {
        uint8_t* dst = (uint8_t*)v;
        size_t remaining = (size_t)n * sizeof(float);
        uint32_t a = addr;
        while (remaining > 0) {
            size_t chunk = remaining < PSRAM_MAX_READ_CHUNK ? remaining : PSRAM_MAX_READ_CHUNK;
            psram_read(spi, a, dst, chunk);
            dst += chunk; a += chunk; remaining -= chunk;
        }
        uint32_t stored = 0;
        psram_read(spi, sum_addr, (uint8_t*)&stored, sizeof(stored));
        if (stored == kv_checksum(v, n)) return; // verified good
        kv_read_retries++;
    }
    kv_read_failures++;
    memset(v, 0, (size_t)n * sizeof(float));
#endif
}

// One-shot diagnostic: tells us exactly where a NaN/Inf first appears.
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

// Same as matmul(), but weights are int8 with one fp32 scale per group_size
// contiguous elements (v2 quantized format). Dequantizes on the fly and
// multiplies against the (still fp32) activation vector - simpler and just
// as correct as int8xint8 dot products, at some cost in raw speed.
// PERF: the obvious formulation - `sum += q[k] * scale[k / group_size] * x[j]`
// with a 64-bit k - costs one __aeabi_ldivmod library call per multiply-accumulate
// on Cortex-M33 (no hardware 64-bit divide), i.e. millions of libcalls per token.
// Instead we exploit the fact that quantization groups are contiguous over the
// flattened weight array: when n is a multiple of group_size every row starts on
// a group boundary, so we can walk group-by-group with plain pointer bumps. No
// division, no 64-bit math in the hot path, and the scale multiply is hoisted to
// once per group instead of once per element.
static void matmul_q8(float* out, const float* x, const int8_t* q, const float* scale,
                       int n, int d, int group_size) {
    if (group_size > 0 && n % group_size == 0) {
        int groups_per_row = n / group_size;
        const int8_t* qrow = q;
        const float* srow = scale;
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            const float* xp = x;
            for (int g = 0; g < groups_per_row; g++) {
                float acc = 0.0f;
                for (int j = 0; j < group_size; j++) acc += (float)qrow[j] * xp[j];
                sum += acc * srow[g]; // one scale multiply per GROUP, not per element
                qrow += group_size;
                xp += group_size;
            }
            srow += groups_per_row;
            out[i] = sum;
        }
        return;
    }
    // Fallback for the unusual case where rows don't align to group boundaries.
    // Still avoids division by tracking the group index incrementally.
    size_t k = 0, in_group = 0, g = 0;
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += ((float)q[k] * scale[g]) * x[j];
            k++;
            if (++in_group == (size_t)group_size) { in_group = 0; g++; }
        }
        out[i] = sum;
    }
}

// Dispatches to matmul() or matmul_q8() depending on whether q is non-NULL.
// Lets forward() call one thing at each weight site regardless of format.
static void linear(float* out, const float* x, const float* w_fp32,
                    const int8_t* w_q, const float* w_scale, int group_size, int n, int d) {
    if (w_q) {
        matmul_q8(out, x, w_q, w_scale, n, d, group_size);
    } else {
        matmul(out, x, w_fp32, n, d);
    }
}

// Dequantizes `n` consecutive elements starting at flat offset `row*n` from a
// quantized tensor - used for the embedding lookup (one row, not a full matmul).
static void dequant_row(float* out, const int8_t* q, const float* scale, int64_t row, int n, int group_size) {
    size_t base = (size_t)row * (size_t)n;
    const int8_t* qp = q + base;
    if (group_size > 0 && n % group_size == 0) {
        // row starts on a group boundary - walk groups, no division
        const float* sp = scale + base / (size_t)group_size;
        int groups = n / group_size;
        for (int g = 0; g < groups; g++) {
            float s = sp[g];
            for (int j = 0; j < group_size; j++) out[g * group_size + j] = (float)qp[g * group_size + j] * s;
        }
        return;
    }
    size_t g = base / (size_t)group_size, in_group = base % (size_t)group_size;
    for (int j = 0; j < n; j++) {
        out[j] = (float)qp[j] * scale[g];
        if (++in_group == (size_t)group_size) { in_group = 0; g++; }
    }
}

// ---- init -------------------------------------------------------------

#define AK42_MAGIC 0x616b3432u

// Advances *ptr past one quantized tensor (numel int8 values + numel/group_size
// fp32 scales) and returns pointers to its start, matching version2_export's
// serialize_int8() then serialize_fp32(scale) order exactly.
static void take_quantized(const uint8_t** ptr, int64_t numel, int group_size,
                            const int8_t** out_q, const float** out_s) {
    *out_q = (const int8_t*)(*ptr);
    *ptr += numel;
    *out_s = (const float*)(*ptr);
    *ptr += (numel / group_size) * (int64_t)sizeof(float);
}

void transformer_init(Transformer* t, const uint8_t* blob) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    memset(w, 0, sizeof(*w)); // ensures the unused pointer set (fp32 or quantized) is NULL

    uint32_t magic;
    memcpy(&magic, blob, 4);

    const uint8_t* body;

    if (magic == AK42_MAGIC) {
        int32_t version;
        memcpy(&version, blob + 4, 4);
        int32_t hdr[7];
        memcpy(hdr, blob + 8, sizeof(hdr));
        p->dim = hdr[0]; p->hidden_dim = hdr[1]; p->n_layers = hdr[2]; p->n_heads = hdr[3];
        p->n_kv_heads = hdr[4]; p->vocab_size = hdr[5]; p->seq_len = hdr[6];
        uint8_t shared_classifier_byte;
        memcpy(&shared_classifier_byte, blob + 36, 1);
        p->shared_classifier = shared_classifier_byte;
        p->quantized = (version == 2);
        p->group_size = 0;
        if (version == 2) memcpy(&p->group_size, blob + 37, 4);
        body = blob + 256; // v1/v2 header is always padded to exactly 256 bytes
    } else {
        // legacy v0: no magic, header is just the 7 ints at offset 0, and
        // vocab_size's sign doubles as the shared-classifier flag
        int32_t hdr[7];
        memcpy(hdr, blob, sizeof(hdr));
        p->dim = hdr[0]; p->hidden_dim = hdr[1]; p->n_layers = hdr[2]; p->n_heads = hdr[3];
        p->n_kv_heads = hdr[4];
        int32_t vocab_raw = hdr[5];
        p->seq_len = hdr[6];
        p->shared_classifier = vocab_raw > 0;
        p->vocab_size = p->shared_classifier ? vocab_raw : -vocab_raw;
        p->quantized = 0;
        p->group_size = 0;
        body = blob + sizeof(hdr); // legacy header is exactly 28 bytes, no padding
    }

    int64_t dim = p->dim, hidden_dim = p->hidden_dim, n_layers = p->n_layers, vocab_size = p->vocab_size;
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    if (!p->quantized) {
        const float* ptr = (const float*)body;
        w->token_embedding_table = ptr; ptr += vocab_size * dim;
        w->rms_att_weight = ptr;        ptr += n_layers * dim;
        w->wq = ptr;                    ptr += n_layers * dim * (p->n_heads * head_size);
        w->wk = ptr;                    ptr += n_layers * dim * kv_dim;
        w->wv = ptr;                    ptr += n_layers * dim * kv_dim;
        w->wo = ptr;                    ptr += n_layers * (p->n_heads * head_size) * dim;
        w->rms_ffn_weight = ptr;        ptr += n_layers * dim;
        w->w1 = ptr;                    ptr += n_layers * dim * hidden_dim;
        w->w2 = ptr;                    ptr += n_layers * hidden_dim * dim;
        w->w3 = ptr;                    ptr += n_layers * dim * hidden_dim;
        w->rms_final_weight = ptr;      ptr += dim;
        ptr += p->seq_len * head_size / 2; // legacy freq_cis_real placeholder (unused, RoPE computed on the fly)
        ptr += p->seq_len * head_size / 2; // legacy freq_cis_imag placeholder
        w->wcls = p->shared_classifier ? w->token_embedding_table : ptr;
    } else {
        // v2 quantized: fp32 norms first, then quantized tensors in the exact
        // order version2_export() writes them.
        const uint8_t* ptr = body;
        w->rms_att_weight = (const float*)ptr;  ptr += n_layers * dim * sizeof(float);
        w->rms_ffn_weight = (const float*)ptr;  ptr += n_layers * dim * sizeof(float);
        w->rms_final_weight = (const float*)ptr; ptr += dim * sizeof(float);

        int gs = p->group_size;
        take_quantized(&ptr, vocab_size * dim, gs, &w->q_token_embedding_table, &w->s_token_embedding_table);

        w->q_wq = malloc(n_layers * sizeof(int8_t*)); w->s_wq = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * dim, gs, &w->q_wq[l], &w->s_wq[l]);

        w->q_wk = malloc(n_layers * sizeof(int8_t*)); w->s_wk = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * kv_dim, gs, &w->q_wk[l], &w->s_wk[l]);

        w->q_wv = malloc(n_layers * sizeof(int8_t*)); w->s_wv = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * kv_dim, gs, &w->q_wv[l], &w->s_wv[l]);

        w->q_wo = malloc(n_layers * sizeof(int8_t*)); w->s_wo = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * dim, gs, &w->q_wo[l], &w->s_wo[l]);

        w->q_w1 = malloc(n_layers * sizeof(int8_t*)); w->s_w1 = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * hidden_dim, gs, &w->q_w1[l], &w->s_w1[l]);

        w->q_w2 = malloc(n_layers * sizeof(int8_t*)); w->s_w2 = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, hidden_dim * dim, gs, &w->q_w2[l], &w->s_w2[l]);

        w->q_w3 = malloc(n_layers * sizeof(int8_t*)); w->s_w3 = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) take_quantized(&ptr, dim * hidden_dim, gs, &w->q_w3[l], &w->s_w3[l]);

        if (!p->shared_classifier) {
            take_quantized(&ptr, vocab_size * dim, gs, &w->q_wcls, &w->s_wcls);
        } else {
            w->q_wcls = w->q_token_embedding_table;
            w->s_wcls = w->s_token_embedding_table;
        }
    }

    RunState* s = &t->state;

#ifdef KV_CACHE_IN_SRAM
    // Clamp AFTER weight-pointer setup (the v0 layout uses seq_len for the
    // legacy freq_cis placeholders, so clamping earlier would misalign it).
    if (p->seq_len > KV_DEBUG_MAX_SEQ) {
        printf("[KV-SRAM DEBUG] clamping seq_len %d -> %d to fit cache in SRAM\n",
               p->seq_len, KV_DEBUG_MAX_SEQ);
        p->seq_len = KV_DEBUG_MAX_SEQ;
    }
    s->kv_slot_bytes = (uint32_t)((size_t)kv_dim * sizeof(float)); // no checksum needed in SRAM
#else
    s->kv_slot_bytes = (uint32_t)((size_t)kv_dim * sizeof(float) + sizeof(uint32_t));
#endif

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

    size_t one_cache = (size_t)p->n_layers * p->seq_len * s->kv_slot_bytes;
    s->key_cache_base = 0;
    s->value_cache_base = (uint32_t)one_cache;

#ifdef KV_CACHE_IN_SRAM
    s->sram_kv = malloc(one_cache * 2);
    printf("[KV-SRAM DEBUG] KV cache in SRAM: %u bytes %s\n",
           (unsigned)(one_cache * 2), s->sram_kv ? "OK" : "*** ALLOC FAILED ***");
    if (!s->sram_kv) {
        printf("  lower KV_DEBUG_MAX_SEQ and rebuild\n");
    }
#else
    s->sram_kv = NULL;
    printf("KV cache in PSRAM: %u bytes\n", (unsigned)(one_cache * 2));
#endif
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

    if (p->quantized) {
        dequant_row(s->x, w->q_token_embedding_table, w->s_token_embedding_table, token, dim, p->group_size);
    } else {
        memcpy(s->x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));
    }

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + (size_t)l * dim, dim);

        int gs = p->group_size;

        linear(s->q, s->xb,
               w->wq ? w->wq + (size_t)l * dim * dim : NULL,
               w->q_wq ? w->q_wq[l] : NULL,
               w->q_wq ? w->s_wq[l] : NULL,
               gs, dim, dim);
        // k/v go straight into small local buffers before being written to PSRAM
        float* k_local = s->keyt;
        float* v_local = s->valt;
        linear(k_local, s->xb,
               w->wk ? w->wk + (size_t)l * dim * kv_dim : NULL,
               w->q_wk ? w->q_wk[l] : NULL,
               w->q_wk ? w->s_wk[l] : NULL,
               gs, dim, kv_dim);
        linear(v_local, s->xb,
               w->wv ? w->wv + (size_t)l * dim * kv_dim : NULL,
               w->q_wv ? w->q_wv[l] : NULL,
               w->q_wv ? w->s_wv[l] : NULL,
               gs, dim, kv_dim);
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

        uint32_t slot = s->kv_slot_bytes;
        uint32_t key_addr   = s->key_cache_base   + (uint32_t)(l * p->seq_len + pos) * slot;
        uint32_t value_addr = s->value_cache_base + (uint32_t)(l * p->seq_len + pos) * slot;
        kv_write(psram, s, key_addr, k_local, kv_dim);
        kv_write(psram, s, value_addr, v_local, kv_dim);

        // Pass 1: attention scores for ALL heads, reading each cached position once
        for (int tstep = 0; tstep <= pos; tstep++) {
            uint32_t kaddr = s->key_cache_base + (uint32_t)(l * p->seq_len + tstep) * slot;
            kv_read(psram, s, kaddr, s->keyt, kv_dim);
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
            uint32_t vaddr = s->value_cache_base + (uint32_t)(l * p->seq_len + tstep) * slot;
            kv_read(psram, s, vaddr, s->valt, kv_dim);
            for (int hh = 0; hh < p->n_heads; hh++) {
                int kvh = hh / kv_mul;
                float a = s->att[hh * p->seq_len + tstep];
                float* xbh = s->xb + hh * head_size;
                const float* vh = s->valt + kvh * head_size;
                for (int i = 0; i < head_size; i++) xbh[i] += a * vh[i];
            }
        }

        linear(s->xb2, s->xb,
               w->wo ? w->wo + (size_t)l * dim * dim : NULL,
               w->q_wo ? w->q_wo[l] : NULL,
               w->q_wo ? w->s_wo[l] : NULL,
               gs, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        check_finite("x_after_attn", s->x, dim, l, pos);

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (size_t)l * dim, dim);
        linear(s->hb, s->xb,
               w->w1 ? w->w1 + (size_t)l * dim * hidden_dim : NULL,
               w->q_w1 ? w->q_w1[l] : NULL,
               w->q_w1 ? w->s_w1[l] : NULL,
               gs, dim, hidden_dim);
        linear(s->hb2, s->xb,
               w->w3 ? w->w3 + (size_t)l * dim * hidden_dim : NULL,
               w->q_w3 ? w->q_w3[l] : NULL,
               w->q_w3 ? w->s_w3[l] : NULL,
               gs, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float v = s->hb[i];
            v *= (1.0f / (1.0f + expf(-v))); // SiLU
            s->hb[i] = v * s->hb2[i];
        }
        linear(s->xb, s->hb,
               w->w2 ? w->w2 + (size_t)l * hidden_dim * dim : NULL,
               w->q_w2 ? w->q_w2[l] : NULL,
               w->q_w2 ? w->s_w2[l] : NULL,
               gs, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
        check_finite("x_after_ffn", s->x, dim, l, pos);
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    linear(s->logits, s->x, w->wcls, w->q_wcls, w->s_wcls, p->group_size, dim, p->vocab_size);
    check_finite("logits", s->logits, p->vocab_size, p->n_layers, pos);
    return s->logits;
}
