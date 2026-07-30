#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void tokenizer_init(Tokenizer* t, const uint8_t* blob, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab_str = malloc((size_t)vocab_size * sizeof(char*));
    t->vocab_len = malloc((size_t)vocab_size * sizeof(int));
    t->vocab_score = malloc((size_t)vocab_size * sizeof(float));

    const uint8_t* p = blob;
    int32_t maxlen;
    memcpy(&maxlen, p, 4); p += 4;
    t->max_token_length = maxlen;

    for (int i = 0; i < vocab_size; i++) {
        float score; int32_t len;
        memcpy(&score, p, 4); p += 4;   // unaligned-safe reads: string lengths make offsets odd
        memcpy(&len, p, 4); p += 4;
        t->vocab_score[i] = score;
        t->vocab_len[i] = len;
        t->vocab_str[i] = (const char*)p;
        p += len;
    }
}

static int str_lookup(const char* str, int len, Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab_len[i] == len && memcmp(str, t->vocab_str[i], len) == 0) return i;
    }
    return -1;
}

int tokenizer_encode(Tokenizer* t, const char* text, int* tokens, int max_tokens) {
    int n = 0;
    if (max_tokens > 0) tokens[n++] = 1; // BOS

    for (const char* c = text; *c != '\0' && n < max_tokens; c++) {
        int id = str_lookup(c, 1, t);
        if (id != -1) {
            tokens[n++] = id;
        } else {
            tokens[n++] = (unsigned char)(*c) + 3; // raw byte fallback token
        }
    }

    // greedy BPE merge
    while (n > 1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < n - 1; i++) {
            char merge_buf[64];
            int l1 = t->vocab_len[tokens[i]];
            int l2 = t->vocab_len[tokens[i + 1]];
            if (l1 + l2 >= (int)sizeof(merge_buf)) continue;
            memcpy(merge_buf, t->vocab_str[tokens[i]], l1);
            memcpy(merge_buf + l1, t->vocab_str[tokens[i + 1]], l2);
            int id = str_lookup(merge_buf, l1 + l2, t);
            if (id != -1 && t->vocab_score[id] > best_score) {
                best_score = t->vocab_score[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
        n--;
    }
    return n;
}

void tokenizer_decode_piece(Tokenizer* t, int prev_token, int token, char* out, int* out_len) {
    const char* piece = t->vocab_str[token];
    int len = t->vocab_len[token];

    if (prev_token == 1 && len > 0 && piece[0] == ' ') { piece++; len--; } // strip sentencepiece dummy space after BOS

    // raw byte token looks like "<0xAB>" (len == 6)
    if (len == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        char hex[3] = { piece[3], piece[4], 0 };
        out[0] = (char)strtol(hex, NULL, 16);
        *out_len = 1;
        return;
    }

    memcpy(out, piece, len);
    *out_len = len;
}

int sample_argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

static uint32_t rng_state = 0x2545F491u;
static float randf(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (rng_state >> 8) / 16777216.0f;
}

int sample_temperature(float* logits, int n, float temperature) {
    if (temperature <= 0.0f) return sample_argmax(logits, n);
    for (int i = 0; i < n; i++) logits[i] /= temperature;
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - maxv); sum += logits[i]; }
    for (int i = 0; i < n; i++) logits[i] /= sum;
    float r = randf(), cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += logits[i]; if (r < cdf) return i; }
    return n - 1;
}
