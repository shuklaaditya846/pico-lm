#pragma once
#include <stdint.h>
#include <stddef.h>
#include "psram_spi.h"

// Standard llama2.c export.py header: 7x int32
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

// Pointers straight into the flash-embedded model blob (XIP, read-only, "free" mmap)
typedef struct {
    const float* token_embedding_table; // (vocab_size, dim)
    const float* rms_att_weight;        // (layer, dim)
    const float* wq;                    // (layer, dim, n_heads*head_size)
    const float* wk;                    // (layer, dim, n_kv_heads*head_size)
    const float* wv;                    // (layer, dim, n_kv_heads*head_size)
    const float* wo;                    // (layer, n_heads*head_size, dim)
    const float* rms_ffn_weight;        // (layer, dim)
    const float* w1;                    // (layer, dim, hidden_dim)
    const float* w2;                    // (layer, hidden_dim, dim)
    const float* w3;                    // (layer, dim, hidden_dim)
    const float* rms_final_weight;      // (dim,)
    const float* wcls;                  // (vocab_size, dim) - == token_embedding_table if shared
} TransformerWeights;

// Small per-step scratch buffers live in SRAM (they're tiny).
// The KV cache is the one thing that doesn't fit in SRAM for anything but a
// toy model, so it lives in PSRAM instead, addressed as byte offsets.
typedef struct {
    float* x;      // dim
    float* xb;     // dim
    float* xb2;    // dim
    float* hb;     // hidden_dim
    float* hb2;    // hidden_dim
    float* q;      // dim (n_heads*head_size)
    float* keyt;   // kv_dim  (scratch for one PSRAM read)
    float* valt;   // kv_dim  (scratch for one PSRAM read)
    float* att;    // n_heads * seq_len
    float* logits; // vocab_size
    uint32_t key_cache_base;   // PSRAM byte offset, size = n_layers*seq_len*kv_dim*4
    uint32_t value_cache_base; // PSRAM byte offset, same size, placed right after
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

// blob must point at the start of a llama2.c export.py .bin file that has
// been embedded into flash (see CMakeLists.txt objcopy rule).
void transformer_init(Transformer* t, const uint8_t* model_blob);

float* transformer_forward(Transformer* t, psram_spi_inst_t* psram, int token, int pos);
