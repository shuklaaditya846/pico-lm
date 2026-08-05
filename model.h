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
    int quantized;         // 1 if this is a v2 int8-quantized export, 0 if v0 legacy fp32
    int group_size;        // quantization group size (v2 only)
    int shared_classifier; // 1 if the classifier weight is tied to the embedding table
} Config;

// Pointers straight into the flash-embedded model blob (XIP, read-only, "free" mmap).
// Two parallel sets: fp32 pointers (used for v0 legacy exports, and always for the
// three RMSNorm weight groups which are never quantized), and int8+scale pointers
// (used for v2 quantized exports). Exactly one set is populated per tensor at runtime,
// selected by Config.quantized.
typedef struct {
    // fp32 path
    const float* token_embedding_table; // (vocab_size, dim)
    const float* rms_att_weight;        // (layer, dim) - fp32 in both formats
    const float* wq;                    // (layer, dim, n_heads*head_size)
    const float* wk;                    // (layer, dim, n_kv_heads*head_size)
    const float* wv;                    // (layer, dim, n_kv_heads*head_size)
    const float* wo;                    // (layer, n_heads*head_size, dim)
    const float* rms_ffn_weight;        // (layer, dim) - fp32 in both formats
    const float* w1;                    // (layer, dim, hidden_dim)
    const float* w2;                    // (layer, hidden_dim, dim)
    const float* w3;                    // (layer, dim, hidden_dim)
    const float* rms_final_weight;      // (dim,) - fp32 in both formats
    const float* wcls;                  // (vocab_size, dim) - == token_embedding_table if shared

    // int8 quantized path (v2 only) - each tensor is int8 data + one fp32 scale
    // per group_size contiguous elements (flattened, row-major). The exporter
    // writes each LAYER's tensor as [int8][scale] immediately one after another
    // (not all layers' int8 batched before all scales), so per-layer tensors
    // need one pointer per layer rather than a single base + stride.
    const int8_t* q_token_embedding_table; const float* s_token_embedding_table; // single tensor
    const int8_t** q_wq; const float** s_wq; // [n_layers] each
    const int8_t** q_wk; const float** s_wk;
    const int8_t** q_wv; const float** s_wv;
    const int8_t** q_wo; const float** s_wo;
    const int8_t** q_w1; const float** s_w1;
    const int8_t** q_w2; const float** s_w2;
    const int8_t** q_w3; const float** s_w3;
    const int8_t* q_wcls; const float* s_wcls; // single tensor
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

// blob must point at the start of a llama2.c export.py .bin file (v0 legacy fp32,
// or v2 int8-quantized) embedded into flash (see CMakeLists.txt objcopy rule).
// Format is auto-detected from the magic number, same as upstream run.c/runq.c.
void transformer_init(Transformer* t, const uint8_t* model_blob);

float* transformer_forward(Transformer* t, psram_spi_inst_t* psram, int token, int pos);
