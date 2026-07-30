#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct {
    int vocab_size;
    const char** vocab_str; // pointers straight into the flash-embedded tokenizer blob
    int* vocab_len;
    float* vocab_score;
    int max_token_length;
} Tokenizer;

// tok_blob points at the start of a tok*.bin file embedded into flash.
void tokenizer_init(Tokenizer* t, const uint8_t* tok_blob, int vocab_size);

// Encodes text into tokens (prepends BOS=1). Returns number of tokens written.
int tokenizer_encode(Tokenizer* t, const char* text, int* tokens, int max_tokens);

// Decodes token (given previous token, for BOS-leading-space handling) into out.
// out_capacity is the size of out; the piece is truncated (never overflowed) to fit.
// Writes *out_len bytes (no null terminator).
void tokenizer_decode_piece(Tokenizer* t, int prev_token, int token, char* out, size_t out_capacity, int* out_len);

int sample_argmax(const float* logits, int n);

// probindex_scratch must point at an array of n ProbIndex entries (caller-owned,
// allocate once with vocab_size entries and reuse across calls).
typedef struct { float prob; int index; } ProbIndex;

// Full reference sampling: temperature scaling + softmax + top-p (nucleus) truncation.
// temperature <= 0 -> greedy argmax. topp <= 0 or >= 1 -> full-distribution sampling (no truncation).
int sample(float* logits, int n, float temperature, float topp, ProbIndex* probindex_scratch);
