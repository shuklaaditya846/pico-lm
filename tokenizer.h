#pragma once
#include <stdint.h>

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
// out must be at least 4 bytes. Writes *out_len bytes (no null terminator).
void tokenizer_decode_piece(Tokenizer* t, int prev_token, int token, char* out, int* out_len);

int sample_argmax(const float* logits, int n);
int sample_temperature(float* logits, int n, float temperature);
