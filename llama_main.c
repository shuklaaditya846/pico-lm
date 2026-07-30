// llama2.c on Pico 2 W: weights read directly from flash (XIP), KV cache in PSRAM.
// Equivalent of: ./run stories260K.bin -z tok512.bin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"
#include "psram_spi.h"
#include "model.h"
#include "tokenizer.h"

// Symbols come from objcopy-embedding stories260K.bin / tok512.bin (see CMakeLists.txt)
extern const uint8_t _binary_stories260K_bin_start[];
extern const uint8_t _binary_tok512_bin_start[];

#define PROMPT ""          // empty = model free-runs from BOS, like the CLI with no prompt
#define TEMPERATURE 0.6f   // 0.0f = greedy/deterministic, matches `run`'s default of 1.0 otherwise
#define MAX_TOKENS 256      // cap generation length (<= seq_len from the model header)

int main() {
    stdio_init_all();
    sleep_ms(3000); // give the USB CDC host time to attach so early prints aren't lost

    printf("\n=== llama2.c on Pico 2 W (flash weights + PSRAM KV cache) ===\n");

    psram_spi_inst_t psram = psram_spi_init_clkdiv(pio0, -1, 2.0, true);

    Transformer transformer;
    transformer_init(&transformer, _binary_stories260K_bin_start);
    Config* p = &transformer.config;
    printf("dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads,
           p->vocab_size, p->seq_len);

    Tokenizer tok;
    tokenizer_init(&tok, _binary_tok512_bin_start, p->vocab_size);

    int prompt_tokens[128];
    int n_prompt = tokenizer_encode(&tok, PROMPT, prompt_tokens, 128);

    int steps = MAX_TOKENS;
    if (steps > p->seq_len) steps = p->seq_len;

    printf("\nGenerating (%d steps max):\n\n", steps);

    int token = prompt_tokens[0]; // BOS
    int pos = 0;
    char piece_buf[4];
    int plen;

    while (pos < steps) {
        float* logits = transformer_forward(&transformer, &psram, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1]; // still feeding the prompt
        } else {
            next = sample_temperature(logits, p->vocab_size, TEMPERATURE);
        }

        tokenizer_decode_piece(&tok, token, next, piece_buf, &plen);
        fwrite(piece_buf, 1, plen, stdout);
        fflush(stdout);

        token = next;
        pos++;
        if (next == 1) break; // BOS re-appearing marks end of sequence, same as upstream run.c
    }

    printf("\n\n=== done ===\n");
    while (true) tight_loop_contents();
}
