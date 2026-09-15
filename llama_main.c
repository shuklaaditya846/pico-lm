// llama2.c on Pico 2 W: weights read directly from flash (XIP), KV cache in PSRAM.
// Interactive REPL over USB serial: type a prompt, press enter, watch it generate.

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

#define TEMPERATURE 1.0f   // matches `run`'s default
#define TOP_P 0.9f         // matches `run`'s default nucleus sampling cutoff
#define MAX_TOKENS 256     // cap generation length per prompt (<= seq_len from the model header)
#define PROMPT_BUF_SIZE 128

// Blocking line read from USB serial, with basic backspace/echo handling.
static void read_line(char* buf, int size) {
    int i = 0;
    while (i < size - 1) {
        int c = getchar();
        if (c == '\r' || c == '\n') {
            if (i == 0) continue; // swallow a stray leading CR/LF
            break;
        }
        if (c == 8 || c == 127) { // backspace / DEL
            if (i > 0) { i--; printf("\b \b"); fflush(stdout); }
            continue;
        }
        putchar(c); fflush(stdout); // local echo, since most terminals won't do it for you
        buf[i++] = (char)c;
    }
    buf[i] = '\0';
    printf("\n");
}

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

    ProbIndex* probindex = malloc((size_t)p->vocab_size * sizeof(ProbIndex));

    int steps = MAX_TOKENS;
    if (steps > p->seq_len) steps = p->seq_len;

    char prompt[PROMPT_BUF_SIZE];
    int prompt_tokens[PROMPT_BUF_SIZE];
    char piece_buf[64];

    printf("\nType a prompt and press enter. Ctrl+C / reset to stop.\n");

    while (true) {
        printf("\n> ");
        fflush(stdout);
        read_line(prompt, sizeof(prompt));
        if (prompt[0] == '\0') continue;

        int n_prompt = tokenizer_encode(&tok, prompt, prompt_tokens, PROMPT_BUF_SIZE);

        int token = prompt_tokens[0]; // BOS
        int pos = 0;
        int plen;

        while (pos < steps) {
            float* logits = transformer_forward(&transformer, &psram, token, pos);

            int next;
            if (pos < n_prompt - 1) {
                next = prompt_tokens[pos + 1]; // still feeding the prompt back in
            } else {
                next = sample(logits, p->vocab_size, TEMPERATURE, TOP_P, probindex);
            }

            int should_stop = (pos + 1 >= n_prompt) && (next == 1); // BOS reappearing = model's own stop signal
            if (!should_stop) {
                tokenizer_decode_piece(&tok, token, next, piece_buf, sizeof(piece_buf), &plen);
                fwrite(piece_buf, 1, plen, stdout);
                fflush(stdout);
            }

            token = next;
            pos++;
            if (should_stop) break;
        }
        printf("\n");
    }
}
