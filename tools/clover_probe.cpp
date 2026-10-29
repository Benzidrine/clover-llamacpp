// Feed one block of `block` slots: prompt tokens followed by mask-token padding, run a
// single non-causal (block-diffusion preview) forward, and print the top prediction at
// each prompt position. Used to check the preview path against the reference model
// without running a full 128-token denoising loop.

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

static bool dump_enabled = false;

static bool cb_eval(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!dump_enabled || !t->name[0]) {
        return false;
    }
    const char * n = t->name;
    if (strncmp(n, "l_out", 5) != 0 && strcmp(n, "result_norm") != 0 && strcmp(n, "result_output") != 0) {
        return false;
    }
    if (ask) {
        return true;
    }

    const int64_t ne = ggml_nelements(t);
    std::vector<float> buf(ne);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, ne * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(ne);
        ggml_backend_tensor_get(t, raw.data(), 0, ne * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < ne; i++) { buf[i] = ggml_fp16_to_fp32(raw[i]); }
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw(ne);
        ggml_backend_tensor_get(t, raw.data(), 0, ne * sizeof(ggml_bf16_t));
        for (int64_t i = 0; i < ne; i++) { buf[i] = ggml_bf16_to_fp32(raw[i]); }
    } else {
        return true;
    }

    double sum = 0.0, sum2 = 0.0;
    float vmin = 1e30f, vmax = -1e30f;
    for (int64_t i = 0; i < ne; i++) {
        sum  += buf[i];
        sum2 += (double) buf[i] * buf[i];
        vmin = std::min(vmin, buf[i]);
        vmax = std::max(vmax, buf[i]);
    }
    const double mean = sum / ne;
    const double var  = sum2 / ne - mean * mean;
    fprintf(stderr, "DUMP %-12s ne=%6lld mean=%9.4f std=%8.4f min=%9.3f max=%9.3f\n",
            n, (long long) t->ne[0] * t->ne[1], mean, std::sqrt(std::max(0.0, var)), vmin, vmax);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf 'prompt text' [n_block] [topk]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const int    n_block    = argc > 3 ? atoi(argv[3]) : 32;
    const int    topk       = argc > 4 ? atoi(argv[4]) : 8;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 128;
    cparams.n_batch   = n_block;
    cparams.n_ubatch  = n_block;
    cparams.n_seq_max = 1;

    dump_enabled = getenv("PROBE_DUMP") && atoi(getenv("PROBE_DUMP")) != 0;
    if (dump_enabled) {
        cparams.cb_eval           = cb_eval;
        cparams.cb_eval_user_data = nullptr;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    // Block-diffusion preview is bidirectional within the block; PROBE_CAUSAL=1 selects the
    // ordinary causal GDN path as a control.
    const bool causal = getenv("PROBE_CAUSAL") && atoi(getenv("PROBE_CAUSAL")) != 0;
    llama_set_causal_attn(ctx, causal);

    const int plen = strlen(prompt);
    std::vector<llama_token> tokens(n_block + 16);
    int n_tok = llama_tokenize(vocab, prompt, plen, tokens.data(), tokens.size(), true, true);
    if (n_tok < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    tokens.resize(n_tok);

    fprintf(stderr, "prompt tokens (%d):", n_tok);
    for (auto t : tokens) {
        fprintf(stderr, " %d", t);
    }
    fprintf(stderr, "\n");

    if (n_tok > n_block) {
        fprintf(stderr, "prompt does not fit in one block\n");
        return 1;
    }

    const llama_token mask_id = llama_vocab_mask(vocab);
    // Padding slot contents. Using the mask token is the true diffusion case; a real token
    // gives identical input to both paths so their layer statistics can be compared.
    llama_token pad_id = mask_id;
    if (getenv("PROBE_PAD_ID")) {
        pad_id = (llama_token) atoi(getenv("PROBE_PAD_ID"));
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    char piece[256];

    // Reference arrangement: commit the prompt causally, then preview one block of mask
    // tokens at positions continuing after it.
    if (getenv("PROBE_PREFILL")) {
        llama_set_causal_attn(ctx, true);
        struct llama_batch_ext * b1 = llama_batch_ext_init(ctx);
        for (int i = 0; i < n_tok; i++) {
            llama_pos pos = i;
            const int32_t idx = llama_batch_ext_add_token(b1, 0, tokens[i]);
            llama_batch_ext_set_pos(b1, idx, &pos);
            llama_batch_ext_set_output_logits(b1, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b1) != 0) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }
        llama_batch_ext_free(b1);

        llama_set_causal_attn(ctx, false);
        struct llama_batch_ext * b2 = llama_batch_ext_init(ctx);
        for (int i = 0; i < n_block; i++) {
            llama_pos pos = n_tok + i;
            const int32_t idx = llama_batch_ext_add_token(b2, 0, pad_id);
            llama_batch_ext_set_pos(b2, idx, &pos);
            llama_batch_ext_set_output_logits(b2, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b2) != 0) {
            fprintf(stderr, "preview failed\n");
            return 1;
        }
        llama_batch_ext_free(b2);

        const int n_show = std::min(n_block, 12);
        for (int i = 0; i < n_show; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            std::vector<int> order(n_vocab);
            for (int v = 0; v < n_vocab; v++) {
                order[v] = v;
            }
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
            fprintf(stdout, "gen %2d:", i);
            for (int k = 0; k < topk; k++) {
                const int id = order[k];
                int l = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, false);
                if (l < 0) { l = 0; }
                piece[l] = 0;
                std::string s;
                for (int j = 0; j < l; j++) {
                    const unsigned char c = (unsigned char) piece[j];
                    if (c == '\n')     { s += "\\n"; }
                    else if (c < 0x20) { char bb[8]; snprintf(bb, sizeof(bb), "\\x%02x", c); s += bb; }
                    else               { s += piece[j]; }
                }
                fprintf(stdout, "  [%d %s %.2f]", id, s.c_str(), logits[id]);
            }
            fprintf(stdout, "\n");
        }
        fflush(stdout);

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    const int n_feed = causal ? n_tok : n_block;

    struct llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (int i = 0; i < n_feed; i++) {
        const llama_token t = i < n_tok ? tokens[i] : pad_id;
        llama_pos pos = i;
        const int32_t idx = llama_batch_ext_add_token(batch, 0, t);
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, idx, true);
    }

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    for (int i = 0; i < n_tok; i++) {
        const float * logits = llama_get_logits_ith(ctx, i);
        std::vector<int> order(n_vocab);
        for (int v = 0; v < n_vocab; v++) {
            order[v] = v;
        }
        std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });

        fprintf(stdout, "pos %2d (tok %6d):", i, tokens[i]);
        for (int k = 0; k < topk; k++) {
            const int id = order[k];
            int l = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, false);
            if (l < 0) { l = 0; }
            piece[l] = 0;
            // make control characters visible
            std::string s;
            for (int j = 0; j < l; j++) {
                const unsigned char c = (unsigned char) piece[j];
                if (c == '\n')      { s += "\\n"; }
                else if (c < 0x20)  { char b[8]; snprintf(b, sizeof(b), "\\x%02x", c); s += b; }
                else                { s += piece[j]; }
            }
            fprintf(stdout, "  [%d %s %.2f]", id, s.c_str(), logits[id]);
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
