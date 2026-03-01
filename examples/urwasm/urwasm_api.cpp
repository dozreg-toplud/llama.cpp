// examples/memload_api.cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "llama-model.h"


static llama_model  * g_model = nullptr;
static llama_context* g_ctx   = nullptr;

static void ggml_log_shhh(enum ggml_log_level level,
    const char * text,
    void * user_data)
{
    (void)level;
    (void)text;
    (void)user_data;
}

extern "C" void llama_cpp_init_from_gguf_bytes(uint8_t *bytes, uint32_t len)
{
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }

    llama_backend_init();
    llama_log_set(ggml_log_shhh, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap     = false;
    mparams.use_mlock    = false;

    try {
        g_model = llama_model_load_from_bytes(bytes, len, mparams);
        if (g_model == 0) {
            fprintf(stderr, "llama_model_load_from_bytes returned null\r\n");
            return;
        }
    }
    catch (const std::exception & e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return;
    }
    catch (...) {
        fprintf(stderr, "Unknown exception\n");
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    g_ctx = llama_init_from_model(g_model, cparams);
}

static llama_token greedy_next_token(const float * logits, int n_vocab)
{
    int best_i = 0;
    float best = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best) { best = logits[i]; best_i = i; }
    }
    return (llama_token) best_i;
}

extern "C" char *infer_c_string(const char *prompt, int *token_count)
{
    if (!g_model || !g_ctx) return nullptr;

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    
    std::vector<llama_token> tokens;
    tokens.resize(4096);  // XX reconsider limit


    int n = llama_tokenize(vocab, prompt, (int32_t)std::strlen(prompt),
                           tokens.data(), (int32_t)tokens.size(),
                           /*add_special*/ true, /*parse_special*/ true);
    if (n < 0) return nullptr;
    tokens.resize((size_t)n);

    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    for (int i = 0; i < (int)tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i]   = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == (int)tokens.size() - 1);
    }
    batch.n_tokens = (int)tokens.size();

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return nullptr;
    }
    llama_batch_free(batch);

    const int max_new = 64;  // XX reconsider limit
    std::string out;

    int count = 0;

    llama_sampler_chain_params s_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(s_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        64,     // last n tokens to penalize
        1.20f,  // repeat penalty (multiplicative)
        0.00f,  // frequency penalty
        0.00f   // present penalty
    ));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.02f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xcafebabe));

    for (int step = 0; step < max_new; step++) {

        llama_token next = llama_sampler_sample(sampler, g_ctx, -1);
        llama_sampler_accept(sampler, next);
        count++;

        if (next == llama_vocab_eos(vocab)) break;

        // detokenize piece
        char piece[4096];
        int n_piece = llama_token_to_piece(vocab, next, piece, (int)sizeof(piece), 0, true);
        if (n_piece > 0) out.append(piece, piece + n_piece);

        // feed token back
        llama_batch b2 = llama_batch_init(1, 0, 1);
        b2.token[0] = next;
        b2.pos[0]   = (int)tokens.size() + step;
        b2.n_seq_id[0] = 1;
        b2.seq_id[0][0] = 0;
        b2.logits[0] = true;
        b2.n_tokens = 1;

        if (llama_decode(g_ctx, b2) != 0) {
            llama_batch_free(b2);
            break;
        }
        llama_batch_free(b2);
    }

    // return malloc'd C string
    char * cstr = (char*) std::malloc(out.size() + 1);
    if (!cstr) return nullptr;
    std::memcpy(cstr, out.data(), out.size());
    cstr[out.size()] = '\0';
    *token_count = count;
    return cstr;
}