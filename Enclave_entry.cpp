#include <string>
#include <vector>
#include <random>
#include <mutex>
#include <assert.h>
#include "llama.h"

struct gpt_params {
    int32_t n_predict     = 128;  // new tokens to predict
    int32_t repeat_last_n = 64;   // last n tokens to penalize
    int32_t n_parts       = -1;   // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512;  // context size
    int32_t n_batch       = 8;    // batch size for prompt processing
    int32_t n_keep        = 0;    // number of tokens to keep from initial prompt

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    std::string input_prefix = ""; // string to prefix user inputs with


    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_start = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool ignore_eos        = false; // do not stop generating after eos
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

static std::mutex ctx_mutex;
static llama_context *ctx = NULL;

extern "C" int load_model(const char* model_path, int seed) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    if (ctx != NULL) {
        fprintf(stderr, "%s: model already loaded\n", __func__);
        return 0;
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, seed);

    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = 512;
        lparams.n_parts    = -1;
        lparams.seed       = seed;
        lparams.f16_kv     = true;

        ctx = llama_init_from_file(model_path, lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model_path);
            return 1;
        }
    }
    return 0;
}

extern "C" int completion(const char *p_prompt, int n_threads) {
    auto prompt = std::string(p_prompt);

    std::lock_guard<std::mutex> lock(ctx_mutex);
    gpt_params params;
    // Add a space in front of the first character to match OG llama tokenizer behavior
    prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        if (params.n_keep > 0) {
        fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    {
        int n_past     = 0;
        int n_remain   = params.n_predict;
        int n_consumed = 0;

        std::vector<llama_token> embd;

        while (n_remain != 0 || params.interactive) {
            // predict
            if (embd.size() > 0) {
                // infinite text generation via context swapping
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (n_past + (int) embd.size() > n_ctx) {
                    const int n_left = n_past - params.n_keep;

                    n_past = params.n_keep;

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
                }

                if (llama_eval(ctx, embd.data(), embd.size(), n_past, n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
            }

            n_past += embd.size();
            embd.clear();

            if ((int) embd_inp.size() <= n_consumed) {
                // out of user input, sample next token
                const int32_t top_k          = params.top_k;
                const float   top_p          = params.top_p;
                const float   temp           = params.temp;
                const float   repeat_penalty = params.repeat_penalty;

                llama_token id = 0;

                {
                    auto logits = llama_get_logits(ctx);

                    if (params.ignore_eos) {
                        logits[llama_token_eos()] = 0;
                    }

                    id = llama_sample_top_p_top_k(ctx,
                            last_n_tokens.data() + n_ctx - params.repeat_last_n,
                            params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // add it to the context
                embd.push_back(id);

                // decrement remaining sampling budget
                --n_remain;
            } else {
                // some user input remains from prompt, forward it to processing
                while ((int) embd_inp.size() > n_consumed) {
                    embd.push_back(embd_inp[n_consumed]);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[n_consumed]);
                    ++n_consumed;
                    if ((int) embd.size() >= params.n_batch) {
                        break;
                    }
                }
            }

            for (auto id : embd) {
                printf("%s", llama_token_to_str(ctx, id));
            }
        }
        llama_print_timings(ctx);
    }
    return 0;
}
