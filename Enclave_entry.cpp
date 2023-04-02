#include <string>
#include <vector>
#include <random>
#include <mutex>
#include <assert.h>
#include "llama.h"

struct gpt_params {
    int32_t seed          = -1;   // RNG seed
    int32_t n_threads     = 4;
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

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string prompt = "";
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

