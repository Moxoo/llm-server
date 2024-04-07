#ifndef QWEN_hpp
#define QWEN_hpp

#include "llm/llm.hpp"

// llm models
class Qwen_7b : public Llm {
public:
    Qwen_7b() {
        model_name_ = "Qwen_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 0, 32, 128};
        hidden_size_ = 4096;
        tokenizer_.reset(new Tiktoken);
    }
private:
    std::vector<int> tokenizer(const std::string& system, const std::string& query) override;
    VARP gen_attention_mask(int seq_len) override;
    VARP gen_position_ids(int seq_len) override;
    bool is_stop(int token_id) override;
};

class Qwen_1_8b : public Qwen_7b {
public:
    Qwen_1_8b() {
        model_name_ = "Qwen_1.8b";
        layer_nums_ = 24;
        key_value_shape_ = {2, 1, 0, 16, 128};
        hidden_size_ = 2048;
        tokenizer_.reset(new Tiktoken);
    }
};

#endif // QWEN_hpp
