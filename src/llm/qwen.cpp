#include "llm/qwen.hpp"

// Qwen_7b
std::vector<int> Qwen_7b::tokenizer(const std::string& system, const std::string& query) {
    // auto prompt = "\n<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = tokenizer_encode(system);
    ids.insert(ids.begin(), {198, 151644, 8948, 198}); // \n<|im_start|>system\n
    ids.insert(ids.end(), {151645, 198});              // <|im_end|>\n
    ids.insert(ids.end(), {151644, 872, 198});         // <|im_start|>user\n
    for (auto &&query_id : tokenizer_encode(query))    // query
    {
        ids.insert(ids.end(), query_id);
    }
    ids.insert(ids.end(), {151645, 198, 151644, 77091, 198}); // <|im_end|>\n<|im_start|>assistant\n
    return ids;
}

VARP Qwen_7b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            ptr[seq_len * i + j] = j <= i;
        }
    }
    return attention_mask;
}

VARP Qwen_7b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = all_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Qwen_7b::is_stop(int token_id) {
    return token_id >= 151645;
}
