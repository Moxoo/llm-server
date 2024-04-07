//
// Created by maxu on 24-3-22.
//

#ifndef BASE_LLM_HPP
#define BASE_LLM_HPP
#include <memory>
#include <atomic>

#include "string"
#include "vector"
#include "tokenizer.hpp"
#include <functional>
#include "http/controller.hpp"

struct Summary
{
    std::string out;
    size_t total_token{0};
    size_t prompt_token{0};
    size_t new_token{0};
    double total_time{0};
    double prefill_time{0};
    double decode_time{0};
    double total_speed{0};
    double prefill_speed{0};
    double decode_speed{0};
    double chat_speed{0};
};

class BaseLlm
{
public:
    BaseLlm()
        : waiting_(false)
    { tokenizer_ = std::make_unique<Sentencepiece>(); }
    virtual ~BaseLlm() = default;

    /**
     * @brief: 加载大模型
     * @author: xu1.ma
     */
    virtual void load(const std::string &model_dir) = 0;

    /**
     * @brief: 大模型推理
     * @author: xu1.ma
     */
    virtual std::string infer(const std::string &system_str, const std::string &user_str) = 0;

    /**
     * @brief: 大模型异步推理
     * @author: xu1.ma
     */
    virtual void asynInfer(const std::string &system_str, const std::string &user_str, void *userdata, std::function<void(std::string &, void *)> cb, std::function<void(void *)> cbF) = 0;

    /**
     * @brief: 把prompt编码为token序列
     * @author: xu1.ma
     */
    virtual std::vector<int> tokenizer(const std::string &system, const std::string &query) = 0;

    /**
     * @brief: 模型是否在推理
     * @author: xu1.ma
     */
    bool onIdle()
    { return !waiting_.load(); }


protected:
    std::atomic<bool> waiting_;
    std::string model_dir_;
    std::unique_ptr<Tokenizer> tokenizer_;

    std::function<void(std::string &, void *)> callback_;
    std::function<void(void *)> callbackF_;
    httplib::BaseController* infer_;

    Summary sum_;
};

#endif //BASE_LLM_HPP
