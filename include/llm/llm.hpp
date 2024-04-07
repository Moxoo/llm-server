#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <streambuf>
#include <functional>
#include <unordered_map>
#include <atomic>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "tokenizer.hpp"
#include "base_llm.hpp"

using namespace MNN;
using namespace Express;
class Tokenizer;

// llm stream buffer with callback

class LlmStreamBuffer: public std::streambuf
{
public:
    using CallBack = std::function<void(const char *str, size_t len)>;;
    LlmStreamBuffer(CallBack callback)
        : callback_(callback)
    {}

protected:
    virtual std::streamsize xsputn(const char *s, std::streamsize n) override
    {
        if (callback_)
        {
            callback_(s, n);
        }
        return n;
    }

private:
    CallBack callback_ = nullptr;
};

class Llm: public BaseLlm
{
public:
    Llm() = default;
    ~Llm() override = default;

    /**
     * @brief: 创建特定的mnn-llm
     * @author: xu1.ma
     */
    static Llm *createLLM(const std::string &path, std::string model_type = "auto");

    /**
     * @brief: 加载mnn-llm模型
     * @author: xu1.ma
     */
    void load(const std::string &model_dir) override;

    /**
     * @brief: 大模型异步推理
     * @author: xu1.ma
     */
    void asynInfer(const std::string &system_str, const std::string &user_str, void *userdata, std::function<void(std::string &, void *)> cb, std::function<void(void *)> cbF) override;

    /**
     * @brief: 推理接口实现
    * @author: xu1.ma
    */
    std::string infer(const std::string &system_str, const std::string &user_str) override;

    /**
     * @brief: 从磁盘中读取嵌入层权重 节省内存
     * @author: xu1.ma
     */
    VARP disk_embedding(const std::vector<int> &input_ids);

    /**
     * @brief: 推理出下一个token
     * @author: xu1.ma
     */
    int forward(const std::vector<int> &input_ids);

    /**
     * @brief: 把句子转成token序列
     * @author: xu1.ma
     */
    std::vector<int> tokenizer_encode(const std::string &input_str);

    /**
     * @brief: 把token解码成词
     * @author: xu1.ma
     */
    std::string decode(int id);

    void chat();
    void warmup();
    std::string response(const std::string &input_str, std::ostream *os = &std::cout, const char *end_with = nullptr);

    float load_progress() const
    { return load_progress_; }
    void reset();
    void print_speed();
public:
    std::string system_prompt_;
    std::vector<int> history_;
    // forward info
    int max_seq_len_ = 1024;
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
private:
    virtual VARP gen_attention_mask(int seq_len) = 0;
    virtual VARP gen_position_ids(int seq_len) = 0;
    virtual bool is_stop(int token_id) = 0;
protected:
    // model configs
    bool is_single_ = false;
    int layer_nums_ = 0;
    int hidden_size_ = 4096;
    std::vector<int> key_value_shape_ = {};
    std::string model_name_ = "";
    // gen info
    float load_progress_ = 0.f;
private:
    // MNN Modules
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<VARP> past_key_values_;
    // model dir
};

#endif // LLM_hpp
