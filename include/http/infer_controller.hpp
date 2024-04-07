#ifndef INFER_CONTROLLER_H
#define INFER_CONTROLLER_H

#include "http/server.hpp"
#include "http/controller.hpp"
#include "llm/llm.hpp"
#include "llm/base_llm.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"
#include "absl/strings/str_cat.h"

class EventDispatcher
{
public:
    EventDispatcher()
    {
    }

    void wait_event(httplib::DataSink *sink)
    {
        std::unique_lock<std::mutex> lk(m_);
        int id = id_;
        cv_.wait(lk, [&]
        { return cid_ == id; });
        if (message_.empty())
        {
            sink->done();
            return;
        }
        sink->write(message_.data(), message_.size());
    }

    void send_event(const std::string &message)
    {
        std::lock_guard<std::mutex> lk(m_);
        cid_ = id_++;
        message_ = message;
        cv_.notify_one();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic_int id_{0};
    std::atomic_int cid_{-1};
    std::string message_;
};

class InferController : public httplib::BaseController
{
  public:
    explicit InferController(BaseLlm* llm_handel);
    explicit InferController();
    ~InferController();
  private:
    void bind() override;
    void inferStartApi(const httplib::Request& req, httplib::Response& resp);
    void eventStartApi(const httplib::Request& req, httplib::Response& resp);
    bool inferParamCorrect(const std::string &requestBody, std::string &system, std::string &user);

    BaseLlm* llm_handel_;
    std::unique_ptr<EventDispatcher> ed_;

    static void onTokenCallBack(std::string &token, void* userdata);
    static void onTokenFinish(void* userdata);
};

#endif // INFER_CONTROLLER_H
