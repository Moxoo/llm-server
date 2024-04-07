#include <memory>

#include "http/infer_controller.hpp"
#include <chrono>

InferController::InferController(BaseLlm *llm_handel)
    : llm_handel_(llm_handel)
{
}

InferController::InferController()
    : llm_handel_(nullptr)
{
}

InferController::~InferController()
{

}

void InferController::bind()
{
    // BindController可以绑定httplib中所有参数为(const httplib::Request& req, httplib::Response& resp)的方法
    server_->Post("/infer", BindController(&InferController::inferStartApi, this));
    server_->Post("/stream_infer", BindController(&InferController::eventStartApi, this));
}

void InferController::eventStartApi(const httplib::Request &req, httplib::Response &resp)
{
    std::string h1_request_headers_str;
    std::string data = req.body;
    absl::StrAppend(&h1_request_headers_str, req.method, " ", req.path, " ", req.version);
    std::cout << h1_request_headers_str << "\n" << req.body << "\n" << std::endl;

    std::string system;
    std::string user;

    if (!inferParamCorrect(data, system, user))
    {
        std::string jsonData = "{\"type\": \"post\", \"code\": 3001}";
        resp.set_content(jsonData, "application/json");
        return;
    }

    if (llm_handel_->onIdle())
    {
        ed_ = std::make_unique<EventDispatcher>();
        resp.set_chunked_content_provider("text/event-stream", [&](size_t /*offset*/, httplib::DataSink &sink){ed_->wait_event(&sink);return true;});
        std::thread([=](){llm_handel_->asynInfer(system, user, this, &InferController::onTokenCallBack, &InferController::onTokenFinish);}).detach();
        return;
    }
    std::string jsonData = "{\"type\": \"post\", \"code\": 4001}";
    resp.set_content(jsonData, "application/json");
}

void InferController::inferStartApi(const httplib::Request &req, httplib::Response &resp)
{
    std::string h1_request_headers_str;
    std::string data = req.body;
    absl::StrAppend(&h1_request_headers_str, req.method, " ", req.path, " ", req.version);
    std::cout << h1_request_headers_str << "\n" << req.body << "\n" << std::endl;

    std::string system;
    std::string user;

    if (!inferParamCorrect(data, system, user))
    {
        std::string jsonData = "{\"type\": \"post\", \"code\": 3001}";
        resp.set_content(jsonData, "application/json");
        return;
    }
    std::string response = "";
    if (llm_handel_->onIdle())
    {
        response = llm_handel_->infer(system, user);
        std::string jsonData = "{\"type\": \"post\", \"code\": 2001, \"response\":" + response + "}";
        resp.set_content(jsonData, "application/json");
        return;
    }
    std::string jsonData = "{\"type\": \"post\", \"code\": 4001}";
    resp.set_content(jsonData, "application/json");
}

bool InferController::inferParamCorrect(const std::string &requestBody, std::string &system, std::string &user)
{
    rapidjson::Document d;
    d.Parse(requestBody.c_str());
    if (d.HasParseError())
    {
        return false;
    }
    if (d.HasMember("prompt") && d["prompt"].IsObject())
    {
        if (d["prompt"].HasMember("system") && d["prompt"]["system"].IsString())
        {
            if (d["prompt"].HasMember("user") && d["prompt"]["user"].IsString())
            {
                system = d["prompt"]["system"].GetString();
                user = d["prompt"]["user"].GetString();
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

void InferController::onTokenCallBack(std::string &token, void *userdata)
{
    auto* data = static_cast<InferController *>(userdata);
    std::stringstream ss;
    ss << "data: " << token << "\n\n";
    data->ed_->send_event(ss.str());
}

void InferController::onTokenFinish(void *userdata)
{
    auto* data = static_cast<InferController *>(userdata);
    data->ed_->send_event("");
}
