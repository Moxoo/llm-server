#include "llm/llm.hpp"
#include "http/server.hpp"
#include "http/infer_controller.hpp"
#include "cmdline/cmdline.h"
#include <iostream>
#include "llm/base_llm.hpp"
#include <thread>

int main(int argc, char *argv[])
{
    cmdline::parser line;
    line.add<std::string>("platform", 'f', "nn platform: mnn", false, "mnn");
    line.add<std::string>("model", 'm', "model folder", false, "../model/qwen-1.8b-int8-chat");
    line.add<std::string>("host", 'h', "http server host(default 0.0.0.0)", false, "0.0.0.0");
    line.add<int>("port", 'p', "specify the port(default 8080) to listen on", false, 8080, cmdline::range(1, 65535));
    line.add<int>("worker_num", 'w', "specify the http worker thread number(default 1) to use", true);
    line.parse_check(argc, argv);

    std::string platform = line.get<std::string>("platform");
    std::string model_dir = line.get<std::string>("model");

    std::unique_ptr<BaseLlm> llm_handle = nullptr;

    if (platform == "mnn")
    {
        llm_handle.reset(Llm::createLLM(model_dir));
    }

    if (!llm_handle)
    {
        return -1;
    }

    // http server
    std::string host = line.get<std::string>("host");
    int port = line.get<int>("port");
    int worker_num = line.get<int>("worker_num");

    std::thread([&](){llm_handle->load(model_dir);}).detach();

    std::unique_ptr<httplib::HttpServer> http_server = std::make_unique<httplib::HttpServer>(host, port, worker_num);
    std::shared_ptr<InferController> controller = std::make_shared<InferController>(llm_handle.get());
    http_server->addController(controller);
    http_server->listen();
    return 0;
}
