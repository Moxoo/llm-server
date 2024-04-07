#ifndef SERVER_H
#define SERVER_H

#include "httplib.h"
#include "controller.hpp"
#include "http_thread_pool.hpp"
#include <functional>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace httplib {
    class BaseController;
    // http server, 用于管理httplib::Server对象
    class HttpServer {
      public:
        HttpServer(std::shared_ptr<httplib::HttpThreadPool> pool, const std::string& host, int port, int workers, int socket_flags = 0);
        HttpServer(std::shared_ptr<httplib::HttpThreadPool> pool, int port, int workers, int socket_flags = 0);
        HttpServer(const std::string& host, int port, int workers, int socket_flags = 0);
        HttpServer(int port, int workers, int socket_flags = 0);
        HttpServer(const HttpServer&) = delete;
        HttpServer() = delete;
        ~HttpServer();

        void listen();
        void addController(std::shared_ptr<BaseController> controller);
        httplib::Server* getHttplibServer();
      
      private:
        void listenInThread();
        std::shared_ptr<httplib::HttpThreadPool> httpThreadPool_;;

        std::string host_;
        int port_;
        int workers_;
        std::unique_ptr<httplib::Server> server_;
        int socket_flags_;
        std::vector<std::shared_ptr<BaseController> > controllerVec_;
        friend class BaseController;
    };
}   // namespace httplib

#endif // SERVER_H
