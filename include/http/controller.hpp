#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "httplib.h"
#include "server.hpp"

namespace httplib {
    class HttpServer;
    class BaseController {
      friend class HttpServer;
      public:
        BaseController() = default;
        BaseController(BaseController&) = delete;
        ~BaseController() = default;

      protected:
        httplib::Server* server_;
        template <class Func, class T>
        auto BindController(Func&& func, T&& obj) {
            httplib::Server::Handler handler = std::bind(func, obj, std::placeholders::_1, std::placeholders::_2);
            return handler;
        }
        // 绑定具体的请求响应地址和请求响应方法
        virtual void bind();

      private:
        void initToServer(httplib::Server* _server);
        void initToServer(httplib::HttpServer* _server);
    };
}   // namespace controller

#endif // CONTROLLER_H