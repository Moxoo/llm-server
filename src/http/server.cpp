#include "http/server.hpp"

httplib::HttpServer::HttpServer(std::shared_ptr<httplib::HttpThreadPool> pool, const std::string& host, int port, int workers, int socket_flags)
    : httpThreadPool_(pool), host_(host), port_(port), workers_(workers), socket_flags_(socket_flags)
{
    server_ = std::make_unique<httplib::Server>();
}

httplib::HttpServer::HttpServer(std::shared_ptr<httplib::HttpThreadPool> pool, int port, int workers, int socket_flags)
    : httpThreadPool_(pool), host_(std::string("0.0.0.0")), port_(port), workers_(workers), socket_flags_(socket_flags)
{
    server_ = std::make_unique<httplib::Server>();
}

httplib::HttpServer::HttpServer(const std::string& host, int port, int workers, int socket_flags)
    : host_(host), port_(port), workers_(workers), socket_flags_(socket_flags)
{
    server_ = std::make_unique<httplib::Server>();
}

httplib::HttpServer::HttpServer(int port, int workers, int socket_flags)
    : host_(std::string("0.0.0.0")), port_(port), workers_(workers), socket_flags_(socket_flags)
{
    server_ = std::make_unique<httplib::Server>();
}

httplib::HttpServer::~HttpServer()
{
    std::cout << "dis HttpServer" << std::endl;
}

void httplib::HttpServer::listen()
{
    std::cout << "http server listen to " << port_ << std::endl;
    server_->new_task_queue = [=] { return new ThreadPool(workers_); };
    server_->set_keep_alive_timeout(200);
    server_->listen(host_, port_, socket_flags_);
    std::cout << "http server OK" << std::endl;
//     listenInThread();
}

void httplib::HttpServer::addController(std::shared_ptr<httplib::BaseController> controller)
{
    controllerVec_.push_back(controller);
    controller->initToServer(this);
}

httplib::Server* httplib::HttpServer::getHttplibServer()
{
    return server_.get();
}

void httplib::HttpServer::listenInThread()
{
    if (httpThreadPool_ != nullptr) 
    {
        httpThreadPool_->addAThread("listen_master");
        std::cout << "http server listen to " << port_ << std::endl;
        httpThreadPool_->enqueue(std::bind(&httplib::Server::listen, server_.get(), host_, port_, socket_flags_));
    }
    else{
        std::cout << "no pool " << port_ << std::endl;
    }
}
