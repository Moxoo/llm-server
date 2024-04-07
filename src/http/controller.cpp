#include "http/controller.hpp"

void httplib::BaseController::initToServer(httplib::Server* server)
{
    server_ = server;
    this->bind();
}

void httplib::BaseController::initToServer(httplib::HttpServer* server)
{
    server_ = server->getHttplibServer();
    this->bind();
}

void httplib::BaseController::bind()
{
    throw std::runtime_error(std::string("HttpServer must override ") + __FUNCTION__);
}