#include "http/http_thread_pool.hpp"

httplib::HttpThreadPool::HttpThreadPool() : shutdown_(false)
{

}

httplib::HttpThreadPool::HttpThreadPool(size_t n) : shutdown_(false)
{
    while (n)
    {
        std::thread task(worker(*this));
        task.detach();
        threads_.emplace_back(std::move(task));
        n--;
    }
}

httplib::HttpThreadPool::~HttpThreadPool()
{
    std::cout << "dis HttpThreadPool" << std::endl;
    shutdown();
}

void httplib::HttpThreadPool::enqueue(std::function<void()> fn)
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        jobs_.push_back(std::move(fn));
    }
    cond_.notify_one();
}

void httplib::HttpThreadPool::shutdown()
{
    // Stop all worker threads...
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cond_.notify_all();
}

void httplib::HttpThreadPool::addAThread(const std::string &threadName)
{
    std::thread task(worker(*this));
    auto handle = task.native_handle();
    pthread_setname_np(handle, threadName.c_str());
    task.detach();
    threads_.emplace_back(std::move(task));
}

void httplib::HttpThreadPool::worker::operator()() 
{
    while (true) {
        std::function<void()> fn;
        {
            std::unique_lock<std::mutex> lock(pool_.mutex_);

            pool_.cond_.wait(
                lock, [&] { return !pool_.jobs_.empty() || pool_.shutdown_; });

            if (pool_.shutdown_ && pool_.jobs_.empty()) { break; }

            fn = std::move(pool_.jobs_.front());
            pool_.jobs_.pop_front();
        }

        assert(true == static_cast<bool>(fn));
        fn();
    }
}