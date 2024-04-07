#ifndef HTTP_THREAD_POOL_H
#define HTTP_THREAD_POOL_H

#include "http/httplib.h"

namespace httplib {
    class HttpThreadPool : public TaskQueue {
      public:
        explicit HttpThreadPool();
        HttpThreadPool(size_t n);
        HttpThreadPool(const ThreadPool&) = delete;
        ~HttpThreadPool();
        void enqueue(std::function<void()> fn) override;
        void shutdown() override;

        void addAThread(const std::string &threadName);

      private:
        struct worker {
            explicit worker(HttpThreadPool& pool) : pool_(pool) {}
            void operator()();
            HttpThreadPool& pool_;
        };
        friend struct worker;

        std::vector<std::thread> threads_;
        std::list<std::function<void()>> jobs_;

        bool shutdown_;

        std::condition_variable cond_;
        std::mutex mutex_;
    };
} // namespace httplib

#endif // HTTP_THREAD_POOL_H
