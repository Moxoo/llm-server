# 基于[MNN](https://github.com/alibaba/MNN)和[cpp-httplib](https://github.com/yhirose/cpp-httplib)实现的简单llm推理服务
- 支持CPU推理(同步推理和流式推理)
- 基于[Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)的MNN格式的量化int8模型,可以参考项目[llm-export](https://github.com/wangzhaode/llm-export)和[mnn-llm](https://github.com/wangzhaode/mnn-llm)获取更多模型支持

# llm-server编译
## x86_64
```
mkdir build && cd build
cmake .. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSTEM_PROCESSOR=x86_64 \
-DBUILD_FOR_AARCH64=OFF \
-DMNN_SEP_BUILD=OFF \
-DMNN_BUILD_TOOLS=OFF \
-DMNN_LOW_MEMORY=ON \
-DUSING_DISK_EMBED=ON \
-DDUMP_PROFILE_INFO=ON
make -j4
```
## arm64
```
mkdir build && cd build
export cross_compile_toolchain=/path/to/sda/aarch64/toolchain
cmake .. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
-DBUILD_FOR_AARCH64=ON \
-DMNN_SEP_BUILD=OFF \
-DMNN_BUILD_TOOLS=OFF \
-DMNN_LOW_MEMORY=ON \
-DUSING_DISK_EMBED=ON \
-DDUMP_PROFILE_INFO=ON \
-DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++
make -j4
```

# run
## 启动服务端
首先需要完成model文件[下载](https://github.com/wangzhaode/mnn-llm/releases/tag/qwen-1.8b-mnn-int8)，并把所有的模型文件放在./model/qwen-1.8b-int8-chat目录下
```
taskset f0 ./llm-server -f mnn -m ./model/qwen-1.8b-int8-chat -h 0.0.0.0 -p 8080 -w 4
```
## 使用curl客户端测试
### 流式推理
```
curl -v  http://127.0.0.1:8080/stream_infer -d '{
    "prompt": {
        "system": "You are a helpful assistant.",
        "user": "你好,你是谁?"
    }
}'
*   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to 127.0.0.1 (127.0.0.1) port 8080 (#0)
> POST /stream_infer HTTP/1.1
> Host: 127.0.0.1:8080
> User-Agent: curl/7.58.0
> Accept: */*
> Content-Length: 111
> Content-Type: application/x-www-form-urlencoded
> 
* upload completely sent off: 111 out of 111 bytes
< HTTP/1.1 200 OK
< Content-Type: text/event-stream
< Keep-Alive: timeout=200, max=5
< Transfer-Encoding: chunked
< 
data: 你好

data: ！

data: 我是

data: 来自

data: 达

data: 摩

data: 院

data: 的大

data: 规模

data: 语言

data: 模型

data: ，

data: 我

data: 叫

data: 通

data: 义

data: 千

data: 问

data: 。

* Connection #0 to host 127.0.0.1 left intact
```

### 同步推理
```
curl -v  http://127.0.0.1:8080/infer -d '{
    "prompt": {
        "system": "You are a helpful assistant.",
        "user": "你好,你是谁?"
    }
}'
*   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to 127.0.0.1 (127.0.0.1) port 8080 (#0)
> POST /infer HTTP/1.1
> Host: 127.0.0.1:8080
> User-Agent: curl/7.58.0
> Accept: */*
> Content-Length: 111
> Content-Type: application/x-www-form-urlencoded
> 
* upload completely sent off: 111 out of 111 bytes
< HTTP/1.1 200 OK
< Content-Length: 121
< Content-Type: application/json
< Keep-Alive: timeout=200, max=5
< 
* Connection #0 to host 127.0.0.1 left intact
{"type": "post", "code": 2001, "response":你好！我是来自达摩院的大规模语言模型，我叫通义千问。}
```

# 参考
- https://github.com/wangzhaode/mnn-llm
- https://github.com/yhirose/cpp-httplib
- https://blog.csdn.net/qq_51217746/article/details/130245032#t8
- https://huggingface.co/Qwen/Qwen-1_8B-Chat
