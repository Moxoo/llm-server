cmake_minimum_required(VERSION 3.5)
PROJECT(llm-server)

option(BUILD_FOR_AARCH64 "Build for arm64." ON)
option(MNN_LOW_MEMORY "save memory." ON)

option(USING_DISK_EMBED "Using disk embedding to save memory." ON)
if (USING_DISK_EMBED)
    add_definitions(-DUSING_DISK_EMBED)
endif()

option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)
if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

option(CPPHTTPLIB_USE_POLL "cpp http lib use poll" ON)
if (CPPHTTPLIB_USE_POLL)
    add_definitions(-DCPPHTTPLIB_USE_POLL)
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-result")
SET(ABSL_PROPAGATE_CXX_STD ON)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidjson/include
)

# Build libllm.so
SET(LLM_SRCS
    src/llm/llm.cpp
    src/llm/qwen.cpp
    src/llm/tokenizer.cpp
)
add_library(llm SHARED ${LLM_SRCS})
TARGET_LINK_LIBRARIES(llm MNN)

# Build libhttp.so
SET(HTTP_SRCS
    src/http/server.cpp
    src/http/controller.cpp
    src/http/infer_contorller.cpp
    src/http/http_thread_pool.cpp
)
add_library(http SHARED ${HTTP_SRCS})

# Build executable binaries.
ADD_EXECUTABLE(llm-server main.cpp)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp)
TARGET_LINK_LIBRARIES(llm-server llm http
    absl::base
    absl::strings
    absl::str_format
    absl::time
    absl::flat_hash_map
    absl::synchronization
)
