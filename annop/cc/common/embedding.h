/*
 * @Description: class embedding
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-08 22:00:31
 * @LastEditTime: 2024-04-08
 */

#pragma once

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

using DataType = tensorflow::DataType;

class Buffer : tensorflow::core::RefCounted {
  public:
    explicit Buffer(void* data_ptr) : data_(data_ptr) {}
    ~Buffer() {}

    void* data() const noexcept { return data_; }

    template <typename T>
    T* base() const noexcept {
        return reinterpret_cast<T*>(data());
    }

    tensorflow::DataType type() const noexcept { return dtype_; }

  private:
    tensorflow::DataType dtype_{tensorflow::DataType::DT_INT8};
    void* const data_;
};

class Embedding {
  public:
    Embedding() = default;
    Embedding(DataType type, int64_t n, int dim);

  private:
    Buffer* buf_;
    DataType type_;
    int64_t n_;
    int dim_;
};

}  // namespace common
}  // namespace annop