/*
 * @Description: class embedding
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-08 22:00:31
 * @LastEditTime: 2024-04-08
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "buffer.hh"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

class Embedding : public Buffer {
  public:
    DataType type;
};

class EmbeddingHolder {
  public:
    EmbeddingHolder() = delete;
    EmbeddingHolder(DataType type, int64_t n, int dim);
    ~EmbeddingHolder();

    // template <typename T>
    float* gather_embedding(int64_t offset);

  private:
    Buffer* buf_;
    DataType type_;
    int64_t n_;
    int dim_;
};

}  // namespace common
}  // namespace annop