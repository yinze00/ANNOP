/*
 * @Description: ANN Index
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-21 18:27:19
 * @LastEditTime: 2024-04-21
 */

#include <faiss/Index.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

using DataType = tensorflow::DataType;
//
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
    explicit Embedding(DataType type, int64_t n, int dim);

  private:
    Buffer* buf_;
    DataType type_;
    int64_t n_;
    int dim_;
};
class Graph {};

class AIndex {
  public:
    explicit AIndex(int n, int dim) {}

  private:
    Embedding embedding_;
};

template <typename T, typename U>
struct Index {
    using Inner_ID_Type = uint32_t;

  public:
    Index() = default;
    ~Index() = default;
    Index(const std::string& name, uint32_t dim, uint64_t n)
        : dim_(dim), n_(n), index_name_(name) {}

  public:
    virtual void GatherEmbeddings(const std::vector<Inner_ID_Type>& ids,
                                  std::vector<T>& element_embeddings) = 0;

    virtual std::vector<T> GatherEmbeddings(
        const std::vector<Inner_ID_Type>& ids) {
        std::vector<T> x;
        GatherEmbeddings(ids, x);
        return x;
    }

    virtual std::vector<T> GatherCandidates(
        const std::vector<Inner_ID_Type>& ids,
        std::vector<Inner_ID_Type>& candidates) = 0;

    virtual std::vector<Inner_ID_Type> GatherCandidates(
        const std::vector<Inner_ID_Type>& ids) {
        std::vector<Inner_ID_Type> cands;
        GatherCandidates(ids, cands);
        return cands;
    }

  public:
    // Getter,Setter.

  public:
    uint32_t dim_;                // dimensions
    uint64_t n_;                  // the numebr of elemnts
    std::vector<U> external_ids;  // inner id to external id;
    std::string index_name_{"default"};
};

template <typename T, typename U>
using IndexPtr = std::shared_ptr<Index<T, U>>;

template struct Index<float, uint32_t>;
template struct Index<double, uint32_t>;
template struct Index<float, uint64_t>;

}  // namespace common
}  // namespace annop