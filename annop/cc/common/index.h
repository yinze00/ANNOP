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

namespace annop {
namespace common {

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