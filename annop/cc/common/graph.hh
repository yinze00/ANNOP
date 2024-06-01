/*
 * @Description: Graph class, containing the graph structure of edges
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-08 22:11:44
 * @LastEditTime: 2024-04-08
 */

#pragma once

#include <google/protobuf/generated_message_reflection.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "buffer.hh"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/types.h"

namespace annop {
namespace common {

template <typename T, typename enable = void>
struct LinkedList;

template <typename T>
struct LinkedList<T, typename std::enable_if<std::is_integral<T>::value>::type>
    : public Buffer {
  public:
    explicit LinkedList(T n, int m) : n_(n), m_(m), Buffer(new T[n * m]) {}
    ~LinkedList() = default;
    inline std::tuple<int, T*> gather_neighbors(size_t index) {
        return std::make_tuple(m_, this->template base<T>() + index * m_);
    }

  public:
    T n_;
    int m_;
};

template <typename T>
struct LinkedListDeleter {
    void operator()(LinkedList<T>* obj) const { obj->Unref(); }
};

template <typename T>
using LinkedListPtr = std::unique_ptr<LinkedList<T>, LinkedListDeleter<T>>;

/*
 * Directed Acyclic Graph, Like NSW, which may has only one layer, we called
 * it level0, level0 contains the whole corpus's linked list (neighbors)
 */
class Graph {
  public:
    using LinkedListType = LinkedList<uint32_t>;
    using LinkedListPtrType = LinkedListPtr<uint32_t>;

  public:
    explicit Graph(DataType type, uint64_t n, int m);
    ~Graph();

    // get neis
    void get_label(uint32_t, uint64_t&);
    uint64_t get_label(uint32_t index) {
        uint64_t res;
        get_label(index, res);
        return res;
    }
    void get_labels(const std::vector<uint32_t>&, std::vector<uint64_t>&);
    std::vector<uint64_t> get_labels(const std::vector<uint32_t>& indice) {
        std::vector<uint64_t> res;
        get_labels(indice, res);
        return res;
    }
    // set neis

    float* gather_neighbors(size_t index);

    void gather_neighbors(const std::vector<size_t>& indice);

    void set_labels(std::vector<uint64_t>& labels);

  protected:
    uint64_t n_{0};  // number of elements
    uint32_t m_{0};  // number of neighbors per element

    LinkedListPtrType linklist_;

    std::vector<uint64_t> labels_;
};

/* hierachy Graph like hnsw */
class HGraph : public Graph {
  public:
    HGraph(DataType type, uint64_t n, int m) : Graph(type, n, m) {
        ;
        ;
    }
};

}  // namespace common
}  // namespace annop