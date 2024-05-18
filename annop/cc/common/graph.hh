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
    std::tuple<int, T*> gather_neighbors(size_t index) {
        return std::make_tuple(m_, this->template base<T>() + index * m_);
    }

  public:
    T n_;
    int m_;
};

/*
 * Directed Acyclic Graph, Like NSW, which may has only one layer, we called it
 * level0, level0 contains the whole corpus's linked list (neighbors)
 */
class Graph {
  public:
    using LinkedListType = LinkedList<uint32_t>;

  public:
    explicit Graph(DataType type, uint64_t n, int m);
    ~Graph();

    // get neis
    // set neis
    

  protected:
    uint64_t n_{0};  // number of elements
    uint32_t m_{0};  // number of neighbors per element

    std::unique_ptr<LinkedListType> linklist_;

    std::vector<uint64_t> labels_;
    std::vector<uint64_t> labels;
};

/* hierachy Graph like hnsw */
class HGraph : public Graph {};

}  // namespace common
}  // namespace annop