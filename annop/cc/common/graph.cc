/*
 * @Description: graph impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-13 23:39:22
 * @LastEditTime: 2024-04-13
 */
#include "graph.hh"

#include <cstddef>
#include <cstdint>

namespace annop {
namespace common {

// ctor
Graph::Graph(DataType type, uint64_t n, int m) : n_(n), m_(m) {
    labels_.reserve(n);
    linklist_.reset(new LinkedListType(n_, m_));
}

// dtor
Graph::~Graph() {
    if (linklist_) {
        linklist_->Unref();
    }
}

}  // namespace common
}  // namespace annop