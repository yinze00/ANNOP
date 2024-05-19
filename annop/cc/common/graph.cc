/*
 * @Description: graph impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-13 23:39:22
 * @LastEditTime: 2024-04-13
 */
#include "graph.hh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace annop {
namespace common {

// ctor
Graph::Graph(DataType type, uint64_t n, int m) : n_(n), m_(m) {
    labels_.reserve(n);
    linklist_.reset(new LinkedListType(n_, m_));
    // linklist_ = std::make_unique<LinkedListType>(n, m);
}

// dtor
Graph::~Graph() {
    if (linklist_) {
        linklist_->Unref();
    }
}

void Graph::get_label(uint32_t index, uint64_t& label) {}

void Graph::get_labels(const std::vector<uint32_t>& indice,
                       std::vector<uint64_t>& labels) {}

}  // namespace common
}  // namespace annop