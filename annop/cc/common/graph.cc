/*
 * @Description: graph impl
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-13 23:39:22
 * @LastEditTime: 2024-04-13
 */
#include "graph.hh"

#include <cstdint>

#include "embedding.h"

namespace annop {
namespace common {

// ctor
Graph::Graph(DataType type, uint64_t n, int m) : n_(n), m_(m) {}

}  // namespace common
}  // namespace annop