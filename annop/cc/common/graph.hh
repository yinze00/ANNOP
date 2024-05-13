/*
 * @Description: Graph class, containing the graph structure of edges
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-08 22:11:44
 * @LastEditTime: 2024-04-08
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "buffer.hh"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

/* Directed Acyclic Graph, Like NSW, which may has only one layer, we called it
 * level0, level0 contains the whole corpus's linked list (neighbors) */
class Graph {
    // explicit Graph()
  public:
    explicit Graph(DataType type, uint64_t n, int m);
    ~Graph() = default;

  protected:
    uint64_t n_{0};  // number of elements
    int m_{0};       // number of neighbors per element
};

class HGraph : public Graph {};

}  // namespace common
}  // namespace annop