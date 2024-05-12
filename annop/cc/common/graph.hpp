/*
 * @Description: Graph class, containing the graph structure of edges
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-05-08 22:11:44
 * @LastEditTime: 2024-05-08
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "embedding.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace annop {
namespace common {

class Graph {
    // explicit Graph()
};

class HGraph : public Graph {

};

}  // namespace common
}  // namespace annop