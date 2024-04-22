/*
 * @Description: ann index data holder
 * @Author: junwei.wang@zju.edu.cn
 * @Date: 2024-04-01 16:17:13
 * @LastEditTime: 2024-04-01
 */

#include <iostream>
namespace annop {
namespace common {

struct ANNIndexHolderBase {
  public:
    virtual bool set_index(const std::string& index) = 0;
    virtual bool get_index(const std::string& index) = 0;
};

}  // namespace common
}  // namespace annop