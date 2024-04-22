#pragma once

#include <mutex>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

namespace tensorflow {

using CPUDEVICE = Eigen::ThreadPoolDevice;

template <typename T, typename TID> 
class GatherNeighbors : OpKernel {
public:
  void Compute(OpKernelContext *ctx) {}
private:
  mutable std::mutex mtx_;
  
};

} // namespace tensorflow