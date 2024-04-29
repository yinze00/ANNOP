#pragma once

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>

#include <mutex>

namespace tensorflow {

using CPUDEVICE = Eigen::ThreadPoolDevice;

template <typename T, typename TID>
class GatherNeighbors : OpKernel {
  public:
    explicit GatherNeighbors(OpKernelConstruction* context);

    void Compute(OpKernelContext* ctx) {}

  private:
    mutable std::mutex mtx_;
};

}  // namespace tensorflow