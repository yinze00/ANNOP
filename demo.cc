// example.cpp

#include "tensorflow/core/framework/shape_inference.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/cc/client/client_session.h>
#include "tensorflow/core/framework/tensor.h"
#include "annop/kernel/ZeroOutOp.h"


#include <iostream>
using namespace std;
using namespace tensorflow;


int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope(); 
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v =
      MatMul(root.WithOpName("v"), A, b,
             MatMul::TransposeB(
                 true)); // <- in your case you should put here your custom Op


  // auto z = ZeroOutOp(root.WithOpName("z"), v);
//   auto final = ZeroOut()
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}

// int main(int argc, char *argv[]) {
//   printf("All registered ops:\n%s\n",
//          tensorflow::OpRegistry::Global()->DebugString(false).c_str());
//   return 0;
// }