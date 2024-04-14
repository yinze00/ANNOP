// example.cpp

#include "tensorflow/core/framework/shape_inference.h"
#include <tensorflow/core/framework/api_def.pb.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_gen_lib.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include <tensorflow/cc/client/client_session.h>
// #include "annop/kernel/ZeroOutOp.h"
#include "annop/cc/annops/time_two_ops.h"
#include "annop/cc/annops/annop_ops.h"
#include <tensorflow/cc/ops/math_ops.h>

#include <iostream>
using namespace std;
using namespace tensorflow;

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  // printf("All registered ops:\n%s\n",
  //        tensorflow::OpRegistry::Global()->DebugString(true).c_str());
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b,
                  MatMul::TransposeB(
                      true)); // <- in your case you should put here your custom
                              //  Op

  // auto z = ZeroOutOp(root.WithOpName("z"), v);
  auto final = TimeThree(root.WithOpName("final"), v);
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({final}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}

// int main(int argc, char *argv[]) {
//   // printf("All registered ops:\n%s\n",
//   //        tensorflow::OpRegistry::Global()->DebugString(true).c_str());
//   OpList ops, to_print_ops;
//   OpRegistry::Global()->Export(0, &ops);
//   std::vector<string> api_def_dirs;
//   api_def_dirs.emplace_back(std::string("/home/yinze/Desktop/ANNOP/annop/cc/api_def"));
//   ApiDefMap api_def_map(ops);
//   if (!api_def_dirs.empty()) {
//     Env *env = Env::Default();
//     // Only load files that correspond to "ops".
//     auto newops = to_print_ops.mutable_op();
//     for (const auto &op : ops.op()) {
//       for (const auto &api_def_dir : api_def_dirs) {
//         const std::string api_def_file_pattern =
//             io::JoinPath(api_def_dir, "api_def_" + op.name() + ".pbtxt");
//         LOG(INFO) << op.name();
//         if (env->FileExists(api_def_file_pattern).ok()) {
//           *newops->Add() = op;
//           TF_CHECK_OK(api_def_map.LoadFile(env, api_def_file_pattern));
//           LOG(INFO) << api_def_map.GetApiDef(op.name())->Utf8DebugString();
//         }
//       }
//     }
//   }

//   to_print_ops.PrintDebugString();
//   // ops.add_op();
//   api_def_map.UpdateDocs();
//   return 0;
// }