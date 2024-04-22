#include "tensorflow/core/framework/shape_inference.h"
#include <memory>
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
#include "annop/cc/annops/annop_ops.h"
#include "annop/cc/annops/time_two_ops.h"
#include <iostream>
#include <tensorflow/cc/ops/math_ops.h>

#include <tensorflow/cc/saved_model/loader.h>

#include <gflags/gflags.h>

using namespace std;
using namespace tensorflow;

// DEFINE_string(sdf,"sdf");
DEFINE_string(modelpath,
              "/home/yinze/Desktop/tensorflow/tensorflow/cc/saved_model/"
              "testdata/half_plus_two_pbtxt/00000123/",
              "sdf");

int main(int argc, char **argv) {
  // ops::TimeTwo tt;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  printf("All registered ops:\n%s\n",
         tensorflow::OpRegistry::Global()->DebugString(false).c_str());
  // auto bundle = absl::make_unique<SavedModelBundle>();
  auto bundle = std::make_shared<SavedModelBundle>();

  // Create dummy options.
  tensorflow::SessionOptions sessionOptions;
  tensorflow::RunOptions runOptions;

  LOG(INFO) << "Start to load model from " << FLAGS_modelpath;

  // Load the model bundle.
  const auto loadResult = tensorflow::LoadSavedModel(
      sessionOptions, runOptions, FLAGS_modelpath, {"serve"}, bundle.get());

  TF_CHECK_OK(loadResult);

  LOG(INFO) << bundle->meta_graph_def.DebugString();

  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
  tensor.vec<float>()(0) = 20.f;
  tensor.vec<float>()(1) = 6000.f;

  // Link the data with some tags so tensorflow know where to put those data
  // entries.
  std::vector<std::pair<std::string, tensorflow::Tensor>> feedInputs = {
      {"user_emb", tensor}};
  std::vector<std::string> fetches = {"dd"};

  // We need to store the results somewhere.
  std::vector<tensorflow::Tensor> outputs;

  // Let's run the model...
  // bundle->session
  auto status = bundle->session->Run(feedInputs, fetches, {}, &outputs);

  TF_CHECK_OK(status);

  // ... and print out it's predictions.
  for (const auto &record : outputs) {
    LOG(INFO) << record.DebugString();
  }
  return 0;
}