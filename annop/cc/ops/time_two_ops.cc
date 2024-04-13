#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("TimeTwo")
    .Attr("T: {int32, float}")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });