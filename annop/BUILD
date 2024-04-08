package(default_visibility = ["//visibility:public"])

load("//:annop.bzl", "tf_gen_op_libs", "tf_gen_op_wrappers_cc")

# cc_library(
#     name = "annop", 
#     hdrs = glob([
#         "kernel/*.h"
#     ]),
#     srcs = glob(["kernel/*.cc"]) + glob(["ops/*.cc"]),
#     linkstatic = False,
#     deps = [
#         "@tensorflow//:tensorflow"
#     ]
# )

# tf_gen_op_libs(
#     is_external = False,
#     op_lib_names = [
#         "annop_op"              # output: annop_op_op_lib
#     ],
#     # deps = [
#     #     "@tensorflow//:tensorflow",
#     # ]
# )

cc_library(
    name = "annop_op_op_lib",
    srcs = ["ops/annop_op.cc"],
    deps = ["@tensorflow//:tensorflow_framework"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    linkstatic = 1,
)

tf_gen_op_wrappers_cc(
    name = "annop_ops",
    op_lib_names = [
        "annop_op",
    ],
)
