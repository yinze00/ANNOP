package(default_visibility = ["//visibility:public"])

load("//:annop.bzl", "tf_gen_op_libs", "tf_gen_op_wrappers_cc", "tf_gen_op_wrapper_cc")

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

# cc_library(
#     name = "annop_op_op_lib",
#     srcs = ["ops/annop_op.cc"],
#     deps = ["@tensorflow//:tensorflow_framework"],
#     visibility = ["//visibility:public"],
#     alwayslink = 1,
#     linkstatic = 1,
# )

# tf_gen_op_wrappers_cc(
#     name = "annop_ops_core",
#     op_lib_names = [
#         "annop_ops",
#         "time_two_ops",
#     ],
#     api_def_srcs = ["//annop/cc/api_def:api_def"],
#     pkg = "//annop/cc"
# )


# tf_gen_op_wrapper_cc(
#     name = "annop_ops",
#     out_ops_file = "annop_ops",
#     pkg = "//annop/cc",
# )

# cc_library(
#     name = "annop_ops_cc",
#     srcs = ["annop_ops.cc"],
#     hdrs = ["annop_ops.h"],
#     deps = ["@tensorflow//:tensorflow_framework"],
#     visibility = ["//visibility:public"],
#     alwayslink = 1,
# )


# for demo.cc

cc_library(
    name = "annop",
    deps = [
        # "//annop/cc:annop_ops_cc",
        "//annop/cc/kernel:ann_kernel",
        "//annop/cc:annop_ops",
        "//annop/cc:annop_ops_op_lib",
        # "//annop/cc:annop_ops_op_lib",
        "//annop/cc:time_two_ops_op_lib",
        # "//annop/cc:time_three_ops_op_lib",
        "@tensorflow"
    ]
)