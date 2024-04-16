package(default_visibility = ["//visibility:public"])

load("//:annop.bzl", "tf_gen_op_libs", "tf_gen_op_wrappers_cc", "tf_gen_op_wrapper_cc")

tf_gen_op_libs(
    # is_external = False,
    op_lib_names = [        # filenames under `ops/`
        "annop_ops",
        "time_two_ops"
    ],
    deps = [
        "@tensorflow//:tensorflow_cc",
    ],
)


# tf_gen_op_wrapper_cc(
#     name = "annop_ops",
#     out_ops_file = "annop_ops",
#     api_def_srcs = ["//annop/cc/api_def:api_def"]
# )

tf_gen_op_wrappers_cc(
    name = "annop_ops",
    api_def_srcs = ["//annop/cc/api_def:api_def"],
    op_lib_names = [
        "annop_ops",
        "time_two_ops"
    ],
)

cc_library(
    name = "ops",
    visibility = ["//visibility:public"],
    deps = [
        "time_two_ops_op_lib",
        "annop_ops_op_lib"
    ],
    alwayslink = 1,
)


# cc_library(
#     name = "annop_ops_cc",
#     srcs = ["annop_ops.cc"],
#     hdrs = ["annop_ops.h"],
#     visibility = ["//visibility:public"],
#     deps = [
#         "@tensorflow//:tensorflow_cc",
#     ],
#     alwayslink = 1,
# )


# cc_library(
#     name = "annop_ops_cc",
#     srcs = ["annops/annop_ops.cc"],
#     hdrs = ["annops/annop_ops.h"],
#     visibility = ["//visibility:public"],
#     deps = [
#         "@tensorflow//:tensorflow_cc",
#     ],
#     alwayslink = 1,
# )

