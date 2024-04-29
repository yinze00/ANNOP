package(default_visibility = ["//visibility:public"])

load("//:annop.bzl", "tf_kernel_library")

tf_kernel_library(
    name = "zero_out_op",
    srcs = [
        "zero_out_op.cc",
    ],
    hdrs = ["zero_out_op.h"],
    deps = [
        "@tensorflow//:tensorflow"
    ]
)

tf_kernel_library(
    name = "time_two_op",
    srcs = [
        "time_two_op.cc"
    ],
    hdrs = [
        "time_two_op.h"
    ],
    deps = [
        "@tensorflow//:tensorflow"
    ]
)

tf_kernel_library(
    name = "time_three_op",
    srcs = [
        "time_three_op.cc"
    ],
    hdrs = [
        "time_three_op.h"
    ],
    deps = [
        "@tensorflow//:tensorflow"
    ]
)

tf_kernel_library(
    name = "gather_neighbors",
    prefix = "gather_neighbors",
    deps = [
        "@tensorflow//:tensorflow"
    ]
)

cc_library(
    name = "ann_kernel",
    deps = [
        ":time_two_op",
        ":time_three_op",
        ":zero_out_op"
    ]
)