package(default_visibility = ["//visibility:public"])

load("//:annop.bzl", "tf_kernel_library")

cc_library(
    name = "common",
    hdrs = glob(["*.hh"]),
    srcs = [    
        "embedding.cc",
        "graph.cc",
    ],
    copts = [
        "-std=c++17"
    ],
    deps = [
        "@tensorflow//:tensorflow"
    ]
)