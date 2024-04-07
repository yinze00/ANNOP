package(default_visibility = ["//visibility:public"])


cc_library(
    name = "annop", 
    hdrs = glob([
        "kernel/*.h"
    ]),
    srcs = glob(["kernel/*.cc"]) + glob(["ops/*.cc"]),
    linkstatic = False,
    deps = [
        "@tensorflow//:tensorflow"
    ]
)
