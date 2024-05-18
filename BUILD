


cc_binary(
    name = "demo",
    srcs = ["demo.cc"],
    copts = [
        "-g -O1",
    ],
    deps = [
        "//annop:annop"
        # "@tensorflow"
    ],
)

cc_binary(
    name = "load_and_run",
    srcs = ["load_and_run.cc"],
    copts = [
        "-g -O1",
    ],
    linkopts = [
        "-lgflags"
    ],
    deps = [
        "//annop:annop",
        "//annop/cc:time_two_ops_op_lib",
        # "@tensorflow"
    ],
)

cc_binary(
    name = "hnsw",
    srcs = ["hnsw.cc"],
    copts = [
        "-g -O1",
    ],
    linkopts = [
        "-lgflags"
    ],
    deps = [
        "//annop:annop",
        "//annop/cc:time_two_ops_op_lib",
        # "@tensorflow"
    ],
)