workspace(name = "annop")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


# git_repository(
#     name = "tensorflow",
#     remote = "git@github.com:yinze00/libtensorflow.git"
# )


# local_repository(
#     name = "tensorflow",
#     # build_file = "/home/yinze/Desktop/tensorflow/BUILD",
#     path = "/home/yinze/Desktop/tensorflow",
# )


# WORKSPACE

http_archive(
    name = "gtest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz"],
    strip_prefix = "googletest-release-1.11.0",
)



new_local_repository(
    name = "tensorflow",
    build_file = "/home/yinze/libtf1.15/BUILD",
    path = "/home/yinze/libtf1.15",
)