# CMakeLists.txt
find_package(TensorFlow REQUIRED)
add_executable(example demo.cc)


add_subdirectory(annops)

target_include_directories(example PRIVATE ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(example PUBLIC annops)
target_link_libraries(example PUBLIC ${TensorFlow_LIBRARIES})

# You may also link cuda if it is available.
# find_package(CUDA)
# if(CUDA_FOUND)
#   target_link_libraries(example ${CUDA_LIBRARIES})
# endif()