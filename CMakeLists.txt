cmake_minimum_required(VERSION 3.14)

# 1. 强制告诉 CMake nvcc 编译器在哪里
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")

# 2. 明确指定 RTX 3090 的 Ampere 架构算力为 86
set(CMAKE_CUDA_ARCHITECTURES "86")

# 注意：上面这两句 set 必须严格写在 project 命令的上方！
project(RIFE_TRT_LF_CPP CUDA CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 寻找 OpenCV
# ==========================================
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
# ==========================================
# 2. 寻找 CUDA
# ==========================================
find_package(CUDAToolkit REQUIRED) # CMake 3.17+ 推荐的现代 CUDA 寻找方式

# ==========================================
# 3. 设置你手动解压的 TensorRT 根目录
# ==========================================
set(TRT_ROOT "/home/ldh/Programs/TensorRT-8.6.1.6")

# 寻找头文件路径
find_path(TRT_INCLUDE_DIR NvInfer.h
        PATHS ${TRT_ROOT}/include
        NO_DEFAULT_PATH
)

# 寻找核心库文件路径
find_library(TRT_LIBRARY nvinfer
        PATHS ${TRT_ROOT}/lib
        NO_DEFAULT_PATH
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIBRARY)
    message(FATAL_ERROR "Still could not find TensorRT at ${TRT_ROOT}. Please check if the directory exists.")
endif()

# ==========================================
# 4. 编译可执行文件 (移除了重复的 main.cpp)
# ==========================================
add_executable(RIFE_TRT_CPP main.cpp
        kernels.cu)

# ==========================================
# 5. 强制绑定头文件目录 (解决 NvInfer.h 找不到的核心)
# ==========================================
target_include_directories(RIFE_TRT_CPP PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${TRT_INCLUDE_DIR}
)

# ==========================================
# 6. 链接所有神仙库
# ==========================================
target_link_libraries(RIFE_TRT_CPP PRIVATE
        ${OpenCV_LIBS}
        CUDA::cudart    # CUDA 运行时库
        ${TRT_LIBRARY}  # TensorRT 核心库

        ${OPENGL_LIBRARIES}
        GLEW
        glfw
)

# ==========================================
# 7. 设置 RPATH，确保程序运行时能找到动态库
# ==========================================
set_target_properties(RIFE_TRT_CPP PROPERTIES INSTALL_RPATH "${TRT_ROOT}/lib")
set_target_properties(RIFE_TRT_CPP PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)