CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# 指定工程名称
PROJECT(RUN_GPP)

SET(CMAKE_CXX_FLAGS "-std=c++11")
ADD_DEFINITIONS("-Wall -g -O3")
# 设置头文件搜索目录
SET(Eigen ./Eigen)
SET(Dlib ./dlib-19-17 )
INCLUDE_DIRECTORIES(${Eigen} ${Dlib} ./src)
# 导入OpenCV库
FIND_PACKAGE(OpenCV REQUIRED )
# 设置动态库链接目录  相当于-I./...
#LINK_DIRECTORIES( ${CAFFE_ROOT}/build/lib/ ${CUDA_HOME}/lib64/)
# 设置要链接的库文件  相当于 -lX1  -pthread
SET(LIB_PROJECT pthread X11 ${OpenCV_LIBS})
#设置可执行文件生成目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
#编译当前文件夹下的cpp
AUX_SOURCE_DIRECTORY(. DIR_SRCS)
#将子目录src  dlib/source编译为动态链接库GPP  在对应的子目录下还应该有个.list
ADD_SUBDIRECTORY(./src GPP)
ADD_SUBDIRECTORY(./dlib-19-17/dlib/all source)
#制定生成目标
ADD_EXECUTABLE(RUN_GPP ${DIR_SRCS})
#添加连接库
TARGET_LINK_LIBRARIES(RUN_GPP GPP source ${LIB_PROJECT})




