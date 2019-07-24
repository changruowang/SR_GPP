SRCS = $(wildcard *.cpp) $(wildcard *.c) $(wildcard ./dlib-19-17/dlib/all/*.cpp)  
OBJS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))   # OBJS将$(SRCS)下的.cpp文件转化为.o文件
CXX = g++                   # 代表所使用的编译器
LIBS = $(shell pkg-config --libs  opencv) -pthread -lX11
CXXFLAGS = -I./Eigen -I./dlib-19-17 $(shell pkg-config --cflags  opencv)  -std=c++11  -Wall -O0 -DHAVE_CONFIG_H  #附加参数
OUTPUT = process   	  #输出程序名称
 
all:$(OUTPUT)
$(OUTPUT) : $(OBJS)
	$(CXX) $^ -o $@ $(LIBS) 
 
%.o : %.cpp
	$(CXX) -c $< $(CXXFLAGS)
 
.PHONY:clean
clean:
	rm -rf *.out *.o     #清除中间文件及生成文件

