cmake_minimum_required(VERSION 3.17)
project(testversion)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(testversion main.cpp data.h)

target_link_libraries(testversion ${OpenCV_LIBS})