cmake_minimum_required(VERSION 2.8.3)
project(image_representation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE DEBUG)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(catkin_simple REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS python numpy)
find_package(PythonLibs REQUIRED)
find_package(sdevo_msgs REQUIRED)


catkin_simple(ALL_DEPS_REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

cs_add_executable(image_representation
   src/ImageRepresentation.cpp
   src/ImageRepresentation_node.cpp
)

target_link_libraries(image_representation
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   yaml-cpp
)

cs_install()