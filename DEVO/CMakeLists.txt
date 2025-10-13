cmake_minimum_required(VERSION 3.0.2)
project(sdevo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  dvs_msgs
  sensor_msgs
)

catkin_package()

install(PROGRAMS
  src/get_voxel.py
  src/pubPPVoxel.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)