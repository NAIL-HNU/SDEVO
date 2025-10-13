cmake_minimum_required(VERSION 2.8.3)
project(sdevo_msgs)

# search for everything we need to build the messages, dont forget the message_generation
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  TimedFloat32MultiArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

