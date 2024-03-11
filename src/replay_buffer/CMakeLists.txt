cmake_minimum_required(VERSION 3.8)
project(replay_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ml_interfaces REQUIRED)

# executables
add_executable(supervised_memory src/supervised_replay_buffer.cpp)
ament_target_dependencies(supervised_memory rclcpp ml_interfaces)

add_executable(transition_memory src/transition_replay_buffer.cpp)
ament_target_dependencies(transition_memory rclcpp ml_interfaces)

install(TARGETS
  supervised_memory
  transition_memory
  DESTINATION lib/${PROJECT_NAME})

ament_package()
