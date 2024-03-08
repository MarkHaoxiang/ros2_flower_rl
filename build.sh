colcon build --packages-skip-up-to webots_ros2 \
    --cmake-args=-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON

source install/local_setup.zsh
