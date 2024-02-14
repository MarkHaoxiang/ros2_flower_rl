# System workspace
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y

# Venv
if ! [ -d "venv" ]; then
  python3 -m venv venv  --system-site-packages --symlinks
fi
source venv/bin/activate

# Project workspace
colcon build --cmake-args -DPython3_EXECUTABLE="venv/bin/python"
source install/local_setup.bash

# Replace with local venv
export PYTHON_PATH='/home/markhaoxiang/Projects/fl/ros2_flower_rl/venv/lib/python3.11/site-packages'

