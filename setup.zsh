# System workspace
source /opt/ros/humble/setup.zsh
rosdep install -i --from-path src --rosdistro humble -y

# Venv
if ! [ -d "venv" ]; then
  python3 -m venv venv  --system-site-packages --symlinks
  touch ./venv/COLCON_IGNORE
#  pip install -r requirements.txt
fi
source venv/bin/activate

# Project workspace
source install/local_setup.zsh

export __ROS_ENV_READY__=1
