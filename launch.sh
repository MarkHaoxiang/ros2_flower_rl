#!/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

[[ -z "$__ROS_ENV_READY__" ]] && source "$SCRIPT_DIR/setup.sh" || echo "ROS Environment Ready"

python3 misc/toy_fl_server.py 2>&1 | tee >(while read -r line; do
    if [[ $line == *"Requesting initial parameters from one random client"* ]]; then
        echo "Triggered, launching ROS Nodes as Flwr Clients..."
        ros2 launch launcher toy_problem_launch.py
    fi
done)
