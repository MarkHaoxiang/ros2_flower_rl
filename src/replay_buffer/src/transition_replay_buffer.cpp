#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ml_interfaces/msg/transition.hpp"
#include "ml_interfaces/srv/sample_transition.hpp"
#include "replay_buffer.cpp"

int main(int argc, char *argv[])
{
    // Build Node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReplayBuffer<ml_interfaces::msg::Transition, ml_interfaces::srv::SampleTransition>>();
    // Launch Node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}