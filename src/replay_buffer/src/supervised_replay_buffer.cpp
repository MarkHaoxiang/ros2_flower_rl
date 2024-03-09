#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ml_interfaces/msg/feature_label_pair.hpp"
#include "ml_interfaces/srv/sample_feature_label_pair.hpp"
#include "replay_buffer.cpp"

int main(int argc, char *argv[])
{
    // Build Node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReplayBuffer<ml_interfaces::msg::FeatureLabelPair, ml_interfaces::srv::SampleFeatureLabelPair>>();
    // Launch Node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}