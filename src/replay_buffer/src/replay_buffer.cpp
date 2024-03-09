#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
using std::placeholders::_1;

template <typename T>
class ReplayBuffer : public rclcpp::Node
{
public:
    ReplayBuffer() : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<T>(
            "topic",
            10,
            std::bind(&ReplayBuffer::topic_callback, this, _1));
    }

private:
    void topic_callback(const typename T::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReplayBuffer<std_msgs::msg::String>>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}