#include <memory>

#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
using std::placeholders::_1;

template <typename T>
class ReplayBuffer : public rclcpp::Node
{
    // FIFO Replay Buffer

public:
    ReplayBuffer() : Node("replay_buffer")
    {
        auto size_desc = rcl_interfaces::msg::ParameterDescriptor{};
        size_desc.description = "Maximum number of elements in the Replay Buffer";
        this->declare_parameter("size", 1024);

        subscription_ = this->create_subscription<T>(
            "observations",
            10,
            std::bind(&ReplayBuffer::topic_callback, this, _1));
    }

private:
    // Memory
    std::vector<std::shared_ptr<T>> memory = {};
    int memory_index = 0;

    // Subscription for input data
    std::shared_ptr<rclcpp::Subscription<std_msgs::msg::String>> subscription_;
    void topic_callback(const std::shared_ptr<T> msg)
    {
        // Get Maximum Memory Size
        size_t size = static_cast<size_t> (this->get_parameter("size").as_int());
        // Memory Management
            // Memory is not full
        if (this->memory.size() < size) {
            this->memory_index = this->memory.size();
            this->memory.push_back(msg);
        }
        else {
            // Memory Full
            this->memory.at(this->memory_index) = msg;
        }
        this->memory_index = (this->memory_index+1) % size;

        // RCLCPP_INFO(this->get_logger(), "I heard data. %d %d %d", this->memory_index, size, this->memory.size());
    }
};

int main(int argc, char *argv[])
{
    // Build Node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReplayBuffer<std_msgs::msg::String>>();
    // Launch Node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}