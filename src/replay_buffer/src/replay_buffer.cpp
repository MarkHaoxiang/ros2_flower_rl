#include <memory>

#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
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

        this->set_parameter_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ReplayBuffer::set_parameter_callback, this, std::placeholders::_1)
        );

        this->subscription_ = this->create_subscription<T>(
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

        RCLCPP_INFO(this->get_logger(), "I heard data. %d %d %d", this->memory_index, size, this->memory.size());
    }

    // Memory Size Management Callback
        // Assumes rarely call, so not correct wrt. keeping FIFO
    std::shared_ptr<OnSetParametersCallbackHandle> set_parameter_callback_handle_;
    rcl_interfaces::msg::SetParametersResult set_parameter_callback(
        const std::vector<rclcpp::Parameter> & parameters
    ) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "Size parameter adjusted";
        for (const auto & param: parameters) {
            if (param.get_name() == "size") {
                int new_size = param.as_int();
                // Increase or constant
                if (static_cast<size_t>(new_size) >= this->memory.size()) {
                    this->memory_index = this->memory.size();   
                }
                else {
                    // Naively keep first 'size' elements
                    this->memory.resize(static_cast<size_t> (new_size));
                    this->memory_index = 0;
                }
            }

        }
        return result;        
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