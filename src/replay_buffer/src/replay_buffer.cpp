#include <vector>
#include <algorithm>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

template <typename T, typename S>
class ReplayBuffer : public rclcpp::Node
{
    // FIFO Replay Buffer

public:
    ReplayBuffer() : Node("replay_buffer")
    {
        RCLCPP_INFO(this->get_logger(), "Starting replay_buffer!");
        auto size_desc = rcl_interfaces::msg::ParameterDescriptor{};
        size_desc.description = "Maximum number of elements in the Replay Buffer";
        this->declare_parameter("size", 1024);

        this->set_parameter_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ReplayBuffer::set_parameter_callback, this, std::placeholders::_1));

        this->subscription_ = this->create_subscription<T>(
            "observations",
            10,
            std::bind(&ReplayBuffer::topic_callback, this, _1));

        this->service_ = this->create_service<S>(
            "replay_buffer_sample",
            std::bind(&ReplayBuffer::sample, this, _1, _2));
    }

private:
    // Memory
    std::vector<std::shared_ptr<T>> memory = {};
    int memory_index = 0;

    // Subscription for input data
    std::shared_ptr<rclcpp::Subscription<T>> subscription_;
    void topic_callback(const std::shared_ptr<T> msg)
    {
        // Get Maximum Memory Size
        size_t size = static_cast<size_t>(this->get_parameter("size").as_int());
        // Memory Management
        // Memory is not full
        if (this->memory.size() < size)
        {
            this->memory_index = this->memory.size();
            this->memory.push_back(msg);
        }
        else
        {
            // Memory Full
            this->memory.at(this->memory_index) = msg;
        }
        this->memory_index = (this->memory_index + 1) % size;

        // Debugging
        // RCLCPP_INFO(this->get_logger(), "I heard data. %d %d %d", this->memory_index, size, this->memory.size());
    }

    // Memory Size Management Callback
    // Assumes rarely call, so not correct wrt. keeping FIFO
    std::shared_ptr<OnSetParametersCallbackHandle> set_parameter_callback_handle_;
    rcl_interfaces::msg::SetParametersResult set_parameter_callback(
        const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "Size parameter adjusted";
        for (const auto &param : parameters)
        {
            if (param.get_name() == "size")
            {
                int new_size = param.as_int();
                // Increase or constant
                if (static_cast<size_t>(new_size) >= this->memory.size())
                {
                    this->memory_index = this->memory.size();
                }
                else
                {
                    // Naively keep first 'size' elements
                    this->memory.resize(static_cast<size_t>(new_size));
                    this->memory_index = 0;
                }
            }
        }
        return result;
    }

    // Data Sampling Service
    std::mt19937_64 rng{std::random_device{}()};
    std::shared_ptr<rclcpp::Service<S>> service_;
    void sample(const std::shared_ptr<typename S::Request> request,
                 std::shared_ptr<typename S::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Sampling Service Requested");
        size_t n = static_cast<size_t>(request->n);
        // RCLCPP_INFO(this->get_logger(), "%d", n);
        if (n > this->memory.size())
        {
            n = this->memory.size();
        }
        std::vector<std::shared_ptr<T>> out{};
        std::sample(
            this->memory.begin(),
            this->memory.end(),
            std::back_inserter(out),
            n,
            this->rng);
        
        for (size_t i = 0; i < out.size(); i ++) {
            response->batch.push_back(*out.at(i));
        }
    }
};
