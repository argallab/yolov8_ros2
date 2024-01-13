#ifndef POINTCLOUD_FILTER_SRV_INCLUDE_GUARD_HPP
#define POINTCLOUD_FILTER_SRV_INCLUDE_GUARD_HPP

#include <array>
#include <memory>
#include <math.h>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "object_detection_interfaces/srv/filter_pointcloud.hpp"


class Pointcloud_Filter : public rclcpp::Node
{
    public:
        
        /// @brief Construct a new Camera Converter object
        explicit Pointcloud_Filter();

    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr camera_cloudptr;

        bool cloud_in_flag;
        // std::string pointcloud_topicname, srv_name;

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr camera_sub;
        rclcpp::Service<object_detection_interfaces::srv::FilterPointcloud>::SharedPtr filter_pointcloud_service;

        /// @brief Function to slice point cloud 
        void service_callback(const std::shared_ptr<object_detection_interfaces::srv::FilterPointcloud::Request> request,
            std::shared_ptr<object_detection_interfaces::srv::FilterPointcloud::Response> response);
            
        /// @brief Callback function for camera pointcloud
        void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

};

#endif  // POINTCLOUD_FILTER_SRV_INCLUDE_GUARD_HPP