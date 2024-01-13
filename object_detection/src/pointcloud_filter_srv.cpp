#include "rclcpp/rclcpp.hpp"
#include "object_detection/pointcloud_filter_srv.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

#define filter_angle_range 0.08726646259971647

Pointcloud_Filter::Pointcloud_Filter() : Node("pointcloud_filter_srv") {

    this->cloud_in_flag = false;

    this->declare_parameter("cloud_topic", "/awl/camera_points");

    this->camera_cloudptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    std::string pointcloud_topicname = this->get_parameter("cloud_topic").as_string();
    std::string srv_name = "/filter_pointcloud" + pointcloud_topicname;

    /// @brief Subscribe to LUCI camera info topics
    camera_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>
        (pointcloud_topicname, 10, std::bind(&Pointcloud_Filter::cloud_callback, this, _1));

    /// @brief Publish to AW camera info topics
    filter_pointcloud_service = this->create_service<object_detection_interfaces::srv::FilterPointcloud>
        (srv_name, std::bind(&Pointcloud_Filter::service_callback, this, _1, _2));

    RCLCPP_INFO(this->get_logger(), "pointcloud_filter_srv params --- cloud_topic: %s", pointcloud_topicname.c_str());
    RCLCPP_INFO(this->get_logger(), "pointcloud_filter_srv initialized.");
}

void Pointcloud_Filter::service_callback(const std::shared_ptr<object_detection_interfaces::srv::FilterPointcloud::Request> request,
    std::shared_ptr<object_detection_interfaces::srv::FilterPointcloud::Response> response) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloudptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PCLPointCloud2::Ptr out_cloud2ptr(new pcl::PCLPointCloud2()); 
    
    std::cout << "Received request to filter pointcloud about ray:" << std::endl;
    std::cout << "\t[ " << request->x_dir << ", " << request->y_dir << ", " << request->z_dir << " ]" << std::endl;
    if (this->cloud_in_flag) {
        float hfov_angle = atan2(request->x_dir, request->z_dir);
        float vfov_angle = atan2(request->y_dir, request->z_dir);

        float hfov_angle_min = hfov_angle - filter_angle_range;
        float hfov_angle_max = hfov_angle + filter_angle_range;
        float vfov_angle_min = vfov_angle - filter_angle_range;
        float vfov_angle_max = vfov_angle + filter_angle_range;

        for (size_t i = 0; i < this->camera_cloudptr->size(); ++i) {
            float h_azimuth = atan2(this->camera_cloudptr->points[i].x, this->camera_cloudptr->points[i].z);
            float v_azimuth = atan2(this->camera_cloudptr->points[i].y, this->camera_cloudptr->points[i].z);
            if (h_azimuth >= hfov_angle_min && h_azimuth <= hfov_angle_max && 
                v_azimuth >= vfov_angle_min && v_azimuth <= vfov_angle_max) {
                out_cloudptr->push_back(this->camera_cloudptr->points[i]);
            }
        }
        std::cout << "Filtered pointcloud size: " << out_cloudptr->size() << std::endl; 
        pcl::toPCLPointCloud2(*out_cloudptr, *out_cloud2ptr);
        std::cout << "\tafter pcl::toPCLPointCloud2" << std::endl;
        pcl_conversions::fromPCL(*out_cloud2ptr, response->out_cloud);
        std::cout << "\t\tafter pcl_conversions::fromPCL" << std::endl;
        response->success = true;
        std::cout << "Converted pointcloud to ros message and returned" << std::endl; 
        return;
    } else {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "No pointcloud received yet.");
        response->success = false;
        return;
    }

}

void Pointcloud_Filter::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    this->cloud_in_flag = true;
    pcl::PCLPointCloud2::Ptr in_cloudptr(new pcl::PCLPointCloud2());
    auto msg_in = *msg;
    pcl_conversions::toPCL(msg_in, *in_cloudptr);
    pcl::fromPCLPointCloud2(*in_cloudptr, *this->camera_cloudptr);
}
