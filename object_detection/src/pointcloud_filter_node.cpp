#include "object_detection/pointcloud_filter_srv.hpp"


int main(int argc, char * argv[]){ 
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Pointcloud_Filter>());
    rclcpp::shutdown();
    return 0;
}