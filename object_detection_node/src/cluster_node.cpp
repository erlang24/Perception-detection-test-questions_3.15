#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <limits>

using PointCloudT = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudTPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

class ClusterNnode : public rclcpp::Node
{
public:
    ClusterNnode() : Node("cluster_node")
    {
        // 声明参数
        voxel_leaf_size_ = this->declare_parameter<float>("voxel_leaf_size", 0.1);  // 体素滤波 较大的值会减少点云密度 但可能会丢失小物体

        // 每个点周围 sor_mean_k_ 个邻居的平均距离 超过 sor_std_dev_mul_thresh_ 倍标准差则被视为异常值
        sor_mean_k_ = this->declare_parameter<int>("sor_mean_k", 50); 
        sor_std_dev_mul_thresh_ = this->declare_parameter<double>("sor_std_dev_mul_thresh", 1.0); 

        // 欧式聚类
        cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.5); //两个点被认为是同一个聚类的最大距离容差
        min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 20); //一个聚类中包含的最少点数
        max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 1000);

        // 订阅LiDAR点云 /rslidar_points /lidar/points
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10, std::bind(&ClusterNnode::lidarCallback, this, std::placeholders::_1));

        // 添加MarkerArray发布器
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/lidar_boxes", 10);

        RCLCPP_INFO(this->get_logger(), "Fusion node started (fusion disabled).");
    }

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received LiDAR point cloud.");

        // 将ROS点云消息转换为PCL点云
        PointCloudTPtr cloud(new PointCloudT);
        pcl::fromROSMsg(*msg, *cloud);

        // 体素滤波
        PointCloudTPtr cloud_filtered_voxel(new PointCloudT);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel_grid.filter(*cloud_filtered_voxel);
        RCLCPP_INFO(this->get_logger(), "Point cloud after voxel filter: %zu points.", cloud_filtered_voxel->size());

        // 离群点去除
        PointCloudTPtr cloud_filtered_sor(new PointCloudT);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_filtered_voxel);
        sor.setMeanK(sor_mean_k_);
        sor.setStddevMulThresh(sor_std_dev_mul_thresh_);
        sor.filter(*cloud_filtered_sor);
        RCLCPP_INFO(this->get_logger(), "Point cloud after SOR filter: %zu points.", cloud_filtered_sor->size());

        // 欧式聚类
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_filtered_sor);
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered_sor);
        ec.extract(cluster_indices);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
        for (const auto& cluster : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& index : cluster.indices) {
                cloud_cluster->push_back((*cloud_filtered_sor)[index]);
            }
            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            clusters.push_back(cloud_cluster);
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu clusters.", clusters.size());

        // 生成3D边界框
        std::vector<vision_msgs::msg::Detection3D> lidar_detections;
        for (const auto& cluster : clusters) {
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D<pcl::PointXYZ>(*cluster, min_pt, max_pt);

            vision_msgs::msg::Detection3D detection;
            detection.header = msg->header;

            // 设置3D边界框中心点
            detection.bbox.center.position.x = (min_pt.x + max_pt.x) / 2.0;
            detection.bbox.center.position.y = (min_pt.y + max_pt.y) / 2.0;
            detection.bbox.center.position.z = (min_pt.z + max_pt.z) / 2.0;

            // 设置3D边界框尺寸
            detection.bbox.size.x = std::abs(max_pt.x - min_pt.x);
            detection.bbox.size.y = std::abs(max_pt.y - min_pt.y);
            detection.bbox.size.z = std::abs(max_pt.z - min_pt.z);

            lidar_detections.push_back(detection);
        }

        publishDetections(lidar_detections);
    }



    void publishDetections(const std::vector<vision_msgs::msg::Detection3D>& detections)
    {


        // 发布MarkerArray用于可视化
        visualization_msgs::msg::MarkerArray marker_array;

        // 首先添加一个删除所有marker的消息
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.stamp = this->now();
        delete_marker.header.frame_id = "map";
        delete_marker.ns = "detection_boxes";
        delete_marker.id = 0;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        // 添加新的检测框marker
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];

            visualization_msgs::msg::Marker marker;
            marker.header = det.header;
            marker.ns = "detection_boxes";
            marker.id = i + 1;  // 从1开始，因为0用于删除所有marker
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position = det.bbox.center.position;
            marker.pose.orientation = det.bbox.center.orientation;

            marker.scale.x = det.bbox.size.x;
            marker.scale.y = det.bbox.size.y;
            marker.scale.z = det.bbox.size.z;

            // 设置颜色（红色，半透明）
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.5;

            // 增加lifetime以减少闪烁
            marker.lifetime = rclcpp::Duration::from_seconds(0.5);

            marker_array.markers.push_back(marker);
        }

        marker_pub_->publish(marker_array);
    }


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr fusion_pub_;


    // 参数
    float voxel_leaf_size_;
    int sor_mean_k_;
    double sor_std_dev_mul_thresh_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;


    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ClusterNnode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}