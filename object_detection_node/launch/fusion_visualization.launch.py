from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 声明 cluster_node 的参数
    voxel_leaf_size = LaunchConfiguration('voxel_leaf_size', default='0.1')
    sor_mean_k = LaunchConfiguration('sor_mean_k', default='50')
    sor_std_dev_mul_thresh = LaunchConfiguration('sor_std_dev_mul_thresh', default='1.0')
    cluster_tolerance = LaunchConfiguration('cluster_tolerance', default='0.5')
    min_cluster_size = LaunchConfiguration('min_cluster_size', default='10')
    max_cluster_size = LaunchConfiguration('max_cluster_size', default='1000')
    iou_threshold = LaunchConfiguration('iou_threshold', default='0.5')

    # 创建 object_detection_node 节点
    object_detection_node = Node(
        package='object_detection_node',
        executable='object_detection_node',
        name='object_detection_node'
    )

    # 创建 cluster_node 节点
    cluster_node = Node(
        package='object_detection_node',
        executable='cluster_node',
        name='cluster_node',
        parameters=[{
            'voxel_leaf_size': voxel_leaf_size,
            'sor_mean_k': sor_mean_k,
            'sor_std_dev_mul_thresh': sor_std_dev_mul_thresh,
            'cluster_tolerance': cluster_tolerance,
            'min_cluster_size': min_cluster_size,
            'max_cluster_size': max_cluster_size,
            'iou_threshold': iou_threshold
        }]
    )

    return LaunchDescription([
        # 声明 cluster_node 的参数
        DeclareLaunchArgument('voxel_leaf_size', default_value='0.1'),
        DeclareLaunchArgument('sor_mean_k', default_value='50'),
        DeclareLaunchArgument('sor_std_dev_mul_thresh', default_value='1.0'),
        DeclareLaunchArgument('cluster_tolerance', default_value='0.5'),
        DeclareLaunchArgument('min_cluster_size', default_value='10'),
        DeclareLaunchArgument('max_cluster_size', default_value='1000'),
        DeclareLaunchArgument('iou_threshold', default_value='0.5'),

        # 启动节点
        object_detection_node,
        cluster_node
    ])