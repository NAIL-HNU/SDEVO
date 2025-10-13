#!/usr/bin/env python
import rospy
import pickle
import os
import torch  # 如果使用 PyTorch
from dvs_msgs.msg import TimedFloat32MultiArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import time

def load_timestamps_and_files(timestamp_file, data_dir_left, data_dir_right):
    timestamps = []
    file_paths_left = []
    file_paths_right = []
    skip = False
    with open(timestamp_file, 'r') as f:
        for line in f:
            if skip == True:
                skip = False
                # continue
            else:
                skip = True
            timestamp = float(line.strip())
            timestamps.append(timestamp)
            # 左目和右目的文件路径
            file_name_left = f"{data_dir_left}/{timestamp}.pkl"
            file_name_right = f"{data_dir_right}/{timestamp}.pkl"
            if os.path.exists(file_name_left):
                file_paths_left.append(file_name_left)
            else:
                rospy.logwarn(f"Warning: Left file {file_name_left} not found.")
                file_paths_left.append(None)

            if os.path.exists(file_name_right):
                file_paths_right.append(file_name_right)
            else:
                rospy.logwarn(f"Warning: Right file {file_name_right} not found.")
                file_paths_right.append(None)
    return timestamps, file_paths_left, file_paths_right

def publish_voxel_data():
    # ROS 初始化
    rospy.init_node('voxel_publisher', anonymous=True)
    voxel_pub_left = rospy.Publisher('/davis/left/voxel_grid', TimedFloat32MultiArray, queue_size=10)
    voxel_pub_right = rospy.Publisher('/davis/right/voxel_grid', TimedFloat32MultiArray, queue_size=10)

    # 文件路径配置
    # timestamp_file = "/media/njk/njk/data_for_devo/processed/04a/voxels_left/voxel_time.txt"
    # data_dir_left = "/media/njk/njk/data_for_devo/processed/04a/voxels_left"
    # data_dir_right = "/media/njk/njk/data_for_devo/processed/04a/voxels_right"
    timestamp_file = "/media/njk/njk/data_for_devo/processed/04b/voxels_left/voxel_time.txt"
    data_dir_left = "/media/njk/njk/data_for_devo/processed/04b/voxels_left"
    data_dir_right = "/media/njk/njk/data_for_devo/processed/04b/voxels_right"
    # 读取时间戳和对应文件
    timestamps, file_paths_left, file_paths_right = load_timestamps_and_files(
        timestamp_file, data_dir_left, data_dir_right)

    # 发布图像数据
    rate = rospy.Rate(2)  # 设置为 10Hz 发布频率

    # 循环发布每一对 voxel 数据
    for i in range(len(timestamps)):
        if rospy.is_shutdown():  # 检查是否有 ROS shutdown 信号
            rospy.loginfo("Shutting down publisher.")
            break  # 如果收到 shutdown 信号，退出循环

        # 读取左目数据
        if file_paths_left[i]:
            with open(file_paths_left[i], 'rb') as f:
                data_left = pickle.load(f)
                voxel_left, intrinsics_left, t_left = data_left
        else:
            voxel_left = None

        # 读取右目数据
        if file_paths_right[i]:
            with open(file_paths_right[i], 'rb') as f:
                data_right = pickle.load(f)
                voxel_right, intrinsics_right, t_right = data_right
        else:
            voxel_right = None

        # 处理左目数据
        if voxel_left is not None:
            if isinstance(voxel_left, torch.Tensor):
                voxel_left = voxel_left.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            voxel_data_left = voxel_left.flatten().tolist()

            # 创建左目消息
            voxel_msg_left = TimedFloat32MultiArray()
            voxel_msg_left.header.stamp = rospy.Time.from_sec(timestamps[i] / 1e6)  # 时间戳微秒转秒

            # 创建 Float32MultiArray 用于 data，并设置 layout
            voxel_msg_left.data = Float32MultiArray()
            voxel_msg_left.data.data = voxel_data_left
            voxel_msg_left.data.layout = MultiArrayLayout()
            voxel_msg_left.data.layout.dim = [
                MultiArrayDimension(label="depth", size=5, stride=5 * 480 * 640),
                MultiArrayDimension(label="height", size=480, stride=480 * 640),
                MultiArrayDimension(label="width", size=640, stride=640)
            ]

            # 发布左目数据
            voxel_pub_left.publish(voxel_msg_left)
            rospy.loginfo(f"Published voxel left data for timestamp: {timestamps[i]}")

        # 处理右目数据
        if voxel_right is not None:
            if isinstance(voxel_right, torch.Tensor):
                voxel_right = voxel_right.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            voxel_data_right = voxel_right.flatten().tolist()

            # 创建右目消息
            voxel_msg_right = TimedFloat32MultiArray()
            voxel_msg_right.header.stamp = rospy.Time.from_sec(timestamps[i] / 1e6)  # 时间戳微秒转秒

            # 创建 Float32MultiArray 用于 data，并设置 layout
            voxel_msg_right.data = Float32MultiArray()
            voxel_msg_right.data.data = voxel_data_right
            voxel_msg_right.data.layout = MultiArrayLayout()
            voxel_msg_right.data.layout.dim = [
                MultiArrayDimension(label="depth", size=5, stride=5 * 480 * 640),
                MultiArrayDimension(label="height", size=480, stride=480 * 640),
                MultiArrayDimension(label="width", size=640, stride=640)
            ]

            # 发布右目数据
            voxel_pub_right.publish(voxel_msg_right)
            rospy.loginfo(f"Published voxel right data for timestamp: {timestamps[i]}")

        # 控制发布间隔：固定 10Hz
        rate.sleep()  # 这里会强制每秒钟发布 10 次消息

if __name__ == '__main__':
    try:
        publish_voxel_data()
    except rospy.ROSInterruptException:
        pass
