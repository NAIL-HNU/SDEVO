#!/usr/bin/env python3
import rospy
import numpy as np
from sdevo_msgs.msg import TimedFloat32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import os
import yaml
import torch
import time
import threading
from multiprocessing import Process

import sys
sdevo_dir = rospy.get_param('sdevoDirStr', "null")
sys.path.insert(0, sdevo_dir)
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from devo.config import cfg
from devo.devo import DEVO
from devo.utils import Timer

import tf
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf2_ros
import geometry_msgs.msg

class VoxelListener:
    def __init__(self):

        self.received_msgs_count = 0
        self.processed_msgs_count = 0

        calib_dir = rospy.get_param('calibInfoDirStr', "null")
        config_dir = rospy.get_param('configFileStr', "null")
        weights_dir = rospy.get_param('weightsFileStr', "null")

        if calib_dir == "null" or config_dir == "null" or weights_dir == "null" or sdevo_dir == "null":
            print('No directory specified!')
            exit(-1)
        else:
            print('Calibration information directory:', calib_dir)
            print('Configuration file directory:', config_dir)
            print('Weights file directory:', weights_dir)

        # create ROS subscribers
        rospy.init_node('synchronized_voxel_listener', anonymous=True)
        left_sub = Subscriber('/davis/left/voxel_grid', TimedFloat32MultiArray)
        right_sub = Subscriber('/davis/right/voxel_grid', TimedFloat32MultiArray)
        
        calib_dir_l = os.path.join(calib_dir, "left.yaml")
        calib_dir_r = os.path.join(calib_dir, "right.yaml")
        with open(calib_dir_l, 'r') as file:
            config_l = yaml.safe_load(file)
        with open(calib_dir_r, 'r') as file:
            config_r = yaml.safe_load(file)
        
        self.width, self.height = int(config_l['image_width']), int(config_l['image_height'])
        K_l = np.array(config_l['projection_matrix']['data'], dtype=np.float64).reshape((3, 4))
        K_r = np.array(config_r['projection_matrix']['data'], dtype=np.float64).reshape((3, 4))
        self.intrinsics = torch.tensor([K_l[0][0], K_l[1][1], K_l[0][2], K_l[1][2]], dtype=torch.float16).cuda()
        baseline = - K_r[0][3] / K_r[0][0]

        print('Base line:', baseline)
        print('Intrinsics:', self.intrinsics)
        
        # DEVO
        
        cfg.save_traj_dir = sdevo_dir + "/output/poses_tum_format.txt"
        self.ori_pose_dir = cfg.save_traj_dir
        self.interpolated_pose_dir = sdevo_dir + "/output/poses_interpolated.txt"
        cfg.merge_from_file(config_dir)
        cfg.baseline = (float)(baseline)
        print(cfg)

        torch.manual_seed(1234)
        self.slam = DEVO(cfg, weights_dir, evs=True, ht=self.height, wd=self.width, viz=False, viz_flow=False)
        

        self.image_l_buffer = torch.empty((5, self.height, self.width), dtype=torch.float16).cuda()
        self.image_r_buffer = torch.empty((5, self.height, self.width), dtype=torch.float16).cuda()

        sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=2000, slop=0.05)
        sync.registerCallback(self.callback)

        self.last_callback_time = rospy.Time.now()
        self.timer = rospy.Timer(rospy.Duration(3), self.timeout_callback)

        self.pose_pub = rospy.Publisher('/camera_pose', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/trajectory', Path, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.path = Path()
        self.path.header.frame_id = "map"  
        self.count = 0

        print('DEVO initialized')
        

    def pub_pose(self, curr_pose):
        pose = PoseStamped()
        pose.header.frame_id = "map"  # Using the map reference frame
        pose.header.stamp = rospy.Time.from_sec(curr_pose[0])

        # Set position and orientation from current pose data
        pose.pose.position.x = curr_pose[1]
        pose.pose.position.y = curr_pose[2]
        pose.pose.position.z = curr_pose[3]
        pose.pose.orientation.x = curr_pose[4]
        pose.pose.orientation.y = curr_pose[5]
        pose.pose.orientation.z = curr_pose[6]
        pose.pose.orientation.w = curr_pose[7]

        # Publish the pose
        self.pose_pub.publish(pose)
        self.count += 1  # Increment message counter

        # Add pose to path and publish
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)

        # Create TF transform from map to camera frame
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Reference frame
        transform.child_frame_id = "camera"  # Camera frame

        # Set transform position and orientation
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z

        transform.transform.rotation.x = pose.pose.orientation.x
        transform.transform.rotation.y = pose.pose.orientation.y
        transform.transform.rotation.z = pose.pose.orientation.z
        transform.transform.rotation.w = pose.pose.orientation.w

        # Broadcast the TF transform
        self.tf_broadcaster.sendTransform(transform)



    @torch.no_grad()
    def callback(self, voxel_left, voxel_right):
        print("get new voxel pair -------------------------------")

        self.last_callback_time = rospy.Time.now()
        print("last callback time: ", self.last_callback_time)
        start_time = time.time()

        self.image_l_buffer.copy_(torch.tensor(voxel_left.data.data, dtype=torch.float16).view(5, self.height, self.width))
        self.image_r_buffer.copy_(torch.tensor(voxel_right.data.data, dtype=torch.float16).view(5, self.height, self.width))
        t = voxel_left.header.stamp.to_nsec() / 1000
       
        self.slam(t, self.image_l_buffer, self.intrinsics, self.image_r_buffer, self.intrinsics, scale=1)

        t_callback = time.time() - start_time
        print("used time of callback: ", t_callback)


    def run(self):
        rospy.spin()
    

    def timeout_callback(self, event):
        # Check if more than 5 seconds have passed since the last message was received
        current_time = rospy.Time.now()
        time_diff = (current_time - self.last_callback_time).to_sec()
        print("Current time: ", current_time)
        if time_diff > 5:
            try:
                rospy.loginfo("5 seconds without messages, executing the next steps...")
                self.slam.save_traj()
            except ValueError:
                pass
            
def interpolate_pose_file(self, input_file, output_file):
        if not (os.path.exists(input_file)):
            print(f"Original pose file does not exist: {input_file}")
            return
        
        # Read data from file
        data = np.loadtxt(input_file)

        # Extract timestamps, positions, and quaternions
        timestamps = data[:, 0]
        positions = data[:, 1:4]  # x, y, z trajectory coordinates
        quaternions = data[:, 4:]  # qx, qy, qz, qw quaternions

        # Remove duplicate timestamps (if any) and ensure timestamps are increasing
        unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
        positions = positions[unique_indices]
        quaternions = quaternions[unique_indices]

        # Verify that timestamps are strictly increasing
        if not np.all(np.diff(unique_timestamps) > 0):
            raise ValueError("Timestamps must be strictly increasing. Please check input data.")

        # Define target time points for interpolation (1000Hz = 1000 samples per second)
        target_time = np.arange(unique_timestamps[0], unique_timestamps[-1], 1/1000)

        # Interpolate positions using linear interpolation
        interp_x = interp1d(unique_timestamps, positions[:, 0], kind='linear')
        interp_y = interp1d(unique_timestamps, positions[:, 1], kind='linear')
        interp_z = interp1d(unique_timestamps, positions[:, 2], kind='linear')

        # Calculate interpolated positions
        new_positions = np.column_stack((
            interp_x(target_time),
            interp_y(target_time),
            interp_z(target_time)
        ))

        # Use Spherical Linear Interpolation (Slerp) for quaternion interpolation
        rotations = R.from_quat(quaternions)  # Convert quaternions to rotation objects
        slerp = Slerp(unique_timestamps, rotations)  # Create Slerp interpolation object
        new_rotations = slerp(target_time)  # Interpolate quaternions at target times

        # Get interpolated quaternions
        new_quaternions = new_rotations.as_quat()

        # Create TUM format data
        tum_data = []
        for i, t in enumerate(target_time):
            # Use original timestamp range for interpolated timestamps
            timestamp = unique_timestamps[0] + t - target_time[0]  # Ensure timestamps start from original
            # Use interpolated quaternions and positions
            tum_data.append([timestamp, new_positions[i, 0], new_positions[i, 1], new_positions[i, 2],
                            new_quaternions[i, 0], new_quaternions[i, 1], new_quaternions[i, 2], new_quaternions[i, 3]])

        # Save interpolated trajectory in TUM format to output file
        np.savetxt(output_file, tum_data, fmt='%.6f', delimiter=' ', header='# timestamp x y z qx qy qz qw', comments='')

        print(f"Interpolated pose trajectory saved as TUM format file: {output_file}")


if __name__ == '__main__':
    rospy.set_param("/use_sim_time", False)
    listener = VoxelListener()
    listener.run()