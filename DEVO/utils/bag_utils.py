import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from io import StringIO
import sys
from PIL import Image
import io
import tqdm as tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from event_utils import to_voxel_grid
import pickle


def read_first_evs_from_rosbag(bag, evtopic):
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec()/1e3
            break
        break
    return t0_us


def read_evs_from_rosbag(bag, evtopic, H=180, W=240):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            p = 1 if ev.polarity else 0
            evs.append([ev.x, ev.y, ev.ts.to_nsec()/1e3, p])
            # assert ev.x < W and ev.y < H # DEBUG
        progress_bar.update(1)

        # if len(evs) > 1000:
        #     break
    return np.array(evs) # (N, 4)

def read_evs_from_rosbag_and_write(bag, evtopic, f, H=180, W=240):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    first = True
    t0 = 0
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            if first:
                t0 = 0
                first = False
            p = 1 if ev.polarity else 0
            f.write(f"{(ev.ts.to_nsec()/1e3 - t0):.06f} {int(ev.x)} {int(ev.y)} {int(p)}\n")
            # assert ev.x < W and ev.y < H # DEBUG
        progress_bar.update(1)
        if len(evs) > 1000:
            break
    return

def get_ecd_from_rosbag_and_write(bag, evtopic, intrinsics, rectify_map, indir, side, H=180, W=240, delta = 50000):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    first = True
    next_voxel = False
    t0 = 0
    voxel_t = 0
    voxel_dir = os.path.join(indir, "voxels_" + side)
    if not os.path.exists(voxel_dir):
            os.makedirs(voxel_dir)
    index_filepath = os.path.join(indir, "voxels_" + side + "/voxel_time.txt")
    voxel_num = 0
    with open(index_filepath, 'w') as index_file:
        for topic, msg, t in bag.read_messages(evtopic):
            for ev in msg.events:
                if first:
                    t0 = ev.ts.to_nsec()/1e3
                    voxel_t = t0
                    first = False
                if next_voxel:
                    voxel_t = voxel_t + delta
                    voxel_num += 1
                    next_voxel = False

                p = 1 if ev.polarity else 0
                evs.append([ev.ts.to_nsec()/1e3, int(ev.x), int(ev.y), int(p)])
                
                if (ev.ts.to_nsec()/1e3 - voxel_t) > delta:
                    evs_batch = np.array(evs)
                    intrinsics = np.array(intrinsics)

                    if rectify_map is not None:
                        rect = rectify_map[evs_batch[:, 2].astype(np.int32), evs_batch[:, 1].astype(np.int32)]
                        voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
                    else:
                        voxel = to_voxel_grid(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
                    mid_t = voxel_t + delta / 2
                    data = [voxel, intrinsics, mid_t]
                    voxel_file = os.path.join(indir, "voxels_" + side + "/" + str(mid_t) + ".pkl")
                    with open(voxel_file, 'wb') as f1:
                        pickle.dump(data, f1)
                    index_file.write(f"{mid_t}\n")
                    evs = []
                    next_voxel = True
                    print(f"Voxel {voxel_num} saved")
            #     if (voxel_num > 30):
            #         break
            # if (voxel_num > 30):
            #         break
            
            progress_bar.update(1)
        
    return

def read_H_W_from_bag(bag, imgtopic):
    for topic, msg, t in bag.read_messages(imgtopic):
        H, W = msg.height, msg.width
        print(f"Read H, W from bag: {H}, {W}")
        return H, W

def read_images_from_rosbag(bag, imgtopic, H=180, W=240):
    imgs = []
    
    progress_bar = tqdm.tqdm(total=bag.get_message_count(imgtopic))
    for topic, msg, t in bag.read_messages(imgtopic):
        img_str = str(msg)
        img_str = img_str[img_str.find("data")+6:]
        img_str = img_str[1:-1].split(',')
        pixel_values = [int(v) for v in img_str]
        image_array = np.array(pixel_values, dtype=np.uint8)
        image_array = image_array.reshape((msg.height, msg.width))
        imgs.append(image_array)
        progress_bar.update(1)

        if abs(H- msg.height) > 2 or abs(W-msg.width) > 2:
            print(f"WARNING: H, W mismatch: {msg.height}, {msg.width}, {H}, {W}")    

        # if len(imgs) > 50: # TODO: remove!
        #     break
    return imgs


def read_tss_us_from_rosbag(bag, imgtopic):
    tss_us = []
    for topic, msg, t in bag.read_messages(imgtopic):
        tss_us.append(msg.header.stamp.to_nsec() / 1e3)
    return tss_us


def read_poses_from_rosbag(bag, posestopic, T_marker_cam0, T_cam0_cam1):
    progress_bar = tqdm.tqdm(total=bag.get_message_count(posestopic))

    poses = []
    tss_us_gt = []
    for topic, msg, t in bag.read_messages(posestopic):
        if msg._type == "nav_msgs/Odometry":
            ps = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        else:
            ps = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        T_world_marker = np.eye(4)
        T_world_marker[:3, 3] = ps[:3]
        T_world_marker[:3, :3] = R.from_quat(ps[3:]).as_matrix()
        
        T_world_cam = T_world_marker @ T_marker_cam0
        T_world_cam = T_world_cam @ T_cam0_cam1

        T_world_cam = np.concatenate((T_world_cam[:3, 3], R.from_matrix(T_world_cam[:3, :3]).as_quat()))
        poses.append(T_world_cam)

        tss_us_gt.append(msg.header.stamp.to_nsec() / 1e3)
                     
        progress_bar.update(1)
    return np.array(poses), tss_us_gt

def read_calib_from_bag(bag, imtopic):
    for topic, msg, t in bag.read_messages(imtopic):
        K = msg.K
        break
    return K


def read_t0us_evs_from_rosbag(bag, evtopic):
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec()/1e3
            break
        break
    return t0_us