import torch
import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
       poses_l2r, target_l2r, weight_l2r, ii_lr, jj_lr, kk_lr, t0, t1, iterations=2):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                           poses_l2r.data, target_l2r, weight_l2r, ii_lr, jj_lr, kk_lr, t0, t1, iterations)