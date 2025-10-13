import cv2
import numpy as np
import os
import sys
# 加载拼接后的图像
image = cv2.imread('/home/njk/ros/SDEVO_no_speedup/src/sdevo/DEVO/output/1111.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    raise FileNotFoundError("无法加载图像，请检查文件路径。")

# 将图像分割为左右目
height, width = image.shape
left_image = image[:, :width // 2]
right_image = image[:, width // 2:]

# 将左右目图像转换为彩色，以便绘制连线
left_image_color = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
right_image_color = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

# 拼接左右目图像以便显示结果
combined_image = np.hstack((left_image_color, right_image_color))

# 选取一些数值较大的点作为关键点
def select_keypoints(image, num_points=100):
    # 使用阈值提取亮度较高的区域
    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    
    # 找到所有的非零点
    keypoints = np.column_stack(np.where(thresholded > 0))
    
    # 随机选择一些点，避免过于集中
    if len(keypoints) > num_points:
        keypoints = keypoints[np.random.choice(len(keypoints), num_points, replace=False)]
    
    return keypoints

left_keypoints = select_keypoints(left_image)

# 设置Block Matching的参数
block_size = 50  # 定义块的大小
search_range = 80  # 搜索范围

# ZNCC 计算相似度
def zncc(block1, block2):
    block1_mean = np.mean(block1)
    block2_mean = np.mean(block2)
    numerator = np.sum((block1 - block1_mean) * (block2 - block2_mean))
    denominator = np.sqrt(np.sum((block1 - block1_mean) ** 2) * np.sum((block2 - block2_mean) ** 2))
    if denominator == 0:
        return -1  # 如果分母为0，返回-1表示相似度最低
    return numerator / denominator

# 进行Block Matching并计算相似度
disparities = []

for point in left_keypoints:
    y, x = point
    best_offset = 0
    best_zncc = -1
    
    # 获取左目图像中的块
    if y - block_size // 2 < 0 or y + block_size // 2 >= height or x - block_size // 2 < 0 or x + block_size // 2 >= width // 2:
        continue
    left_block = left_image[y - block_size // 2:y + block_size // 2 + 1, x - block_size // 2:x + block_size // 2 + 1]
    
    # 在右目图像中搜索匹配块
    for offset in range(-search_range, search_range + 1):
        x_right = x + offset
        if x_right - block_size // 2 < 0 or x_right + block_size // 2 >= width // 2:
            continue
        right_block = right_image[y - block_size // 2:y + block_size // 2 + 1, x_right - block_size // 2:x_right + block_size // 2 + 1]
        
        # 计算ZNCC相似度
        similarity = zncc(left_block, right_block)
        if similarity > best_zncc:
            best_zncc = similarity
            best_offset = offset
    
    disparities.append((x, y, best_offset))

# 可视化视差结果
for (x, y, disparity) in disparities:
    x_right = x + disparity
    # 在拼接后的图像上绘制关键点
    cv2.circle(combined_image, (x, y), 2, (0, 0, 255), -1)  # 左目关键点
    cv2.circle(combined_image, (x_right + width // 2, y), 2, (0, 255, 0), -1)  # 右目关键点
    # 绘制连线，连接左目和右目对应的点
    cv2.line(combined_image, (x, y), (x_right + width // 2, y), (255, 0, 0), 1)

cv2.imshow('Selected Keypoints with Disparities', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
