import cv2
import random
import math
import numpy as np
from config import CONFIG

"""
1. crop to center
2. rotation
3. flip
h5文件已经对数据做了处理，每个样本是单个人
"""
# [0.7,1.3)
def resize(orgImgs, centers, gt_joints_list):
    output_imgs = []
    output_centers = []
    output_joints_list = []
    for i in range(len(orgImgs)):  # 当前batch内的样本个数
        img = cv2.imread(CONFIG.image_dir + orgImgs[i].decode('utf-8'))  # orgImg是unicode编码
        rdn = random.random()       #[0,1)范围内的浮点随机数
        rdn = (1.3-0.7)*rdn + 0.7
        img = cv2.resize(img, (0,0), fx=rdn, fy=rdn, interpolation=cv2.INTER_LANCZOS4)  #多尺度resize到[0.7,1.3)范围
        output_imgs.append(img)
        center_x = float(np.float64(centers[i][0])) * rdn
        center_y = float(np.float64(centers[i][1])) * rdn
        output_centers.append(np.array([center_x, center_y]))      #中心点resize
        output_joints_list.append(np.array(gt_joints_list[i])*rdn)
    return output_imgs, output_centers, output_joints_list


#以人体中心crop图片 targetSize*targetSize 大小
def crop_to_center(orgImgs, targetSize, scales, centers, gt_joints_list, is_train):
    output_imgs = []
    output_joints_list = []
    for i in range(len(orgImgs)):       #当前batch内的样本个数
        crop_img = np.ones((368, 368, 3)) * 128
        img = orgImgs[i]
        if not is_train:        #train的时候，orgImgs已经经过了resize，已经不只是图像名字了
            img = cv2.imread(CONFIG.image_dir + img.decode( 'utf-8'))  # orgImg是unicode编码

        s = 200 * scales[i] / targetSize  # 相对于368的scale的大小
        img_resized = cv2.resize(img, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_LANCZOS4)  # 相对于368的1尺度
        center_x = float(np.float64(centers[i][0]))/s
        center_y = float(np.float64(centers[i][1]))/s
        half = targetSize / 2  # 目标尺寸的一半

        # 图像中的截取的部分
        x0 = max(0, center_x - half)
        y0 = max(0, center_y - half)
        x1 = min(img_resized.shape[1], center_x + half)
        y1 = min(img_resized.shape[0], center_y + half)

        # 目标中需要填充的部分
        fromx0 = max(0, half - center_x)
        fromx1 = (img_resized.shape[1] - center_x + half) if (img_resized.shape[1] - center_x) < half else 2 * half
        fromy0 = max(0, half - center_y)
        fromy1 = (half + img_resized.shape[0] - center_y) if (center_y + half) > img_resized.shape[0] else 2 * half

        crop_img[int(fromy0 + 0.5):int(fromy1 + 0.5), int(fromx0 + 0.5):int(fromx1 + 0.5), :] = \
            img_resized[int(y0 + 0.5):int(y1 + 0.5), int(x0 + 0.5):int(x1 + 0.5), :]
        output_imgs.append(crop_img)

        #调整ground truth的关节点坐标
        joints = gt_joints_list[i]          #当前这个人的关节坐标
        joints = np.array(joints)*1/s       #1/s是图片的缩放比例，坐标也要等比缩放
        output_joints = []
        for coord in joints:
            if coord[0]<=0.0 and coord[1]<=0.0:     #无效关节，没有在图片中
                output_joints.append([-1000,-1000])
            else:
                output_joints.append([coord[0]+half-center_x, coord[1]+half-center_y])  #crop之后需要调整坐标
        output_joints_list.append(output_joints)
    return output_imgs, output_joints_list

#图片旋转
def rotate(imgs, joints_list):
    deg = random.uniform(-40.0, 40.0)       #论文中是-40~40°
    rotate_imgs = []
    rotate_joints_list = []
    for i in range(len(imgs)):      #当前batch内的样本个数
        img = imgs[i]
        joints = joints_list[i]
        rot_matrix = cv2.getRotationMatrix2D((0,0), deg, 1)
        img = cv2.warpAffine(img, rot_matrix, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)    #已经旋转了的图片
        adjust_joint = []
        for coord in joints:
            x, y = _rotate_coord((img.shape[1], img.shape[0]), coord, deg)
            adjust_joint.append((x, y))
        rotate_imgs.append(img)
        rotate_joints_list.append(adjust_joint)
    return rotate_imgs, rotate_joints_list

def _rotate_coord(shape, point, angle):      #newxy是center的减小值，由于旋转后crop，中点的像素值没变，但是坐标值变了
    angle = -1 * angle / 180.0 * math.pi
    ox, oy = shape
    px, py = point      #当前关节点坐标
    ox /= 2         #原图的中点
    oy /= 2
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)      #坐标旋转变换，逆时针为正方向
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx + 0.5), int(qy + 0.5)

#水平翻转图片
def flip(imgs, joints_list):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return imgs, joints_list    #随机翻转
    flip_imgs = []
    flip_joints_list = []
    for i in range(len(imgs)):      #当前batch内的样本个数
        img = imgs[i]
        joints = joints_list[i]
        img = cv2.flip(img, 1)      #flipCode大于0，水平翻转
        flip_imgs.append(img)
        adjust_joint = []
        for coord in joints:
            adjust_joint.append((img.shape[1] - coord[0], coord[1]))  #x坐标相对于中点对称，y坐标不变
        flip_joints_list.append(adjust_joint)
    return flip_imgs, flip_joints_list




