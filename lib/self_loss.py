import cv2
import numpy as np
import time
import torch
"""******cal_sym函数*******"""
"""输入参数：
path_sym:预测对称轴.txt路径
path_point:要计算的点云.txt路径
path_rgbd:深度图.png的路径
instrin:内参列表，如instrin = [312.9869, 241.3109, 1066.778, 1067.487]
depth_scale，深度尺度，比如10000
输出：unsupport计数点
"""


def cal_sym(path_sym, path_point, path_rgbd, instrin, depth_scale):
    sym_line = np.loadtxt(path_sym)   ##read line
    Data = np.loadtxt(path_point)     ##read point
    unsupport_count=0
    x1 = sym_line[0]                  #对称面上中心点
    y1 = sym_line[1]
    z1 = sym_line[2]
    a = sym_line[3]                   #对称面法向量
    b = sym_line[4]
    c = sym_line[5]
    depth_image = cv2.imread(path_rgbd, -1)  ##depth image
    camera_cx = instrin[0]
    camera_cy = instrin[1]
    camera_fx = instrin[2]
    camera_fy = instrin[3]
    for i in range(0, Data.shape[0]):
        d = a*x1+b*y1+c*z1
        t = (d-(a*Data[i][0]+b*Data[i][1]+c*Data[i][2]))/(a*a+b*b+c*c)
        sym_x = 2 * a * t + Data[i][0]                      ##计算得到的对称点
        sym_y = 2 * b * t + Data[i][1]
        sym_z = 2 * c * t + Data[i][2]
        n_pre = ((sym_x) * camera_fx) / sym_z + camera_cx   ##depth index
        m_pre = ((sym_y) * camera_fy) / sym_z + camera_cy
        x_pre = int(m_pre)
        y_pre = int(n_pre)
        if x_pre >= 480 or x_pre < 0 or y_pre < 0 or y_pre >= 640:  ##防止深度图索引越界
            unsupport_count = unsupport_count+1
        else:
            ori_depth = depth_image[x_pre][y_pre]   ##depth value on surface
            pre_depth = sym_z*depth_scale           ##depth value of symmetric point
            if ori_depth - pre_depth > 250:         ##相差大于2.5cm计数
                unsupport_count = unsupport_count+1

    return unsupport_count


"""******cal_sym函数*******"""
"""输入参数：
path_sym:预测对称轴.txt路径
path_point:要计算的点云.txt路径
path_rgbd:深度图.png的路径
instrin:内参列表，如instrin = [312.9869, 241.3109, 1066.778, 1067.487]
depth_scale，深度尺度，比如10000
输出：unsupport计数点
"""

def self_loss(sym,center, input_point, rgbd, instrin, depth_scale):
    Data = input_point.reshape(1000,3)     ##read point n*6
    unsupport_count=0
    center = center.reshape(3)
    sym = sym.reshape(3)
    x1 = center[0]                  #对称面上中心点
    y1 = center[1]
    z1 = center[2]
    a = sym[0]                   #对称面法向量
    b = sym[1]
    c = sym[2]
    depth_image = rgbd  ##depth image
    camera_cx = instrin[0]
    camera_cy = instrin[1]
    camera_fx = instrin[2]
    camera_fy = instrin[3]
    for i in range(0, Data.shape[0]):
        d = a*x1+b*y1+c*z1
        t = (d-(a*Data[i][0]+b*Data[i][1]+c*Data[i][2]))/(a*a+b*b+c*c)
        sym_x = 2 * a * t + Data[i][0]                      ##计算得到的对称点
        sym_y = 2 * b * t + Data[i][1]
        sym_z = 2 * c * t + Data[i][2]
        n_pre = ((sym_x) * camera_fx) / sym_z + camera_cx   ##depth index
        m_pre = ((sym_y) * camera_fy) / sym_z + camera_cy
        x_pre = int(m_pre)
        y_pre = int(n_pre)
        if x_pre >= 540 or x_pre < 0 or y_pre < 0 or y_pre >= 960:  ##防止深度图索引越界
            unsupport_count = unsupport_count+1
        else:
            ori_depth = depth_image[x_pre][y_pre]   ##depth value on surface
            pre_depth = sym_z*depth_scale           ##depth value of symmetric point
            if ori_depth - pre_depth > 30:         ##相差大于2.5cm计数
                unsupport_count = unsupport_count+1
    return unsupport_count