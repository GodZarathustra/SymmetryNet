import math
import numpy as np
def rotate(pt, v1, v2, theta):
    m = np.zeros((4, 4))
    a = v1[0]
    b = v1[1]
    c = v1[2]

    p = ((v2 - v1) / np.linalg.norm(v2 - v1)).reshape(-1)
    u = p[0]
    v = p[1]
    w = p[2]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w
    au = a * u
    av = a * v
    aw = a * w
    bu = b * u
    bv = b * v
    bw = b * w
    cu = c * u
    cv = c * v
    cw = c * w

    costheta = math.cos(theta * math.pi / 180)
    sintheta = math.sin(theta * math.pi / 180)

    m[0][0] = uu + (vv + ww) * costheta
    m[0][1] = uv * (1 - costheta) + w * sintheta
    m[0][2] = uw * (1 - costheta) - v * sintheta
    m[0][3] = 0

    m[1][0] = uv * (1 - costheta) - w * sintheta
    m[1][1] = vv + (uu + ww) * costheta
    m[1][2] = vw * (1 - costheta) + u * sintheta
    m[1][3] = 0

    m[2][0] = uw * (1 - costheta) + v * sintheta
    m[2][1] = vw * (1 - costheta) - u * sintheta
    m[2][2] = ww + (uu + vv) * costheta
    m[2][3] = 0

    m[3][0] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta
    m[3][1] = (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta
    m[3][2] = (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta
    m[3][3] = 1

    transM = m
    Vrot_x = pt[0] * transM[0][0] + pt[1] * transM[1][0] + pt[2] * transM[2][0] + transM[3][0]
    Vrot_y = pt[0] * transM[0][1] + pt[1] * transM[1][1] + pt[2] * transM[2][1] + transM[3][1]
    Vrot_z = pt[0] * transM[0][2] + pt[1] * transM[1][2] + pt[2] * transM[2][2] + transM[3][2]

    return np.array([Vrot_x,Vrot_y,Vrot_z])

def ref_vrf(sym,center, input_point, rgbd, instrin, depth_scale,im_h,im_w,thresh):
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

        x_pre = m_pre
        y_pre = n_pre
        if x_pre >= im_h or x_pre < 0 or y_pre < 0 or y_pre >= im_w :  ##防止深度图索引越界
            unsupport_count = unsupport_count+1
        elif(x_pre !=x_pre) or (y_pre != y_pre):
            unsupport_count = unsupport_count + 1
        else:
            x_pre = int(x_pre)
            y_pre = int(y_pre)
            ori_depth = depth_image[x_pre][y_pre]   ##depth value on surface
            pre_depth = sym_z*depth_scale           ##depth value of symmetric point
            if (ori_depth - pre_depth) > (thresh*depth_scale):
                unsupport_count = unsupport_count+1
    return unsupport_count

def rot_vrf(v1, v2,input_point, rgbd, instrin, depth_scale,im_h,im_w,thresh):
    Data = input_point.reshape(1000, 3)     ##read point
    unsupport_count=0
    v1x = v1[0]                  #对称轴上的一个点
    v1y = v1[1]
    v1z = v1[2]
    v2x = v2[0]
    v2y = v2[1]
    v2z = v2[2]
    depth_image = rgbd  ##depth image
    instrin = instrin[0]
    camera_cx = instrin[0]
    camera_cy = instrin[1]
    camera_fx = instrin[2]
    camera_fy = instrin[3]

    for i in range(0, Data.shape[0]):
        for theta in range(0,360,30):
            sym_pt = rotate(np.array([Data[i][0],Data[i][1],Data[i][2]]), np.array([v1x,v1y,v1z]), np.array([v2x,v2y,v2z]), theta)
            sym_x = sym_pt[0]                   ##计算得到的对称点
            sym_y = sym_pt[1]
            sym_z = sym_pt[2]
            n_pre = ((sym_x) * camera_fx) / sym_z + camera_cx   ##depth index
            m_pre = ((sym_y) * camera_fy) / sym_z + camera_cy
            x_pre = int(m_pre)
            y_pre = int(n_pre)
            if x_pre >= im_h or x_pre < 0 or y_pre < 0 or y_pre >= im_w:  ##防止深度图索引越界
                unsupport_count = unsupport_count+1
            else:
                ori_depth = depth_image[x_pre][y_pre]   ##depth value on surface
                pre_depth = sym_z*depth_scale           ##depth value of symmetric point
                if (ori_depth - pre_depth) > (thresh*depth_scale):
                    unsupport_count = unsupport_count+1
    return unsupport_count/12
