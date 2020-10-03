import numpy as np
import math
import random

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        print(s)
        print('estimate:', m,)
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def reflect(Data,cent,sym):
    Data = Data.reshape(1000,3)
    cent = cent.reshape(3)
    sym = sym.reshape(-1, 3)
    reflect_points = np.zeros((1000,sym.shape[0],3))
    for j in range(sym.shape[0]):
        x1 = cent[0]
        y1 = cent[1]
        z1 = cent[2]
        a = sym[j,0]
        b = sym[j,1]
        c = sym[j,2]
        ref_point = np.zeros(Data.shape)
        for i in range(0, Data.shape[0]):
            d = a*x1+b*y1+c*z1
            t = (d-(a*Data[i][0]+b*Data[i][1]+c*Data[i][2]))/(a*a+b*b+c*c)
            sym_x = 2 * a * t + Data[i][0]
            sym_y = 2 * b * t + Data[i][1]
            sym_z = 2 * c * t + Data[i][2]
            ref_point[i,:] = np.array([sym_x, sym_y, sym_z])
        reflect_points[:,j,:] = ref_point
    return reflect_points

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