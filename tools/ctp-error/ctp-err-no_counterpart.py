#coding=utf-8
import argparse
import os
import sys
sys.path.append('/home/dell/yifeis/pose_estimation/densefusion_syn_test/')
import math
import random
import time
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.shapenet.dataset import PoseDataset as PoseDataset_ycb
from lib.network_foot import PoseNet, PoseRefineNet
from lib.loss_foot import Loss
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as skc  # 密度聚类

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='/home/dell/yifeis/pose_estimation/render/render_dy/render_wt_pt_proj/data/',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--workers', type=int, default=20, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='pose_model_18_2.179880025753353.pth', help='resume PoseNet model')  # pose_model_current.pth  pose_model_58_1.059006152053674.pth  pose_model_39_1.2116587293644747.pth pose_model_49_1.2153524918986691.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

# proj_dir = '/home/demian/densefusion_rt_cmp_oa/'
# proj_dir = '/home/demian/densefusion_syn/'
proj_dir = '/home/dell/yifeis/pose_estimation/densefusion_syn_test/'
sym_list = training_cat = [0, 3, 7, 10, 11, 12, 14, 16, 17, 19, 21, 22]
visual_dir = proj_dir+'datasets/ycb/dataset_config/visual_data_list.txt'
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prcurve(DIST_THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/foot'  # folder to save trained models
        opt.log_dir = proj_dir + 'visualization'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training

    estimator = torch.nn.DataParallel(PoseNet(num_points=opt.num_points, num_obj=opt.num_objects))
    # torch.nn.DataParallel(module=estimator, device_ids=device_ids)
    estimator.cuda(device_ids[0])
    torch.nn.DataParallel(module=estimator, device_ids=device_ids)
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        estimator = estimator.module
        opt.refine_start = False
        opt.decay_start = False
        # optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        # opt.w *= opt.w_rate
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        opt.w *= opt.w_rate

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('syn_train', opt.num_points, False, opt.dataset_root, opt.noise_trans, opt.refine_start)
        test_dataset = PoseDataset_ycb('syn_frame', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh)


    estimator.eval()
    total_fream = 0

    total_num = 0
    pred_c = 1
    ctp_dis_list = []
    for j, data in enumerate(testdataloader, 0):

        points, choose, img, idx, target_s, target_num, target_mode,_ = data  # the original version
        # if idx not in sym_list:
        #    continue
        points, choose, img, idx, target_s, target_num, target_mode = Variable(points).cuda(), \
                                                                      Variable(choose).cuda(), \
                                                                      Variable(img).cuda(), \
                                                                      Variable(idx).cuda(), \
                                                                      Variable(target_s).cuda(), \
                                                                      Variable(target_num).cuda(), \
                                                                      Variable(target_mode).cuda()

        pred_cent, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points,
                                                                                 choose)

        target_mode = target_mode.data.cpu().numpy().reshape(-1)
        # occ = occ.data.cpu().numpy()[0]
        #  ((occ < 0.7)or(occ>0.8))  : 70-80
        #  ((occ <0.6)or(occ>0.7))  :60-70
        #  (occ>0.6)  :<60
        if (target_mode == 1) :
            continue

        total_fream += 1
        tmp_s = target_s.data.cpu().numpy()
        tmp_s = tmp_s.reshape(-1, 3)
        target_cent = tmp_s[0, :]
        target_sym = tmp_s[1:, :] # target symmetry point
        len_target_sym = np.linalg.norm((target_sym-target_cent),axis=1)

        points = points.view(1000, 3)
        points = points.detach().cpu().data.numpy()

        pred_cent = pred_cent.detach().cpu().data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        cent_pred = (points + pred_cent)

        pred_num = pred_num.view(1000, 3).detach()
        my_num = torch.mean(pred_num, dim=0)
        my_num = my_num.data.cpu().numpy()

        pred_ref = pred_foot_ref.detach().cpu().data.numpy() # (1,1000,9)
        pred_ref = pred_ref.reshape(1000, -1, 3)

        pred_foot_ref = pred_foot_ref.view(1000, -1, 3)
        pred_foot_ref = pred_foot_ref.detach().cpu().data.numpy()

        target_sym = target_sym - target_cent
        # target_sym = target_sym.reshape(-1,3)

        my_sym = pred_ref
        my_norm = np.zeros(my_sym.shape)
        for i in range(my_sym.shape[1]):
            for k in range(3):
                my_norm[:, i, k] = my_sym[:, i, k] / np.linalg.norm(my_sym[:, i, :], axis=1)

        mean_norm = np.mean(my_norm, axis=0)    # n*3
        mean_cent = np.mean(cent_pred, axis=0)  # 1*3
        out_cent = mean_cent

        #############RANSAC
        # foot_pred = pred_foot_ref + points.reshape(1000,1,3)
        # max_iterations = 100
        # goal_inliers = 1000 * 0.7
        # ransac_syms = np.zeros((3,3))
        # for i in range(3):
        #     xyz = foot_pred[:,i,:]
        #     m, b = run_ransac(xyz, estimate, lambda x, y: is_inlier(x, y, 0.01), 10, goal_inliers, max_iterations)
        #     a, b, c, d = m
        #     ransac_syms[i, :] = np.array([a,b,c])/np.linalg.norm(np.array([a,b,c]))

        # ########DBSCAN
        out_sym = np.zeros(mean_norm.shape)
        sym_conf = np.zeros(mean_norm.shape[0])
        norm_conf_list = np.zeros(mean_norm.shape[0])
        for i in range(my_norm.shape[1]):
            this_norm = my_norm[:, i, :].reshape(1000, 3)
            dim_conf = 0
            for t in range(3):
                this_dim = this_norm[:, t].reshape(1000, 1)
                # target_dim = target_sym[i,j]
                mean_dim = np.mean(this_dim, axis=0)
                db = skc.DBSCAN(eps=0.2, min_samples=500).fit(this_dim)   # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
                labels = db.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
                clster_center = np.mean(this_dim[labels[:] == 0], axis=0)
                out_sym[i,t] = clster_center
            #     dim_conf += len(labels[labels[:] == 0]) / len(labels)
            # norm_conf = dim_conf/3
            if np.isnan(out_sym[i]).any():
                norm_conf = 0
            else:
                norm_conf = 1
            norm_conf_list[i] = norm_conf
            sym_conf[i] = my_num[i]*norm_conf

        ######refinement
        # refine_sym = symmetry_icp_refinement(points, out_sym_, mean_cent, 0.005)
        # out_cent = refine_sym[:3]
        # out_sym = refine_sym[3:].reshape(-1, 3)

        #######RANSAC
        # out_cent = mean_cent
        # out_sym = ransac_syms
        # sef_conf = my_num

        my_ref = reflect(points, out_cent, out_sym)
        target_ref = reflect(points, target_cent, target_sym)

        ########self-loss
        # self_conf_list = []
        # for i in range(3):
        #     outlier = self_loss(out_sym[i], out_cent, points, depth, cam_ins, 100)
        #     self_conf = 1 - outlier / 1000
        #     self_conf_list.append(self_conf)
        #     sym_conf[i] = my_num[i]*self_conf#*norm_conf_list[i]

        target_sym = target_sym.reshape(-1, 3)
        # counterpart = np.zeros(pred_ref.shape)
        target_vector = target_ref - points.reshape(1000,1,3).repeat(target_ref.shape[1], axis=1)
        # for i in range(3):
        counterpart = my_ref
        rst_list = []
        for m in range(out_sym.shape[0]):
            if sym_conf[m] <0.5:
                continue
            for n in range(target_sym.reshape(-1, 3).shape[0]):
                target_len = np.linalg.norm(target_vector[:, n, :], axis=1)
                max_len = np.max(target_len)
                dense_dis = np.linalg.norm((my_ref[:, m, :] - target_ref[:, n, :]), axis=1)
                ctp_dis = np.linalg.norm((counterpart[:, m, :] - target_ref[:, n, :]), axis=1)
                dis_mean = np.mean(ctp_dis, axis=0)
                metric_mean = dis_mean / max_len
                rst_list.append(metric_mean)
            ctp_dis_list.append(np.max(rst_list))
            total_num += 1

        print('predcting frame:', j, ' object:', idx, 'target_num', target_num)

    thresh_list = []
    indis_list = []
    ctp_dis_list = np.array(ctp_dis_list)

    np.savetxt('./bigdata/ctp-ablation/ctp-foot_dis_list-frame.txt', ctp_dis_list)
    for t in range(1, 50001):
        dis_thresh = t / 10000
        indis = len(np.where(ctp_dis_list <= dis_thresh)[0])
        indis_list.append(indis / total_num)
        thresh_list.append(dis_thresh)
    return indis_list, thresh_list


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


def ref_pt(pt, cent, sym_vect):
    pt = pt.reshape(1000, 1, 3)
    cent = cent.reshape(1000, 1, 3)
    sym_vect = sym_vect.reshape(1000, -1, 3)
    pt_pred = np.zeros(sym_vect.shape)
    for i in range(sym_vect.shape[1]):
        center = cent.reshape(1000, 3, 1)
        norm = sym_vect[:, i, :].reshape(1000, 1, 3)
        d = -np.matmul(norm, center)
        pt_ = pt-2*(np.matmul(norm, pt.reshape(1000, 3, 1)) + d)*norm
        pt_pred[:, i, :] = pt_.reshape(1000, 3)
    return pt_pred


def reflect(Data,cent,sym):
    Data = Data.reshape(1000,3)
    cent = cent.reshape(3)
    sym = sym.reshape(-1, 3)
    reflect_points = np.zeros((1000,sym.shape[0],3))
    for j in range(sym.shape[0]):
        x1 = cent[0]  #对称面上中心点
        y1 = cent[1]
        z1 = cent[2]
        a = sym[j,0]                   #对称面法向量
        b = sym[j,1]
        c = sym[j,2]
        ref_point = np.zeros(Data.shape)
        for i in range(0, Data.shape[0]):
            d = a*x1+b*y1+c*z1
            t = (d-(a*Data[i][0]+b*Data[i][1]+c*Data[i][2]))/(a*a+b*b+c*c)
            sym_x = 2 * a * t + Data[i][0]                      ##计算得到的对称点
            sym_y = 2 * b * t + Data[i][1]
            sym_z = 2 * c * t + Data[i][2]
            ref_point[i,:] = np.array([sym_x, sym_y, sym_z])
        reflect_points[:,j,:] = ref_point
    return reflect_points

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
    depth_image = rgbd[0]  ##depth image
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


if __name__ == '__main__':
    st_time = time.time()
    savedir = proj_dir + 'tools/bigdata/ctp-ablation/'
    ang_list = [5, 10, 15, 20, 25]
    dis_list = [5, 10, 15, 20, 25]
    distance_list = [0.05, 0.1, 0.3, 0.4, 0.5]
    DIST_THRESHOLD = 15
    plt.style.use('fivethirtyeight')
    i = 0
    disin_list, thresh_list = prcurve(math.tan(DIST_THRESHOLD / 180 * math.pi))
    plt.plot(thresh_list, disin_list, linewidth=3)

    plot_data = np.concatenate((np.array(thresh_list).reshape(-1, 1), np.array(disin_list).reshape(-1, 1)), axis=1)
    np.savetxt(savedir + 'data/' + 'eval-foot-frame-0.5' + '.txt', plot_data)

    end_time = time.time()
    print("run_time=", end_time - st_time)
    plt.axis([0, 3, 0, 1])
    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel('% counterpart error', fontsize=20)
    plt.ylabel('correspondences', fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.title('counterpart eval\n', fontsize=20)
    plt.savefig(savedir + 'ctp-foot-frame-0.5.png', dpi=300, bbox_inches='tight')