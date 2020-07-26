#coding=utf-8
import argparse
import os
import sys
sys.path.append('/home/dell/yifeis/pose_estimation/densefusion_syn_test/')
import random
import time
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datasets.baseline.dataset_eval import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet, PoseRefineNet
from lib.loss2 import Loss
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d
import numpy.ma as ma
# plt.switch_backend('agg')

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
parser.add_argument('--resume_posenet', type=str, default='pose_model_49_1.2153524918986691.pth', help='resume PoseNet model')  # pose_model_current.pth  pose_model_58_1.059006152053674.pth  pose_model_39_1.2116587293644747.pth pose_model_49_1.2153524918986691.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

proj_dir = '/home/dell/yifeis/pose_estimation/densefusion_syn_test/'
sym_list = training_cat = [0, 3, 7, 10, 11, 12, 14, 16, 17, 19, 21, 22]
visual_dir = proj_dir+'datasets/ycb/dataset_config/visual_data_list.txt'
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560]
img_width = 540
img_length = 960

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list ) -1):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list ) -1):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def prcurve(n,mode):
    if mode == 2:
        holdout = 'holdout_instance'
    if mode == 1:
        holdout = 'holdout_view'
    if mode == 3:
        holdout = 'holdout_class'
    result = 'shapenet_result_' + str(n)
    datalist = 'result'+str(n).zfill(2)+holdout
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/ycb'  # folder to save trained models
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
        test_dataset = PoseDataset_ycb(datalist, opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh)

    point_zero = np.zeros((1000,3))

    estimator.eval()

    total_fream = 0
    total_num = 0

    tmp_path = '/home/dell/yifeis/pose_estimation/densefusion_syn_test/tools/'
    data_path = tmp_path + 'baseline4_result_' +str(n) + '_' + holdout + '.txt'
    input_file = open(data_path)
    data_list = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        data_list.append(input_line)
    input_file.close()
    name_list = np.loadtxt(proj_dir + 'name_list2.txt', dtype=str, delimiter='\n')
    class_id = {
        '000': '02691156',
        '001': '02747177',
        '002': '02773838',
        '003': '02808440',
        '004': '02818832',
        '005': '02828884',
        '006': '02876657',
        '007': '02924116',
        '008': '02942699',
        '009': '02946921',
        '010': '02958343',
        '011': '03001627',
        '012': '03211117',
        '013': '03261776',
        '014': '03467517',
        '015': '03513137',
        '016': '03593526',
        '017': '03636649',
        '018': '03642806',
        '019': '03991062',
        '020': '04225987',
        '021': '04256520',
        '022': '04379243'}
    xmap = np.array([[j for i in range(960)] for j in range(540)])  # 480*640, xmap[i,:]==i
    ymap = np.array([[i for i in range(960)] for j in range(540)])  # 480*640, ymap[j,:]==j

    ctp_dis_list = []
    for index in range(len(data_list)):
        print('index', index)
        if (os.path.exists('{0}/{1}-rt.txt'.format(opt.dataset_root, data_list[index])) and \
                os.path.exists('{0}/{1}-depth-crop-occlusion.png'.format(opt.dataset_root, data_list[index])) and \
                os.path.exists('{0}/{1}-k-crop.txt'.format(opt.dataset_root, data_list[index])) and \
                os.path.exists('{0}/{1}-color-crop-occlusion.png'.format(opt.dataset_root, data_list[index])) and \
                os.path.exists('{0}/{1}-occlusion.txt'.format(opt.dataset_root, data_list[index])))==False:
            continue
        occ = np.loadtxt('{0}/{1}-occlusion.txt'.format(opt.dataset_root, data_list[index]))
        rt = np.loadtxt('{0}/{1}-rt.txt'.format(opt.dataset_root, data_list[index]))
        check_rt = np.zeros((4, 4))
        check_depth = 255 * np.ones((540, 960, 3))
        depth_ = cv2.imread('{0}/{1}-depth-crop-occlusion.png'.format(opt.dataset_root, data_list[index]))
        # depth_ = cv2.resize(depth_, (640,480))
        cam_ = np.loadtxt('{0}/{1}-k-crop.txt'.format(opt.dataset_root, data_list[index]))
        if cam_.reshape(-1).shape[0] != 9:
            print('{0}/{1}-k-crop.txt'.format(opt.dataset_root, data_list[index]))
            continue
        input_file = data_list[index]
        class_key = input_file[20:23]
        input_id = int(input_file[20:23])
        ins_num = int(input_file[24:28])
        cls_idx = input_id
        class_name = class_id[class_key]
        instance_ls = name_list[cls_idx][1:-1].split(",")
        ins_name = instance_ls[ins_num][2:-1]
        sym_dir = '/home/dell/dy/shapenetcore/'
        sym_file = sym_dir + class_name + '/' + ins_name + '/' + 'model_sym.txt'

        if os.path.exists(sym_file) == False:
            continue
        model_s = np.loadtxt(sym_file)
        syms = model_s[1:, :]
        check_ = np.zeros((4, 3))
        check_sym = (syms != check_)
        nozero = np.nonzero(check_sym)
        row_id = nozero[0]

        if (rt == check_rt).all() or (depth_ == check_depth).all():
            continue
        elif (row_id.shape[0] == 0) and (model_s.shape[0] != 5):
            continue

        input_file = data_list[index]
        class_key = input_file[20:23]
        input_id = int(input_file[20:23])
        ins_num = int(input_file[24:28])
        img_ = cv2.imread('{0}/{1}-color-crop-occlusion.png'.format(opt.dataset_root, data_list[index]))  # 540*960
        img = img_

        depth = depth_[:, :, 0]
        # depth = cv2.resize(depth_, (640, 480))[:,:,0]
        cam = np.loadtxt('{0}/{1}-k-crop.txt'.format(opt.dataset_root, data_list[index]))
        cam_cx = cam[0, 2]
        cam_cy = cam[1, 2]
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]

        cam_intri = [cam_cx, cam_cy, cam_fx, cam_fy]
        cam_intri = np.array(cam_intri)

        idx = input_id
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 255))
        mask_label = mask_depth
        mask = mask_depth
        mask_real = len(mask.nonzero()[0])

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        img = img[:, rmin:rmax, cmin:cmax]

        img_masked = img

        target_mode = 0
        center = model_s[0, :]

        if row_id.shape[0] == 1:
            if row_id[-1] == 3:
                target_mode = 1
            else:
                target_mode = 0
            multi_s = syms[row_id]
        elif row_id.shape[0] == 2:
            if row_id[-1] == 3:
                target_mode = 2
                multi_s = syms[row_id[0]]
            else:
                target_mode = 0
                multi_s = syms[row_id]
        elif row_id.shape[0] == 3:
            target_mode = 0
            multi_s = syms[row_id]
        else:
            target_mode = 0
            multi_s = syms[row_id]

        multi_s_point = multi_s + center
        multi_s = np.vstack([center, multi_s_point])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # 提取物体的mask维度
        ori = choose
        if len(choose) >1000:  # mask的点数量大于1000,则取前1000
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:1000] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, 1000 - len(choose)), 'wrap')  # 补维数,表示(0.1）两个维度分别补几维

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype \
            (np.float32)  # (1000,1)get masked depth
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        choose = np.array([choose])  # (1,1000)

        #####1000 points
        cam_scale = 100  # cam_scale = 10000
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # (1000,3)


        target_r = rt[:-1, :-1]
        target_t = rt[:-1, 3]
        target_s = np.add(np.dot(multi_s, target_r.T), target_t)

        # s_len = np.linalg.norm(target_s[1]-target_s[0])
        target_num = target_s.shape[0] - 1


        ###########################################################################
        base_dir = data_list[index]
        points = cloud
        img = img_masked
        idx = int(idx)

        target_mode = target_mode
        # occ = occ
        #  ((occ < 0.7)or(occ>0.8)) : 70-80
        #  ((occ <0.6)or(occ>0.7))  : 60-70
        #  (occ>0.6)                : <60
        if (target_mode == 1):
            continue
        if target_s.shape[0]<=1:
            continue
        total_fream += 1
        tmp_s = target_s
        tmp_s = tmp_s.reshape(-1, 3)
        target_cent = tmp_s[0, :]
        target_sym = tmp_s[1:, :] # target symmetry point
        len_target_sym = np.linalg.norm((target_sym-target_cent),axis=1)

        target_sym = target_sym - target_cent
        # target_sym = target_sym.reshape(-1,3)

        #####baseline
        sym_dir = '/home/dell/yifeis/pose_estimation/shape_completion/symmetrynet_data/shapenet_split_sig20/shapenet_figure7_result_15/' + '/' + holdout + '/symmetry_0.5/'
        obj_cls = base_dir[20:23]
        obj_ins = base_dir[24:28]
        obj_frm = base_dir[29:]
        sym_file = sym_dir + obj_cls+'_'+obj_ins+'_'+obj_frm+'.txt'
        conf_file = sym_dir + obj_cls+'_'+obj_ins+'_'+obj_frm+'_conf'+'.txt'
        if os.path.exists(sym_file)==False :
            print(sym_file)
            continue
        base_sym_ = np.loadtxt(sym_file)
        sym_conf = np.loadtxt(conf_file)
        # max_conf = 0.000168601080076769
        # sym_conf = 1
        if np.sum(base_sym_) == 0:
            target_sym = target_sym.reshape(-1, 3)
            out_sym = np.zeros((target_sym.shape))
            target_ref = reflect(points, target_cent, target_sym)
            my_ref = np.zeros((target_ref.shape))
            # sym_conf = np.zeros(target_sym.shape[0]).tolist()
        else:
            nozero_list = []
            for i in range(base_sym_.shape[0]):
                if np.sum(base_sym_[i])!=0:
                    nozero_list.append(i)
            base_sym = base_sym_[nozero_list]
            # my_ref = np.zeros((1000, base_sym.shape[0], 3))
            st_pt = base_sym[0,:]
            out_sym = base_sym[1:,:]

            my_ref = reflect(points, st_pt, out_sym)
            target_ref = reflect(points, target_cent, target_sym)

        target_sym = target_sym.reshape(-1, 3)
        # counterpart = np.zeros(pred_ref.shape)
        target_vector = target_ref - points.reshape(1000,1,3).repeat(target_ref.shape[1], axis=1)

        counterpart = my_ref

        if sym_conf < 0.3:
            continue
        for m in range(out_sym.shape[0]):
            rst_list = []
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

        print('predcting frame:', index, ' object:', idx, 'target_num', target_num)

    thresh_list = []
    indis_list = []
    ctp_dis_list = np.array(ctp_dis_list)

    np.savetxt('./bigdata/ctp-ablation/ctp_baseline_list-ins-0.3.txt', ctp_dis_list)
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

def symmetry_icp_refinement(visual_points, pred_sym, pred_cent,icp_threshold):
    refined_pred_sym = []
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    target_trans = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(visual_points)
    source_colors = np.zeros(visual_points.shape)
    for i in range(source_colors.shape[0]):
        source_colors[i, 0] = 0.2
        source_colors[i, 1] = 0.2
        source_colors[i, 2] = 0.2
    source.colors = open3d.utility.Vector3dVector(source_colors)

    cent = pred_cent.reshape(3)
    sym = pred_sym
    sym = sym.reshape(-1, 3)

    for j in range(sym.shape[0]):
        x1 = cent[0]
        y1 = cent[1]
        z1 = cent[2]
        a = sym[j, 0]
        b = sym[j, 1]
        c = sym[j, 2]
        x2 = cent[0] + sym[j, 0]
        y2 = cent[1] + sym[j, 1]
        z2 = cent[2] + sym[j, 2]
        ref_points = np.zeros(visual_points.shape)
        for i in range(0, visual_points.shape[0]):
            d = a * x1 + b * y1 + c * z1
            t = (d - (a * visual_points[i][0] + b * visual_points[i][1] + c * visual_points[i][2])) / (
                        a * a + b * b + c * c)
            sym_x = 2 * a * t + visual_points[i][0]
            sym_y = 2 * b * t + visual_points[i][1]
            sym_z = 2 * c * t + visual_points[i][2]
            ref_points[i, :] = np.array([sym_x, sym_y, sym_z])

        target.points = open3d.utility.Vector3dVector(ref_points)
        ref_colors = np.zeros(ref_points.shape)
        for i in range(ref_colors.shape[0]):
            ref_colors[i, 0] = 1
            ref_colors[i, 1] = 0.5
            ref_colors[i, 2] = 0.5
        target.colors = open3d.utility.Vector3dVector(ref_colors)

        source.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        target.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        reg_p2l = open3d.registration.registration_icp(
            target, source, icp_threshold, trans_init,
            open3d.registration.TransformationEstimationPointToPlane())
        print(reg_p2l)

        points = np.asarray(target.points)
        ones = np.ones((points.shape[0], 1))
        points = np.hstack((points, ones))
        points = np.dot(reg_p2l.transformation, points.T).T[:, :3]

        target_trans.points = open3d.utility.Vector3dVector(points)
        target_trans_colors = np.zeros(points.shape)
        for i in range(target_trans_colors.shape[0]):
            target_trans_colors[i, 0] = 0.5
            target_trans_colors[i, 1] = 1
            target_trans_colors[i, 2] = 0.5
        target_trans.colors = open3d.utility.Vector3dVector(target_trans_colors)

        p1 = np.array([x1, y1, z1, 1])
        p2 = np.array([x2, y2, z2, 1])
        p1 = np.dot(reg_p2l.transformation, p1)
        p2 = np.dot(reg_p2l.transformation, p2)
        cent_new = p1
        sym_new = p2 - p1
        # draw points here
        # open3d.visualization.draw_geometries([source,target,target_trans])

        if j == 0:
            refined_pred_sym.append(cent_new[0])
            refined_pred_sym.append(cent_new[1])
            refined_pred_sym.append(cent_new[2])
            refined_pred_sym.append(sym_new[0])
            refined_pred_sym.append(sym_new[1])
            refined_pred_sym.append(sym_new[2])
        else:
            refined_pred_sym.append(sym_new[0])
            refined_pred_sym.append(sym_new[1])
            refined_pred_sym.append(sym_new[2])

    refined_pred_sym = np.array(refined_pred_sym)
    print('pred_sym', pred_sym)
    print('refined_pred_sym', refined_pred_sym)
    return refined_pred_sym


if __name__ == '__main__':
    st_time = time.time()
    savedir = proj_dir + 'tools/bigdata/ctp-ablation/'
    ang_list = [5, 10, 15, 20, 25]
    dis_list = [5, 10, 15, 20, 25]
    distance_list = [0.05, 0.1, 0.3, 0.4, 0.5]
    DIST_THRESHOLD = 15
    plt.style.use('fivethirtyeight')
    i = 0
    modelist = [2]
    for mode in modelist:
        if mode == 2:
            holdout = 'holdout_instance'
        if mode == 1:
            holdout = 'holdout_view'
        if mode == 3:
            holdout = 'holdout_class'
        disin_list, thresh_list = prcurve(15, mode)
        plt.plot(thresh_list, disin_list, linewidth=3)

        plot_data = np.concatenate((np.array(thresh_list).reshape(-1, 1), np.array(disin_list).reshape(-1, 1)), axis=1)
        np.savetxt(savedir + 'data/' + 'eval_ctp-baseline4-new-'+holdout + '.txt', plot_data)

        end_time = time.time()
        print("run_time=", end_time - st_time)
        plt.axis([0, 1, 0, 1])
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('% counterpart error', fontsize=20)
        plt.ylabel('correspondences', fontsize=20)
        plt.tick_params(axis='both', labelsize=15)
        plt.title('counterpart eval\n', fontsize=20)
        plt.savefig(savedir + 'ctp-eval-baseline4-new'+holdout+'.png', dpi=300, bbox_inches='tight')
    # plt.savefig(savedir + 'new-frame_ref-mynum_normconf.png')
    # plt.show()