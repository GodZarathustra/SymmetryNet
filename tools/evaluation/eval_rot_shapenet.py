################## readme
### network 3dim output for rot foot, co_loss use **2
##################  0108
import argparse
import os
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.shapenet.dataset_eval import SymDataset as SymDataset_shapenet
from lib.network import SymNet
from lib.loss_all import Loss
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import LineModelND, ransac

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'shapenet', help='shapenet or scan2cad')
parser.add_argument('--dataset_root', type=str, default='/home/dell/yifeis/pose_estimation/render/render_dy/render_wt_pt_proj/data/',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
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
parser.add_argument('--resume_posenet', type=str, default='pose_model_69_2.6220763158244758.pth', help='resume PoseNet model')  #pose_model_49_1.2153524918986691.pth  pose_model_50_1.3063710926430472.pth
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

proj_dir = '/home/dell/yifeis/pose_estimation/densefusion_syn_test/'
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def prcurve(DIST_THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'shapenet':
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/shapenet'  # folder to save trained models
        opt.repeat_epoch = 1  # number of repeat times for one epoch training

    estimator = torch.nn.DataParallel(SymNet(num_points=opt.num_points, num_obj=opt.num_objects))
    # torch.nn.DataParallel(module=estimator, device_ids=device_ids)
    estimator.cuda(device_ids[0])
    torch.nn.DataParallel(module=estimator, device_ids=device_ids)

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        estimator = estimator.module
        opt.refine_start = False
        opt.decay_start = False

    if opt.dataset == 'shapenet':
        dataset = SymDataset_shapenet('syn_train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start)
        test_dataset = SymDataset_shapenet('syn_class', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    estimator.eval()
    fp = 0
    fn = 0
    tp = 0
    fp_conf = []
    tp_conf = []
    fn_conf = []
    prec = []
    recall = []
    total_fream = 0

    total_num = 0

    for j, data in enumerate(testdataloader, 0):
        points, choose, img, idx, target_s, target_num, target_mode, occ , cloud_, depth, cam_ins, num_pt,_= data
        pt_num = num_pt.data.cpu().numpy()[0]
        occ = occ.data.cpu().numpy()[0]
        #  ((occ < 0.7)or(occ>0.8)) : 70-80
        #  ((occ <0.6)or(occ>0.7)) :  60-70
        #  (occ>0.6) :  <60
        if (pt_num < 0.01) or (target_mode == 0) :
            continue
        total_fream += 1
        tmp_s = target_s.data.cpu().numpy()
        tmp_s = tmp_s.reshape(-1, 3)
        target_cent = tmp_s[0, :]
        target_sym = tmp_s[1:, :]  # target symmetry point

        points, choose, img, idx, target_s, target_num, target_mode = Variable(points).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(idx).cuda(), \
                                                                Variable(target_s).cuda(), \
                                                                Variable(target_num).cuda(),\
                                                                Variable(target_mode).cuda()

        pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points,
                                                                                                   choose)

        my_mode = pred_mode.view(1000, 3).detach().cpu().data.numpy()
        my_mode = np.mean(my_mode, axis=0)

        points = points.view(1000, 3)
        points = points.detach().cpu().data.numpy()
        if (points==0).all():
            continue

        pred_num = pred_num.view(1000, 3).detach()
        my_num = torch.mean(pred_num, dim=0)
        my_num = my_num.data.cpu().numpy()

        pred_cent = pred_cent.detach().cpu().data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        cent_pred = points + pred_cent

        pred_rot_foot = pred_rot.view(1000, 3).detach().cpu().data.numpy()  # foot vector
        rot_foot_pred = pred_rot_foot + points  # foot point

        my_cent = np.mean(cent_pred, axis=0)

        target_sym_vec = target_sym - target_cent  # target sym_vetctor 1*3
        target_len = np.linalg.norm(target_sym_vec)

        if rot_foot_pred.shape[0]<=20:
            continue
        model, inliers = ransac(rot_foot_pred, LineModelND, min_samples=700, residual_threshold=0.03, max_trials=1000) #0.03
        maxid = np.max(my_mode[1:])
        ransac_conf = inliers.sum()/1000
        sym_conf = ransac_conf
        orig, direc = model.params
        my_sym_vec = direc

        out_sym_pt = orig + my_sym_vec

        target_point = np.zeros((1000, 3))
        pred_point = np.zeros((1000, 3))
        for i in range(1000):
            target_point[i, :] = rotate(points[i, :], target_cent, target_sym.reshape(3), 180)
            pred_point[i,:] = rotate(points[i, :], orig, out_sym_pt, 180)
        target_len = np.linalg.norm(target_point - points, axis=1)
        max_len = np.max(target_len)
        dense_dis = np.linalg.norm((pred_point - target_point), axis=1)
        dis_mean = np.mean(dense_dis, axis=0)
        metric_mean =dis_mean/max_len

        if metric_mean <= DIST_THRESHOLD:
            tp += 1
            tp_conf.append(sym_conf)
        else :
            fn += 1
            fn_conf.append(sym_conf)
        total_num += 1
        print('predcting frame:', j, ' object:', idx, 'target_num', target_num )

    # fn = total_num - tp

    tp_conf = np.array(tp_conf)
    fp_conf = np.array(fp_conf)
    fn_conf = np.array(fn_conf)
    tp_list = []
    fp_list = []
    fn_list = []
    conf_thresh_list = []

    for t in range(1, 10001):
        conf_thresh = t / 10000
        true_positives = len(np.where(tp_conf >= conf_thresh)[0])
        false_negatives = fn + len(np.where(tp_conf < conf_thresh)[0])
        # false_negatives = fn
        false_positives = len(np.where(fn_conf >= conf_thresh)[0])
        tp_list.append(true_positives)
        fp_list.append(false_positives)
        fn_list.append(false_negatives)
        conf_thresh_list.append(conf_thresh)
        # if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
        if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
            this_prec = true_positives / (false_positives + true_positives)
            this_recall = true_positives / (true_positives + false_negatives)
            prec.append(this_prec)
            recall.append(this_recall)

    print('total_target_sym_number:', total_num)
    print('conf_thresh=', conf_thresh_list)
    print('precision length:', len(prec))
    print('recall length:', len(recall))
    print('prec=', prec)
    print('recall=',recall)

    return recall, prec

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

if __name__ == '__main__':
    st_time = time.time()
    ANGLE_THRESHOLD = math.cos(math.pi / 180 * 20)

    DIST_THRESHOLD = 15
    savedir = proj_dir + 'tools/bigdata/shapenet-new-eval-rot/'
    ang_list = [5, 10, 15, 20]
    dis_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    plt.style.use('fivethirtyeight')

    color_list = ['red', 'blue', 'black', 'green', 'yellow', 'pink']
    i = 0

    recall, prec = prcurve(math.tan(DIST_THRESHOLD / 180 * math.pi))
    plt.plot(recall, prec, label='distance ratio=tan' + str(DIST_THRESHOLD), linewidth=3)
    plot_data = np.concatenate((np.array(recall).reshape(-1, 1), np.array(prec).reshape(-1, 1)), axis=1)

    end_time = time.time()

    print("run_time=", end_time - st_time)
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel('recall', fontsize=20)
    plt.ylabel('precision', fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.title('rot class test on ShapeNet\n', fontsize=20)
    plt.savefig(savedir + 'rot-class-700-0.03.png', dpi=300, bbox_inches='tight')

