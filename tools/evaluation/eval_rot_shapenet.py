import argparse
import os
import sys
import math
import random
import time
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from datasets.shapenet.dataset_eval import SymDataset as SymDataset_shapenet
from lib.network import SymNet
from lib.verification import rot_vrf
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import LineModelND, ransac
from lib.tools import rotate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'shapenet', help='shapenet or scannet')
parser.add_argument('--dataset_root', type=str, default = 'path/to/your/dataset/')
parser.add_argument('--project_root', type=str, default = 'path/to/this/project/')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=32, help='number of data loading workers')
parser.add_argument('--resume_posenet', type=str, default='', help='resume SymNet model')
parser.add_argument('--occ_level', type=str, default='', help='choose level of occlusion: light or heavy or mid')
opt = parser.parse_args()
torch.set_num_threads(32)
proj_dir = opt.project_root
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def prcurve(THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'shapenet':
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/shapenet/'  # folder to save trained models

    # estimator = torch.nn.DataParallel(SymNet(num_points=opt.num_points))
    # torch.nn.DataParallel(module=estimator, device_ids=device_ids)
    estimator = SymNet(num_points=opt.num_points)
    estimator.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    opt.refine_start = False
    opt.decay_start = False

    if opt.dataset == 'shapenet':
        dataset = SymDataset_shapenet('train', opt.num_points, False, opt.dataset_root, proj_dir,opt.noise_trans,
                                      opt.refine_start)
        test_dataset = SymDataset_shapenet('holdout_view', opt.num_points, False, opt.dataset_root, proj_dir,0.0,
                                           opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.num_points_mesh = dataset.get_num_points_mesh()

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
        points, choose, img, idx, target_s, target_num, target_mode, occ, cloud_, depth, cam_ins, num_pt, _ = data
        pt_num = num_pt.data.cpu().numpy()[0]
        occ = occ.data.cpu().numpy()[0]

        if opt.occ_level == 'light':
            occlusion = (occ > 0.6)
        elif opt.occ_level == 'mid':
            occlusion = ((occ < 0.6) or (occ > 0.7))
        elif opt.occ_level == 'heavy':
            occlusion = ((occ < 0.7) or (occ > 0.8))
        else:
            occlusion = True

        if (pt_num < 0.01) or (target_mode == 0) or occlusion:
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
        points = points.cpu().numpy()
        if (points == 0).all():
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
        model, inliers = ransac(rot_foot_pred, LineModelND, min_samples=800, residual_threshold=0.015, max_trials=1000) #0.03
        maxid = np.max(my_mode[1:])
        ransac_conf = inliers.sum()/1000

        orig, direc = model.params
        my_sym_vec = direc

        out_sym_pt = orig + my_sym_vec

        ######## verification
        outlier = rot_vrf(orig, out_sym_pt,points,depth.numpy()[0], cam_ins.numpy(), 100,depth.shape[1],depth.shape[2],thresh=0.2)
        vrf_conf = 1 - outlier / 1000
        sym_conf = vrf_conf * ransac_conf

        target_point = np.zeros((1000, 3))
        pred_point = np.zeros((1000, 3))
        for i in range(1000):
            target_point[i, :] = rotate(points[i, :], target_cent, target_sym.reshape(3), 180)
            pred_point[i,:] = rotate(points[i, :], orig, out_sym_pt, 180)
        target_len = np.linalg.norm(target_point - points, axis=1)
        max_len = np.max(target_len)
        dense_dis = np.linalg.norm((pred_point - target_point), axis=1)
        dis_mean = np.mean(dense_dis, axis=0)
        metric_mean = dis_mean/max_len

        if metric_mean <= THRESHOLD:
            tp += 1
            tp_conf.append(sym_conf)
        else:
            fn += 1
            fn_conf.append(sym_conf)
        total_num += 1
        print('predicting frame:', j, ' object:', idx, 'number of symmetry', target_num )

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
        false_positives = len(np.where(fn_conf >= conf_thresh)[0])
        tp_list.append(true_positives)
        fp_list.append(false_positives)
        fn_list.append(false_negatives)
        conf_thresh_list.append(conf_thresh)
        if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
            this_prec = true_positives / (false_positives + true_positives)
            this_recall = true_positives / (true_positives + false_negatives)
            prec.append(this_prec)
            recall.append(this_recall)

    return recall, prec

if __name__ == '__main__':
    st_time = time.time()
    THRESHOLD = 15
    savedir = proj_dir + 'tools/prcurve/shapenet/'

    recall, prec = prcurve(math.tan(THRESHOLD / 180 * math.pi))
    plot_data = np.concatenate((np.array(recall).reshape(-1, 1), np.array(prec).reshape(-1, 1)), axis=1)
    np.savetxt(savedir + 'data/' + 'rot-new_view' + 'dis=' + str(THRESHOLD) + '.txt', plot_data)

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('PR-curve', fontsize=15)
    plt.axes().set_aspect(1, 'box')
    plt.plot(recall, prec, linewidth=2, color='tab:red', zorder=10, label='ours')
    plt.legend(loc='upper right', fontsize=13)
    plt.grid(True)
    plt.savefig(savedir + "rot-new-view.png")
    plt.show()