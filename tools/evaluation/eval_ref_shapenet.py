#coding=utf-8
import argparse
import os
import sys
# sys.path.append('/your/project/path') # to run on a server
import math
import random
import time
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from datasets.shapenet.dataset_shapenet_eval import SymDataset as SymDataset_shapenet
from lib.network import SymNet
import matplotlib.pyplot as plt
import numpy as np
from lib.verification import ref_vrf
from lib.tools import reflect
import sklearn.cluster as skc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'shapenet', help='shapenet or scannet')
parser.add_argument('--dataset_root', type=str, default = '/your/shapenet/data/path')
parser.add_argument('--project_root', type=str, default = '/your/project/path')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--resume_posenet', type=str, default='shapenet_model.pth', help='resume SymNet model')
parser.add_argument('--occ_level', type=str, default='', help='choose level of occlusion: light or heavy or mid')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
opt = parser.parse_args()
torch.set_num_threads(32)
proj_dir = opt.project_root
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def prcurve(DIST_THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'shapenet':
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/shapenet/'  # folder to save trained models
        opt.repeat_epoch = 1  # number of repeat times for one epoch training

    # estimator = torch.nn.DataParallel(SymNet(num_points=opt.num_points))
    # torch.nn.DataParallel(module=estimator, device_ids=device_ids)
    estimator = SymNet(num_points=opt.num_points)
    estimator.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    opt.refine_start = False
    opt.decay_start = False

    if opt.dataset == 'shapenet':
        dataset = SymDataset_shapenet('train', opt.num_points, False, opt.dataset_root, proj_dir,opt.noise_trans, opt.refine_start)
        test_dataset = SymDataset_shapenet('holdout_class', opt.num_points, False, opt.dataset_root,proj_dir, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\n'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh))

    estimator.eval()
    fp = 0
    tp = 0
    fp_conf = []
    tp_conf = []
    prec = []
    recall = []
    total_fream = 0
    total_num = 0

    for j, data in enumerate(testdataloader, 0):
        points, choose, img, idx, target_s, target_num, target_mode, occ , cloud_,\
        depth, cam_ins,num_pt,_= data

        points, choose, img, idx, target_s, target_num, target_mode = Variable(points).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(idx).cuda(), \
                                                                Variable(target_s).cuda(), \
                                                                Variable(target_num).cuda(),\
                                                                Variable(target_mode).cuda()

        pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points, choose)

        target_mode = target_mode.data.cpu().numpy().reshape(-1)
        pt_num = num_pt.data.cpu().numpy()[0]
        occ = occ.data.cpu().numpy()[0]
        if opt.occ_level == 'light':
            occlusion = (occ > 0.6)
        elif opt.occ_level == 'mid':
            occlusion = ((occ < 0.6) or (occ > 0.7))
        elif opt.occ_level == 'heavy':
            occlusion = ((occ < 0.7) or (occ > 0.8))
        else:
            occlusion = False

        if (pt_num < 0.01) or (target_mode == 1) or occlusion:
            continue

        total_fream += 1
        tmp_s = target_s.data.cpu().numpy()
        tmp_s = tmp_s.reshape(-1, 3)
        target_cent = tmp_s[0, :]
        target_sym = tmp_s[1:, :] # target symmetry point
        len_target_sym = np.linalg.norm((target_sym-target_cent),axis=1)
        num_pt = num_pt.data.cpu().numpy()[0]
        cam_ins = cam_ins.data.cpu().numpy().reshape(4)
        depth = depth.data.cpu().numpy()

        points = points.view(1000, 3)
        points = points.detach().cpu().data.numpy()

        pred_cent = pred_cent.detach().cpu().data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        cent_pred = (points + pred_cent)

        pred_num = pred_num.view(1000, 3).detach()
        my_num = torch.mean(pred_num, dim=0)
        my_num = my_num.data.cpu().numpy()

        pred_ref = pred_ref.detach().cpu().data.numpy() # (1,1000,9)
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

        # ########DBSCAN
        st_time = time.time()
        print('st_time = ',st_time)
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
                db = skc.DBSCAN(eps=0.2, min_samples=500).fit(this_dim)
                labels = db.labels_
                clster_center = np.mean(this_dim[labels[:] == 0], axis=0)
                out_sym[i,t] = clster_center
                dim_conf += len(labels[labels[:] == 0]) / len(labels)

            if np.isnan(out_sym[i]).any():
                norm_conf = 0
            else:
                # norm_conf = dim_conf/3
                norm_conf = 1
            norm_conf_list[i] = norm_conf
            # sym_conf[i] = my_num[i]*norm_conf
        edtime = time.time()-st_time

        ######## verification
        self_conf_list = []
        for i in range(3):
            outlier = ref_vrf(out_sym[i], out_cent, points, depth[0], cam_ins, 100,depth.shape[1],depth.shape[2],thresh=0.2)
            self_conf = 1 - outlier / 1000
            self_conf_list.append(self_conf)
            sym_conf[i] = my_num[i] *self_conf*norm_conf_list[i]

        my_ref = reflect(points, out_cent, out_sym)
        target_ref = reflect(points, target_cent, target_sym)

        target_sym = target_sym.reshape(-1, 3)
        target_vector = target_ref - points.reshape(1000,1,3).repeat(target_ref.shape[1], axis=1)

        for a in range(target_sym.shape[0]):
            for b in range(out_sym.shape[0]):
                target_len = np.linalg.norm(target_vector[:, a, :], axis=1)
                max_len = np.max(target_len)
                dense_dis = np.linalg.norm((my_ref[:, b, :] - target_ref[:, a, :]), axis=1)
                metric = dense_dis
                metric_mean = np.mean(metric)/max_len

                if metric_mean <= DIST_THRESHOLD:
                    tp += 1
                    tp_conf.append(sym_conf[b])
                    break

        for m in range(out_sym.shape[0]):
            shoot = 0
            for n in range(target_sym.reshape(-1, 3).shape[0]):
                target_len = np.linalg.norm(target_vector[:, n, :], axis=1)
                max_len = np.max(target_len)
                dense_dis = np.linalg.norm((my_ref[:, m, :] - target_ref[:, n, :]), axis=1)

                dis_mean = np.mean(dense_dis, axis=0)
                metric_mean = dis_mean/max_len
                if metric_mean <= DIST_THRESHOLD:
                    shoot += 1
            if shoot == 0:
                fp += 1
                fp_conf.append(sym_conf[m])

        print('predcting frame:', j, ' object:', idx, 'target_num', target_num)
        total_num += target_sym.shape[0]


    fn = total_num - tp
    tp_conf = np.array(tp_conf)
    fp_conf = np.array(fp_conf)

    tp_list = []
    fp_list = []
    fn_list = []
    conf_thresh_list = []

    for t in range(1, 10001):
        conf_thresh = t / 10000
        true_positives = len(np.where(tp_conf >= conf_thresh)[0])
        false_negatives = fn + len(np.where(tp_conf < conf_thresh)[0])
        false_positives = len(np.where(fp_conf >= conf_thresh)[0])
        tp_list.append(true_positives)
        fp_list.append(false_positives)
        fn_list.append(false_negatives)
        conf_thresh_list.append(conf_thresh)
        if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
            prec.append(true_positives / (false_positives + true_positives))
            recall.append(true_positives / (true_positives + false_negatives))

    return recall, prec

if __name__ == '__main__':
    st_time = time.time()
    savedir = proj_dir + 'tools/prcurve/shapenet/'
    DIST_THRESHOLD = 15

    recall, prec = prcurve(math.tan(DIST_THRESHOLD / 180 * math.pi))

    plot_data = np.concatenate((np.array(recall).reshape(-1, 1), np.array(prec).reshape(-1, 1)), axis=1)
    np.savetxt(savedir+'data/'+'test'+'.txt', plot_data)

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('PR-curve', fontsize=15)
    plt.axes().set_aspect(1, 'box')
    plt.plot(recall, prec, linewidth=2, color='tab:red', zorder=10, label='ours')
    plt.legend(loc='upper right', fontsize=13)
    plt.grid(True)
    plt.savefig(savedir + "test.png")
    plt.show()
