import argparse
import os
import numpy as np
import math
import random
import time
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset import SymDataset as SymDataset_ycb
from lib.network import SymNet
from lib.tools import reflect
import matplotlib.pyplot as plt
from ransac import *
import sklearn.cluster as skc  

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb')
parser.add_argument('--dataset_root', type=str, default = 'path/to/your/dataset/')
parser.add_argument('--project_root', type=str, default = 'path/to/this/project/')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--resume_symnet', type=str, default='', help='resume SymNet model')
opt = parser.parse_args()

proj_dir = opt.project_root
sym_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]

def prcurve(DIST_THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/ycb/swp'  # folder to save trained models
        opt.log_dir = proj_dir + 'visualization'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training

    else:
        print('Unknown dataset')
        return

    estimator = SymNet(num_points=opt.num_points)
    estimator.cuda()

    if opt.resume_symnet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_symnet)))
        opt.refine_start = False
        opt.decay_start = False

    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        opt.w *= opt.w_rate

    if opt.dataset == 'ycb':
        dataset = SymDataset_ycb('train', opt.num_points, False, opt.dataset_root,proj_dir, opt.noise_trans, opt.refine_start)
        test_dataset = SymDataset_ycb('test', opt.num_points, False, opt.dataset_root,proj_dir, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
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
    pred_c = 1
    ref_list = [1, 3, 4, 5, 6, 8, 10, 11, 13, 15, 18, 20]
    # rot_list = [0, 2, 7, 19]
    for j, data in enumerate(testdataloader, 0):
        points, choose, img, idx, target_s, target_num, target_mode,depth,cam_intri,pt_num = data

        if idx not in ref_list:
            continue
        # if target_mode != 1:
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
                                                                      Variable(target_num).cuda(), \
                                                                      Variable(target_mode).cuda()

        pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points, choose)

        points = points.view(1000, 3)
        points = points.detach().cpu().data.numpy()

        pred_cent = pred_cent.detach().cpu().data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        pred_cent = (points + pred_cent)

        pred_num = pred_num.view(1000, 3).detach()
        my_num = torch.mean(pred_num, dim=0)
        my_num = my_num.data.cpu().numpy().reshape(3)

        pred_ref = pred_ref.detach().cpu().data.numpy().reshape(1000, -1, 3)# (1,1000,9)
        ref_pred = pred_ref+points.reshape(1000,1,3)

        pred_foot_ref = pred_foot_ref.detach().cpu().data.numpy().reshape(1000, -1, 3)
        foot_pred = pred_foot_ref + points.reshape(1000,1,3)

        mode_pred = torch.mean(pred_mode.view(1000,3),dim=0).detach().cpu().data.numpy()

        my_cent = np.mean(pred_cent, axis=0)

        target_sym = target_sym - target_cent

        my_sym = pred_ref
        my_norm = np.zeros(my_sym.shape)
        for i in range(my_sym.shape[1]):
            for k in range(3):
                my_norm[:, i, k] = my_sym[:, i, k] / np.linalg.norm(my_sym[:, i, :], axis=1)

        mean_norm = np.mean(my_norm, axis=0)    # n*3
        mean_cent = np.mean(pred_cent, axis=0)  # 1*3
        out_cent = mean_cent

        ######DBSCAN
        out_sym = np.zeros(mean_norm.shape)
        sym_conf = np.zeros(mean_norm.shape[0])
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
            norm_conf = dim_conf/3
            mode_conf = max(mode_pred[:2])
            sym_conf[i] = my_num[i]* norm_conf

        ########  verification

        my_ref = reflect(points, out_cent, out_sym)
        target_ref = reflect(points, target_cent, target_sym)

        target_vector = target_ref-points.reshape(1000,1,3).repeat(target_ref.shape[1], axis=1)
        dense_dis_list = []
        counter_dis_list = []
        counterpart = pred_ref + points.reshape(1000,1,3)
        for a in range(target_sym.shape[0]):
            for b in range(out_sym.shape[0]):
                target_len = np.linalg.norm(target_vector[:, a, :], axis=1)
                max_len = np.max(target_len)
                dense_dis = np.linalg.norm((my_ref[:, b, :] - target_ref[:, a, :]), axis=1)
                counter_dis = np.linalg.norm((counterpart[:,b,:]-target_ref[:, a, :]), axis=1)
                counter_dis_mean = np.mean(counter_dis,axis=0)
                dis_mean = np.mean(dense_dis, axis=0)
                dense_dis_list.append(dis_mean)
                counter_dis_list.append(counter_dis_mean)
                metric_mean = dis_mean/max_len
                if metric_mean <= DIST_THRESHOLD:
                    tp += 1
                    tp_conf.append(sym_conf[b])
                    break

        for m in range(out_sym.shape[0]):
            shoot = 0
            for n in range(target_sym.shape[0]):
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

        total_num += target_sym.shape[0]

        print('predcting frame:', j, ' object:', idx, 'target_num', target_num)

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
    savedir = proj_dir + 'tools/plot/'
    DIST_THRESHOLD = 15

    recall, prec = prcurve(math.tan(DIST_THRESHOLD / 180 * math.pi))
    plot_data = np.concatenate((np.array(recall).reshape(-1, 1), np.array(prec).reshape(-1, 1)), axis=1)
    np.savetxt(savedir + 'data/' + 'test' + '.txt', plot_data)

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