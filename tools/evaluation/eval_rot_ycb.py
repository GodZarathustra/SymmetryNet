import argparse
import os
import sys
# sys.path.append('/your/project/path') # to run on a server
import math
import random
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset_ycb_eval import SymDataset as SymDataset_ycb
from lib.network import SymNet
import matplotlib.pyplot as plt
from lib.verification import rot_vrf
from lib.tools import rotate

from skimage.measure import LineModelND, ransac
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb')
parser.add_argument('--dataset_root', type=str, default = '/your/ycb/data/path')
parser.add_argument('--project_root', type=str, default = '/your/project/path')
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
parser.add_argument('--resume_symnet', type=str, default='ycb_model.pth', help='resume SymNet model')
opt = parser.parse_args()

proj_dir = opt.project_root
sym_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def prcurve(DIST_THRESHOLD):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir + 'trained_models/ycb/'  # folder to save trained models
        opt.log_dir = proj_dir + 'visualization'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = proj_dir + 'trained_models/linemod'
        opt.log_dir = proj_dir + 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = SymNet(num_points=opt.num_points)
    estimator.cuda()

    if opt.resume_symnet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_symnet)))
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
        dataset = SymDataset_ycb('train', opt.num_points, False, opt.dataset_root, proj_dir,opt.noise_trans, opt.refine_start)
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

    # ref_list = [1, 3, 4, 5, 6, 8, 10, 11, 13, 15, 18, 20]
    # rot_list = [0, 2, 7, 19]
    for j, data in enumerate(testdataloader, 0):
        points, choose, img, idx, target_s, target_num, target_mode, depth, cam_ins, pt_num = data
        if target_mode != 1:
            continue
        # if target_mode != 0:
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

        if rot_foot_pred.shape[0] <= 20:
            continue
        model, inliers = ransac(rot_foot_pred, LineModelND, min_samples=800, residual_threshold=0.015,
                                max_trials=1000)  # 0.03
        maxid = np.max(my_mode[1:])
        ransac_conf = inliers.sum() / 1000

        orig, direc = model.params
        my_sym_vec = direc

        out_sym_pt = orig + my_sym_vec

        ######## verification
        outlier = rot_vrf(orig, out_sym_pt, points, depth.numpy()[0], cam_ins.numpy(), 100, depth.shape[1],
                          depth.shape[2], thresh=0.2)
        vrf_conf = 1 - outlier / 1000
        sym_conf = vrf_conf * ransac_conf

        target_point = np.zeros((1000, 3))
        pred_point = np.zeros((1000, 3))
        for i in range(1000):
            target_point[i, :] = rotate(points[i, :], target_cent, target_sym.reshape(3), 180)
            pred_point[i, :] = rotate(points[i, :], orig, out_sym_pt, 180)
        target_len = np.linalg.norm(target_point - points, axis=1)
        max_len = np.max(target_len)
        dense_dis = np.linalg.norm((pred_point - target_point), axis=1)
        dis_mean = np.mean(dense_dis, axis=0)
        metric_mean = dis_mean / max_len

        if metric_mean <= DIST_THRESHOLD:
            tp += 1
            tp_conf.append(sym_conf)
        else:

            fn += 1
            fn_conf.append(sym_conf)
        total_num += 1
        print('predicting frame:', j, ' object:', idx, 'target_num', target_num)


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

        if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
            this_prec = true_positives / (false_positives + true_positives)
            this_recall = true_positives / (true_positives + false_negatives)
            prec.append(this_prec)
            recall.append(this_recall)

    return recall, prec

if __name__ == '__main__':
    st_time = time.time()
    savedir = proj_dir + 'tools/new-plot/'
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