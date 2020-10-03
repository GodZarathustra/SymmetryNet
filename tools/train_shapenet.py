# --------------------------------------------------------
# SymmetryNet: Learning to Predict Reflectional and Rotational
# Symmetries of 3D Shapes from Single-View RGB-D Images
# Licensed under The MIT License [see LICENSE for details]
# Written by Huang
# This code was build based on Densefusion https://github.com/j96w/DenseFusion
# --------------------------------------------------------

import argparse
import os
import random
import time
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
from torch.autograd import Variable
from datasets.shapenet.dataset_back import SymDataset as SymDataset_shapenet
from lib.network_swp import SymNet
from lib.loss import Loss
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'shapenet', help='shapenet or scan2cad')
parser.add_argument('--dataset_root', type=str, default = 'folder/to/your/dataset')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--workers', type=int, default = 32, help='number of data loading workers')
parser.add_argument('--lr', default=0.00005, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=1, help='learning rate')
parser.add_argument('--w_rate', default=0.9, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=40, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_symnet', type=str, default = '',  help='resume SymNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

proj_dir = 'folder/to/your/project'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # device_ids = range(torch.cuda.device_count())
    # print(device_ids)
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.dataset == 'shapenet':
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = proj_dir+'trained_models/shapenet/'  # folder to save trained models
        opt.log_dir = proj_dir+'experiments/logs/shapenet/'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'scan2cad':
        opt.num_points = 500
        opt.outf = proj_dir+'trained_models/cad'
        opt.log_dir = proj_dir+'experiments/logs/cad'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    estimator = SymNet(num_points = opt.num_points)
    estimator = estimator.cuda()
    # estimator = torch.nn.DataParallel(estimator)
    # estimator = torch.nn.parallel.DistributedDataParallel(estimator)
    #

    if opt.resume_symnet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_symnet)))

    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    opt.w *= opt.w_rate

    train_dataset = SymDataset_shapenet('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, False)
    test_dataset = SymDataset_shapenet('holdout_view', opt.num_points, False, opt.dataset_root, 0.0, False)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.num_points_mesh = train_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\n'.format(len(train_dataset), len(test_dataset)))

    criterion = Loss(opt.num_points_mesh)

    best_test = 0

    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        train_err_cent = 0.0
        train_loss_ref = 0.0
        train_err_ref = 0.0
        train_err_num = 0.0
        train_err_mode = 0.0

        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img,  idx, target_s, target_num, target_mode = data

                points, choose, img, idx, target_s, target_mode = Variable(points).cuda(), \
                                                                              Variable(choose).cuda(), \
                                                                              Variable(img).cuda(), \
                                                                              Variable(idx).cuda(), \
                                                                              Variable(target_s).cuda(), \
                                                                              Variable(target_mode).cuda()

                pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points,choose)

                loss, dis, error_cent, loss_ref, error_ref, error_num, error_mode = criterion(
                    pred_cent, pred_ref, pred_foot_ref, pred_rot,
                    pred_num, pred_mode, target_s,
                    points, opt.w, target_mode)

                loss.backward()

                train_dis_avg += dis.item()
                train_err_cent += error_cent.item()
                train_loss_ref += loss_ref.item()
                train_err_ref += error_ref.item()
                train_err_num += error_num.item()
                train_err_mode += error_mode.item()

                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info(
                        'Train time {0} Epoch {1} Batch {2} Frame {3} error_ref: {8} loss_ref:{9} loss_cent: {7} loss_num: {5} loss_mode:{6} Avg_loss:{4} cls_id: {10}'.format(
                            time.strftime("%Hh %Mm %Ss",
                                          time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size),
                            train_count, train_dis_avg / opt.batch_size,
                            train_err_num / opt.batch_size, train_err_mode / opt.batch_size,
                            train_err_cent / opt.batch_size, train_err_ref / opt.batch_size,
                            train_loss_ref / opt.batch_size, idx.data.cpu().numpy().reshape(-1)[0]))

                    optimizer.step()
                    optimizer.zero_grad()

                    train_dis_avg = 0
                    train_err_cent = 0
                    train_err_num = 0
                    train_err_mode = 0
                    train_err_ref = 0
                    train_loss_ref = 0

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/sym_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))

        test_dis = 0.0
        test_err_num = 0.0
        test_err_mode = 0.0
        test_err_ref = 0.0
        test_loss_ref = 0.0
        test_err_cent = 0.0
        test_count = 0
        ang_tps = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, idx, target_s, target_num,target_mode = data

            points, choose, img, idx, target_s, target_num, target_mode = Variable(points).cuda(), \
                                                                          Variable(choose).cuda(), \
                                                                          Variable(img).cuda(), \
                                                                          Variable(idx).cuda(), \
                                                                          Variable(target_s).cuda(), \
                                                                          Variable(target_num).cuda(), \
                                                                          Variable(target_mode).cuda()

            pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points,choose)
            loss, dis, error_cent, loss_ref, error_ref, error_num, error_mode = criterion(
                pred_cent, pred_ref, pred_foot_ref, pred_rot,
                pred_num, pred_mode, target_s,
                points, opt.w, target_mode)

            test_dis += dis.item()
            test_err_cent += error_cent.item()
            test_err_num += error_num.item()
            test_err_mode += error_mode.item()
            test_loss_ref += loss_ref.item()
            test_err_ref += error_ref.item()

            logger.info(
                'Test time {0} Test Frame:{1} error_ref:{6} loss_ref:{7} loss_cent:{5} loss_num:{3} loss_mode:{4} total_loss:{2} cls_id{8}'.format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - st_time)), test_count, dis, error_num, error_mode,
                    error_cent, error_ref, loss_ref, idx.data.cpu().numpy().reshape(-1)[0]))

            test_count += 1

            if error_ref <= 20:
                ang_tps += 1

        test_dis = test_dis / test_count
        test_err_num = test_err_num / test_count
        test_err_mode = test_err_mode / test_count
        test_err_ref = test_err_ref / test_count
        test_loss_ref = test_loss_ref / test_count
        test_err_cent = test_err_cent / test_count

        pect_ang_tps = ang_tps / test_count
        # angle_loss = math.cos(test_err_ref)

        logger.info('Test time {0} Epoch {1} TEST FINISH loss_ref:{7} angle_tps{8} Avg dis:{2} error_num:{3} error_mode:{4} error_cent:{5} error_ref:{6} '.format(time.strftime("%Hh %Mm %Ss",
                    time.gmtime(time.time() - st_time)), epoch, test_dis, test_err_num, test_err_mode, test_err_cent,test_err_ref, test_loss_ref, pect_ang_tps))
        if pect_ang_tps >= best_test:
            best_test = pect_ang_tps
            torch.save(estimator.state_dict(), '{0}/sym_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))

            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if test_err_ref < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

if __name__ == '__main__':
    main()
