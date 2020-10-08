import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import random
import numpy.ma as ma
import scipy.io as scio

class SymDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, proj_dir,noise_trans):
        if mode == 'train':
            self.path = proj_dir + 'datasets/scan2cad/dataset_config/train.txt'
        elif mode == 'holdout_scene':
            self.path = proj_dir + 'datasets/scan2cad/dataset_config/holdout_scene.txt'
        elif mode == 'holdout_view':
            self.path = proj_dir + 'datasets/scan2cad/dataset_config/holdout_view.txt'

        self.num_pt = num_pt
        self.root = root + 'rgbd/' + mode
        self.symdir = root + 'model_symmetry/'
        self.projdir = proj_dir
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.cld = {}

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])  # 480*640, xmap[i,:]==i
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])  # 480*640, ymap[j,:]==j

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 100
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]
        self.categories = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243', '04401088', '04460130', '04468005', '04530566', '04554684']
        self.num_pt_mesh_small = 1000  # num_point_mesh
        self.num_pt_mesh_large = 2600
        self.front_num = 2
        self.flag = 0
        # print(len(self.list))

    def __getitem__(self, index):
        while 1:
            self.flag = 0
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
            img = Image.open('{0}/{1}-color.jpg'.format(self.root, self.list[index]))
            depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
            label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
            obj_all = meta['cls_indexes'].flatten()
            for idx in range(obj_all.shape[0]):
                obj_choose = meta['cls_dirs'][idx]
                class_name = obj_choose[:8]
                ins_name = obj_choose[9:]
                sym_dir = self.symdir
                sym_file = sym_dir + class_name + '/' + ins_name.strip() + '/' + 'models/' + 'model_normalized_sym.txt'
                if os.path.exists(sym_file) == False:
                    self.flag = 0
                    break
                else:
                    model_s = np.loadtxt(sym_file)
                    syms = model_s[1:, :]
                    check_ = np.zeros((4, 3))
                    check_sym = (syms != check_)
                    nozero = np.nonzero(check_sym)
                    row_id = nozero[0]
                    if (row_id.shape[0] == 0) or (model_s.shape[0] != 5):
                        self.flag = 0
                        break
                    else:
                        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                        mask_label = ma.getmaskarray(ma.masked_equal(label, idx + 1))
                        mask = mask_label * mask_depth
                        mask_real = len(mask.nonzero()[0])
                        if mask_real > self.minimum_num_pt:
                            self.flag = 1
                            break
            if self.flag == 1:
                break
            else:
                index += 1
                continue

        cam = meta['intrinsic_matrix']
        cam_cx = cam[0, 2]
        cam_cy = cam[1, 2]
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]

        cam_intri = [cam_cx, cam_cy, cam_fx, cam_fy]
        cam_intri = np.array(cam_intri)

        obj = meta['cls_indexes'].flatten()

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img_masked = img

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, -1].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        order = idx
        center = model_s[0, :]
        syms = model_s[1:, :]

        check_ = np.zeros((4, 3))
        check_sym = (syms != check_)
        nozero = np.nonzero(check_sym)
        row_id = nozero[0]

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

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            if len(choose)==0:
                debuge2 = 0
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
                debug1 = 0
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(
            np.float32)  # (1000,1)get masked depth
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        choose = np.array([choose])  # (1,1000)

        cam_scale = meta['factor_depth'][0][0]  # cam_scale = 10000
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # (1000,3)
        # print('cloud_shape = ', cloud.shape)
        if self.add_noise:
            cloud = np.add(cloud, add_t)  # (1000,3)

        target_s = np.add(np.dot(multi_s, target_r.T), target_t)
        sym_len = np.linalg.norm(target_s[1:,:] - target_s[0,:],axis=1)

        target_num = target_s.shape[0] - 1
        obj_idx = self.categories.index(class_name)

        occ = meta['occlusion_ratios'][0][idx]
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.LongTensor([int(obj_idx)]), \
               torch.from_numpy(target_s.astype(np.float32)), \
               torch.LongTensor([target_num]), \
               torch.LongTensor([target_mode]), \
               mask_real / (640 * 480)
               # occ,\
               # depth,\
               # cam_intri, \
               # cam_scale, \
               # self.list[index], \
               # order


    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):

        return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]
img_width = 480
img_length = 640


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
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
