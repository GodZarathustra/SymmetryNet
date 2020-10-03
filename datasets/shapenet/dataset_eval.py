import torch.utils.data as data
import cv2
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms

import numpy.ma as ma

proj_dir = '/home/dell/yifeis/symnet/'

class SymDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/split_valid2/train_ls.txt'
        elif mode == 'holdout_ins':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/split_valid2/holdout_ins_ls.txt'
        elif mode == 'holdout_view':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/split_valid2/holdout_view_ls.txt'
        elif mode == 'holdout_class':
            self.path = proj_dir + 'datasets/shapenet/dataset_config/split_valid2/holdout_class_ls.txt'

        self.num_pt = num_pt
        self.root = root
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

        self.xmap = np.array([[j for i in range(960)] for j in range(540)])  # 480*640, xmap[i,:]==i
        self.ymap = np.array([[i for i in range(960)] for j in range(540)])  # 480*640, ymap[j,:]==j

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_small = 500  # num_point_mesh
        self.refine = refine
        self.front_num = 2
        self.name_list = np.loadtxt(proj_dir +'name_list2.txt', dtype=str ,delimiter='\n')
        self.class_id = {
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
        # print(len(self.list))
        self.no_occ = []

    def __getitem__(self, index):

        occ = np.loadtxt('{0}/{1}-occlusion.txt'.format(self.root, self.list[index]))
        rt = np.loadtxt('{0}/{1}-rt.txt'.format(self.root, self.list[index]))
        check_rt = np.zeros((4, 4))
        check_depth = 255*np.ones((540, 960, 3))
        depth_ = cv2.imread('{0}/{1}-depth-crop-occlusion.png'.format(self.root, self.list[index]))
        # depth_ = cv2.resize(depth_, (640,480))
        cam_ = np.loadtxt('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))
        if cam_.reshape(-1).shape[0] != 9:
            print('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))

        input_file = self.list[index]
        # class_key = input_file[20:23]
        # input_id = int(input_file[20:23])
        # ins_num = int(input_file[24:28])
        class_key = input_file[:3]
        input_id = int(class_key)
        ins_num = int(input_file[4:8])
        cls_idx = input_id
        class_name = self.class_id[class_key]
        instance_ls = self.name_list[cls_idx][1:-1].split(",")
        ins_name = instance_ls[ins_num][2:-1]
        sym_dir = '/home/dell/yifeis/shapenetcore/'
        sym_file = sym_dir + class_name + '/' + ins_name + '/' + 'model_sym.txt'
        model_s = np.loadtxt(sym_file)
        syms = model_s[1:, :]
        check_ = np.zeros((4, 3))
        check_sym = (syms != check_)
        nozero = np.nonzero(check_sym)
        row_id = nozero[0]

        input_file = self.list[index]
        class_key = input_file[:3]
        input_id = int(class_key)
        ins_num = int(input_file[4:8])
        img_ = cv2.imread('{0}/{1}-color-crop-occlusion.png'.format(self.root, self.list[index]))  # 540*960
        img = img_

        depth = depth_[:, :, 0]
        # depth = cv2.resize(depth_, (640, 480))[:,:,0]
        cam = np.loadtxt('{0}/{1}-k-crop.txt'.format(self.root, self.list[index]))
        cam_cx = cam[0, 2]
        cam_cy = cam[1, 2]
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]

        cam_intri = [cam_cx ,cam_cy ,cam_fx ,cam_fy]
        cam_intri = np.array(cam_intri)

        add_front = False

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

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        ori = choose
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype \
            (np.float32)  # (1000,1)get masked depth
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  # (1000,1)
        choose = np.array([choose])  # (1,1000)

        depth_masked_ = depth[rmin:rmax, cmin:cmax].flatten()[ori][:, np.newaxis].astype \
            (np.float32)  # (1000,1)get masked depth
        xmap_masked_ = self.xmap[rmin:rmax, cmin:cmax].flatten()[ori][:, np.newaxis].astype(np.float32)  # (1000,1)
        ymap_masked_ = self.ymap[rmin:rmax, cmin:cmax].flatten()[ori][:, np.newaxis].astype(np.float32)  # (1000,1)
        ori = np.array([ori])  # (1,1000)

        #####1000 points
        cam_scale = 100  # cam_scale = 10000
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # (1000,3)

        #########all points
        pt2_ = depth_masked_ / cam_scale
        pt0_ = (ymap_masked_ - cam_cx) * pt2_ / cam_fx
        pt1_ = (xmap_masked_ - cam_cy) * pt2_ / cam_fy
        cloud_ = np.concatenate((pt0_, pt1_, pt2_), axis=1)  # (n,3)

        target_r = rt[:-1, :-1]
        target_t = rt[:-1, 3]
        target_s = np.add(np.dot(multi_s, target_r.T), target_t)

        # s_len = np.linalg.norm(target_s[1]-target_s[0])
        target_num = target_s.shape[0 ] -1
        # print('load one')
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.LongTensor([int(idx)]), \
               torch.from_numpy(target_s.astype(np.float32)), \
               torch.LongTensor([target_num]), \
               torch.LongTensor([target_mode]),\
               occ,\
               cloud_,\
               depth,\
               cam_intri,\
               mask_real/(540*960),\
               self.list[index]

    def __len__(self):
        return self.length

    def get_num_points_mesh(self):

        return self.num_pt_mesh_small

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560]
# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
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
