import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import random
import numpy.ma as ma
import scipy.io as scio

class SymDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, proj_dir,noise_trans, refine):
        if mode == 'train':
            self.path = proj_dir+'datasets/ycb/dataset_config/train_ls.txt'
        elif mode == 'test':
            self.path = proj_dir+'datasets/ycb/dataset_config/test_ls.txt'

        self.num_pt = num_pt
        self.root = root+'rgbd/'+mode
        self.modeldir = root
        self.symdir = root+'model_symmetry/'
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

        class_file = open(self.projdir+'datasets/ycb/dataset_config/classes.txt')
        class_id = 1  # from 1 to 21
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.modeldir, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()
            
            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])  # 480*640, xmap[i,:]==i
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])  # 480*640, ymap[j,:]==j
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]
        self.ref_sym_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 18, 19, 20]
        self.one_sym_list = [0, 3, 5, 10, 11, 13]
        self.two_sym_list = [4, 18, 19, 20]
        self.three_sym_list = [1, 2, 6, 7, 8, 15]
        self.only_axis_list = [12, 17]
        self.axis_and_ref_list = [0, 3, 5]
        self.num_pt_mesh_small = 500  # num_point_mesh
        self.num_pt_mesh_large = 2600
        self.refine = refine
        self.front_num = 2

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        symmetries = np.loadtxt(self.symdir+'symmetries.txt')
        symmetries = symmetries.reshape(21, 5, 3)

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        cam_intri = [cam_cx, cam_cy, cam_fx, cam_fy]
        cam_intri = np.array(cam_intri)
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.list)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32) # get class index

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            mask_real = len(mask.nonzero()[0])
            if mask_real > self.minimum_num_pt :
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]


        img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        order = idx
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # transform sym vectors into points
        cls_idx = int(obj[idx])-1
        model_s = symmetries[cls_idx, :, :]
        target_mode = 0
        if cls_idx in self.one_sym_list:
            multi_s = np.zeros((2, 3))
            multi_s[0, :] = model_s[0, :]
            multi_s[1, :] = model_s[1, :] + model_s[0, :]
            if cls_idx in self.axis_and_ref_list:
                target_mode = 2
            else:
                target_mode = 0
        elif cls_idx in self.only_axis_list:
            multi_s = np.zeros((2, 3))
            multi_s[0, :] = model_s[0, :]
            multi_s[1, :] = model_s[4, :] + model_s[0, :]
            target_mode = 1
        elif cls_idx in self.two_sym_list:
            multi_s = np.zeros((3, 3))
            multi_s[0, :] = model_s[0, :]
            multi_s[1, :] = model_s[1, :] + model_s[0, :]
            multi_s[2, :] = model_s[2, :] + model_s[0, :]
            target_mode = 0
        elif cls_idx in self.three_sym_list:
            multi_s = np.zeros((4, 3))
            multi_s[0, :] = model_s[0, :]
            multi_s[1, :] = model_s[1, :] + model_s[0, :]
            multi_s[2, :] = model_s[2, :] + model_s[0, :]
            multi_s[3, :] = model_s[3, :] + model_s[0, :]
            target_mode = 0
        else:
            multi_s = np.zeros((5, 3))

            # print("not in symmetry list")
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)  #(1000,1)get masked depth
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

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        target = np.dot(model_points, target_r.T)
        target_s = np.add(np.dot(multi_s, target_r.T), target_t)
        target_num = target_s.shape[0]-1
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.LongTensor([int(obj[idx]) - 1]), \
               torch.from_numpy(target_s.astype(np.float32)),\
               torch.LongTensor([target_num]),\
               torch.LongTensor([target_mode]), \
               mask_real / (640 * 480)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
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
