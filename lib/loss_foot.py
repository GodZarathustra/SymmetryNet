### rotation 的 loss 算了mode==1 和mode==2的(mode!=0),
#   ref 算了 mode==0 和mode ==2（mode!=1）
#   co_loss改成了平方
#   权值改了
##############0113 更新：和loss2 一样
##############0114 更新：加了co_loss,调整了权值
from torch.nn.modules.loss import _Loss
import torch
import math
import itertools
import numpy as np
# import self_loss
# from mirror import mirror_point
from scipy.optimize import linear_sum_assignment

def loss_calculation(pred_c, pred_cent, pred_foot_ref, pred_rot, pred_num, pred_mode,
                     target_s, idx, points, w, target_num, target_mode):

    bs = 1
    num_p = 1000

    points = points.contiguous().view(bs * num_p, 1, 3)  # 1000*1*3 input point cloud
    # pred_c = pred_c.contiguous().view(bs * num_p)
    pred_num = pred_num.contiguous().view(bs * num_p, 3)
    pred_mode = pred_mode.contiguous().view(bs * num_p, 3)
    pred_cent = pred_cent.contiguous().view(bs * num_p, 1, 3)
    pred_ref = pred_foot_ref.contiguous().view(bs * num_p, -1, 3)
    pred_foot_ref = pred_foot_ref.contiguous().view(bs * num_p, -1, 3)
    pred_rot = pred_rot.contiguous().view(bs * num_p, -1, 3)
    pred_rot_foot = pred_rot.view(bs * num_p, -1, 3)

    target_mode = target_mode.view(-1)
    target_mode_ = target_mode
    target_mode = target_mode.view(bs, 1, 1).repeat(1, num_p, 1).view(bs * num_p)

    target_s_ = target_s
    target_s = target_s.view(bs, 1, -1, 3).repeat(1, num_p, 1, 1).view(bs * num_p, -1, 3)
    target_cent = target_s[:, 0, :].view(bs * num_p, -1, 3)
    target_sym = target_s[:, 1:, :].view(bs * num_p, -1, 3)
    target_sym_vec = torch.add(target_sym, -target_cent)  # 1000,-1 ,3



    cent_pred = torch.add(points, pred_cent)  # 1000,1,3
    ref_pred = torch.add(points, pred_ref)
    ref_foot_pred = torch.add(points, pred_foot_ref)

    cross_entropy = torch.nn.CrossEntropyLoss()
    mode_loss = cross_entropy(pred_mode, target_mode)
    center_loss = torch.mean(torch.norm((cent_pred - target_cent), dim=2), dim=1)  # (1000)

#########cost matrix#########cos ang of pred norm and target norm############
    mean_pred_ref = torch.mean(pred_ref, dim=0)
    mean_ref_pred = torch.mean(ref_pred, dim=0)
    mean_target_vec = torch.mean(target_sym_vec, dim=0)
    cost_matrix = torch.zeros(mean_pred_ref.shape[0], target_sym_vec.shape[1])

    for i in range(mean_pred_ref.shape[0]):
        for j in range(mean_target_vec.shape[0]):
            a = mean_pred_ref[i, :].view(1, 3)
            b = mean_target_vec[j, :].view(3, 1)
            product = torch.mm(a, b)
            norm_a = torch.norm(a, dim=1)
            norm_b = torch.norm(b,dim=0)
            cost = torch.abs(product / (torch.add(norm_a, 0.00001)*torch.add(norm_b, 0.00001)))
            cost_matrix[i, j] = torch.acos(cost.reshape(-1))

######optimal assiment####### min cost for each point is the point-wise solusion
    row_id_, col_id_ = linear_sum_assignment(cost_matrix.detach().numpy())
    if mean_target_vec.shape[0] >1:
        corr = np.array([row_id_,col_id_]).T
        ordered_id = corr[corr[:,1].argsort()]
        row_id = ordered_id[:,0]
        col_id = ordered_id[:,1]
    else :
        row_id = row_id_
        col_id = col_id_
    ref_out = ref_pred[:, row_id, :]
    ref_out_vec = pred_ref[:, row_id, :]

    ref_out_foot = ref_foot_pred[:, row_id, :]
    ref_out_vec_foot = pred_foot_ref[:, row_id, :]

    mean_ref_out_vec = mean_pred_ref[row_id, :]
    # total_cost = cost_matrix[row_id, col_id].sum() / mean_target_vec.shape[0]

    target_id = label_trans(torch.tensor(row_id)).cuda().float()
    target_ref = ref_pt(points, target_cent, target_sym_vec)[:, col_id, :].cuda()
    target_foot_ref = points + 0.5*(target_ref-points)
    target_sym_vec_orderd = target_sym_vec[:,col_id,:]

    id_loss = torch.nn.BCELoss()
    mean_pred_num = torch.mean(pred_num, dim=0)
    num_loss = id_loss(mean_pred_num, target_id)   # (1)

#######caculate dis error
    # diff_dis = torch.zeros(1000,ref_out.shape[1])
    # for i in range(ref_out.shape[1]):
    #     cent_to_ref = torch.norm(cent_pred-ref_pred[:, i, :].view(1000,1,3), dim=2)
    #     cent_to_pt = torch.norm(cent_pred-points, dim=2)
    #     diff_dis_ = torch.abs(cent_to_pt-cent_to_ref)
    #     diff_dis[:, i] = diff_dis_.view(1000)
    # diff_loss = torch.mean(diff_dis, dim=0).cuda()

    ref_loss = 0
    ref_foot_loss = 0
    ref_co_loss = 0
    rot_foot_loss = 0
    rot_co_loss = 0
    if target_mode_ != 0:
        rot_foot_pred = torch.add(points, pred_rot_foot)#1000,1,3
        point_to_cent = torch.add(-points, target_cent)#1000,1,3
        product = torch.bmm(target_sym_vec.view(1000,1,3), point_to_cent.view(1000,3,1)).view(1000)
        cos = product / (
                torch.norm(point_to_cent.view(1000, 3), dim=1) * torch.norm(target_sym_vec.view(1000,3), dim=1)+0.00001).view(1000)
        point_to_cent_nom = torch.norm(point_to_cent.view(1000,3), dim=1)
        cent_to_foot = (-point_to_cent_nom * cos).view(1000,1).repeat(1,3)*(target_sym_vec.view(1000,3))
        target_rot_foot = target_cent + cent_to_foot.view(1000,1,3)
        # target_rot_foot_new = foot_all(target_cent.view(1000,3),target_sym.view(1000,3),points.view(1000,3))
        rot_foot_loss = torch.mean(torch.norm(target_rot_foot - rot_foot_pred, dim=2), dim=1).cuda() #0.1
        # print('rot_foot_loss=', torch.mean(rot_foot_loss).data.cpu())
        pt_to_foot = rot_foot_pred - points
        rot_co_loss = torch.mean(torch.bmm(pt_to_foot.view(1000,1,3), cent_to_foot.view(1000,3,1)).view(-1)).cuda()**(2)#0.001
        debuger_ = 0

    if target_mode_  != 1:
        ref_out_len = torch.norm(ref_out_vec, dim=2)
        ref_distance = torch.norm((ref_out - target_ref), dim=2)
        # ref_loss = torch.mean(torch.div(ref_distance, ref_out_len+0.00001), dim=1).cuda()
        ref_foot_loss = torch.mean(torch.norm(ref_out_foot - target_foot_ref, dim=2), dim=1).cuda()#0.1
        # ref_co_loss = torch.mean(torch.mean(torch.norm(ref_out_vec_foot * 2 - pred_ref[:, row_id, :], dim=2), dim=1)).cuda()**(2)#0.1
        debuger_ = 0

#######caculate angle error
    if target_mode_ == 1:
        debuger_1 = 0
        pred_axis = cent_pred.view(1000,3) - rot_foot_pred.view(1000,3)
        norm_axis = torch.norm(pred_axis, dim=1).view(1000,1).repeat(1,3)
        best_norm = (pred_axis/(norm_axis+0.000001)).reshape(1000,1,3)
        target_norm = target_sym_vec_orderd[0, :].view(1, 3, 1).repeat(1000,1,1)
        products = torch.abs(torch.bmm(best_norm, target_norm))
    else:
        best_ref = torch.mean(ref_out_vec, dim=0)
        products = torch.zeros(best_ref.shape[0])
        for i in range(best_ref.shape[0]):
            best_norm = best_ref[i, :].view(1, 3).cuda()
            target_norm = target_sym_vec_orderd[0, i, :].view(3, 1)
            product = torch.abs(torch.mm(best_norm, target_norm) / (
                        torch.norm(best_norm, dim=1) * torch.norm(target_norm.contiguous().transpose(1, 0), dim=1)))
            products[i] = product
    # ref_foot_loss =0.4         rot_foot_loss = 2     ref_loss=2
    dis = torch.mean(w * center_loss + ref_foot_loss + rot_foot_loss, dim=0)

    # loss = torch.mean((dis * pred_c - w*torch.log(pred_c)), dim=0) + num_loss
    #ref_co_loss =0.07,          rot_co_loss = 0.01
    loss = dis + 2 * num_loss + mode_loss #+ w * 0.5*ref_co_loss + 0.5* w * rot_co_loss

    # dis_all = torch.mean(dis.view(bs, num_p), dim=1)
    center_dis = torch.mean(center_loss.view(bs, num_p), dim=1)
    # ref_dis = torch.mean(ref_loss.contiguous().view(bs, num_p), dim=1)
    ref_dis = dis
    # if target_mode_ != 0:
    #     angle_error = torch.mean(rot_foot_loss)*100
    #     debuger_2 = 0
    # if target_mode_ != 1:
    #     angle_error = torch.mean(torch.acos(products) / math.pi * 180)

    angle_error = torch.mean(torch.acos(products) / math.pi * 180)

    error_num = torch.mean(num_loss)
    error_mode = torch.mean(mode_loss)

    return loss, loss, center_dis.data.cpu(), ref_dis, angle_error, error_num.data.cpu(), error_mode.cpu()


def ref_pt(pt, cent, sym_vect):
    pt_pred = torch.zeros(sym_vect.shape)
    for i in range(sym_vect.shape[1]):
        center = cent.view(1000,3,1)
        norm = sym_vect[:, i, :].view(1000,1,3)
        d = -torch.bmm(norm,center)
        pt_ = pt-2*(torch.bmm(norm, pt.view(1000,3,1)) + d)*norm
        pt_pred[:, i, :] = pt_.view(1000,3)
    return pt_pred

def all_permutation(task_matrix):
    number_of_choice = len(task_matrix)
    number_of_target = task_matrix.shape[1]
    solutions = []
    values = []

    if number_of_target == 1:
        best_solution = torch.max(task_matrix)
        sol_index = task_matrix.eq(best_solution)

    for each_solution in itertools.permutations(range(number_of_choice)):
        each_solution = list(each_solution)
        solution = []
        value = 0
        for i in range(len(task_matrix)):
            value += task_matrix[i][each_solution[i]]
            solution.append(task_matrix[i][each_solution[i]])
        values.append(value)
        solutions.append(solution)

    min_cost = np.min(values)
    best_solution = solutions[values.index(min_cost)]
    return min_cost, best_solution
def foot_all(begin, end, pt):
    begin_x = begin[:,0]
    begin_y = begin[:,1]
    begin_z = begin[:,2]

    end_x = end[:,0]
    end_y = end[:,1]
    end_z = end[:,2]

    pt_x = pt[:,0]
    pt_y = pt[:,1]
    pt_z = pt[:,2]

    dx = begin_x - end_x
    dy = begin_y - end_y
    dz = begin_z - end_z

    if (abs(dx).data.cpu().numpy().all() < 0.00000001) and (abs(dy).data.cpu().numpy().all() < 0.00000001) and (abs(dz).data.cpu().numpy().all() < 0.00000001):
        retVal = begin

    u = (pt_x - begin_x) * (begin_x - end_x) + \
        (pt_y - begin_y) * (begin_y - end_y) + (pt_z - begin_z) * (begin_z - end_z)
    u = u / ((dx * dx) + (dy * dy) + (dz * dz))

    retVal_x = begin_x + u * dx
    retVal_y = begin_y + u * dy
    retVal_z = begin_z + u * dz
    retVal = torch.cat((retVal_x.reshape(1000,1),retVal_y.reshape(1000,1),retVal_z.reshape(1000,1)),1)

    return retVal

def label_trans(input):
    if input.shape[0] == 3:
        label = torch.tensor([1, 1, 1])
    if input.shape[0] == 2:
        if input.equal(torch.tensor([0, 1])) or input.equal(torch.tensor([1, 0])):
            label = torch.tensor([1, 1, 0])

        if input.equal(torch.tensor([0, 2])) or input.equal(torch.tensor([2, 0])):
            label = torch.tensor([1, 0, 1])

        if input.equal(torch.tensor([1, 2])) or input.equal(torch.tensor([2, 1])):
            label = torch.tensor([0, 1, 1])
    if input.shape[0] == 1:
        if input.equal(torch.tensor([0])):
            label = torch.tensor([1, 0, 0])

        if input.equal(torch.tensor([1])):
            label = torch.tensor([0, 1, 0])

        if input.equal(torch.tensor([2])):
            label = torch.tensor([0, 0, 1])
    return label

#点绕任意向量旋转，右手系
#输入参数old_x，old_y，old_z为旋转前空间点的坐标
#vx，vy，vz为旋转轴向量
#theta为旋转角度角度制，范围在-180到180
#返回值为旋转后坐标点
def RotateByVector(oldpoint, vector, theta):
    r = theta * math.pi / 180
    c = math.cos(r)
    s = math.sin(r)
    old_x = oldpoint[:, 0]
    old_y = oldpoint[:, 1]
    old_z = oldpoint[:, 2]
    vx = vector[:, 0]
    vy = vector[:, 1]
    vz = vector[:, 2]
    new_x = (vx*vx*(1 - c) + c) * old_x + (vx*vy*(1 - c) - vz*s) * old_y + (vx*vz*(1 - c) + vy*s) * old_z
    new_y = (vy*vx*(1 - c) + vz*s) * old_x + (vy*vy*(1 - c) + c) * old_y + (vy*vz*(1 - c) - vx*s) * old_z
    new_z = (vx*vz*(1 - c) - vy*s) * old_x + (vy*vz*(1 - c) + vx*s) * old_y + (vz*vz*(1 - c) + c) * old_z
    return torch.stack((new_x, new_y, new_z),dim=1)

class Loss(_Loss):

    def __init__(self, num_points_mesh):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        # self.sym_list = sym_list

    def forward(self, pred_c, pred_cent,  pred_foot_ref, pred_rot,pred_num, pred_mode,
                            target_s, idx, points, w, target_num, target_mode):

        return loss_calculation(pred_c, pred_cent, pred_foot_ref, pred_rot,pred_num, pred_mode,
                            target_s, idx, points, w, target_num, target_mode)
