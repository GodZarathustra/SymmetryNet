import math
import torch


class Swp1d(torch.nn.Module):
    '''

    '''

    def __init__(self, bs, in_features, out_features, bias=False):
        super(Swp1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(bs, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        #random weights drawn from Gaussian distributions with a fixed standard deviation of 0.005.
        # self.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.weight.size())
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        # input = torch.pow(input, 2)
        y = torch.bmm(input, self.weight)
        return y

#
# N, D_in, C_in, D_out, C_out = 10, 1000,1024, 1, 1024  # 一共10组样本，输入特征为5，输出特征为3
#
#
#
# class Swpnet(torch.nn.Module):
#     def __init__(self):
#         super(Swpnet, self).__init__()  # 第一句话，调用父类的构造函数
#         self.mylayer1 = Swp1d(N, D_in, D_out)
#
#     def forward(self, x):
#         x = self.mylayer1(x)
#         return x
#
#
# model = Swpnet()
# print(model)
# '''运行结果为：
# MyNet(
#   (mylayer1): MyLayer()   # 这就是自己定义的一个层
# )
# '''
#
# # 创建输入、输出数据
# x = torch.randn(N, C_in, D_in)  # （10，1024,1000）
# y = torch.randn(N, C_out, D_out)  # （10，1024,1）
#
# # 定义损失函数
# loss_fn = torch.nn.MSELoss(reduction='sum')
#
# learning_rate = 1e-4
# # 构造一个optimizer对象
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for t in range(10):  #
#
#     # 第一步：数据的前向传播，计算预测值p_pred
#     y_pred = model(x)
#
#     # 第二步：计算计算预测值p_pred与真实值的误差
#     loss = loss_fn(y_pred, y)
#     print(f"第 {t} 个epoch, 损失是 {loss.item()}")
#
#     # 在反向传播之前，将模型的梯度归零，这
#     optimizer.zero_grad()
#
#     # 第三步：反向传播误差
#     loss.backward()
#
#     # 直接通过梯度一步到位，更新完整个网络的训练参数
#     optimizer.step()