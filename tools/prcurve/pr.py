import numpy as np
import matplotlib.pyplot as plt
savedir = './shapenet/'
datadir = './shapenet/data/'  # ycb rot
# file1 = datadir+'heavy-new_ins-oursdis=15.txt'
# file2 = datadir+'mid-new_ins-oursdis=15.txt'
# file3 = datadir+'light-new_ins-oursdis=15.txt'

# file1 = datadir+'heavy-new_class-oursdis=15.txt'
# file2 = datadir+'mid-new_class-oursdis=15.txt'
# file3 = datadir+'light-new_class-oursdis=15.txt'
#
# file1 = datadir+'heavy-new_view-oursdis=15.txt'
# file2 = datadir+'mid-new_view-oursdis=15.txt'
# file3 = datadir+'light-new_view-oursdis=15.txt'

file1 = datadir+'new_class-oursdis=15.txt'
file2 = datadir+'new_ins-oursdis=15.txt'
file3 = datadir+'new_view-oursdis=15.txt'

pr1 = np.loadtxt(file1)
pr2 = np.loadtxt(file2)
pr3 = np.loadtxt(file3)

recall1 = pr1[:,0].tolist()
precision1 = pr1[:,1].tolist()

recall2 = pr2[:,0].tolist()
precision2 = pr2[:,1].tolist()

recall3 = pr3[:,0].tolist()
precision3 = pr3[:,1].tolist()

plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title('PR-curve of shapenet', fontsize=15)
plt.axes().set_aspect(1, 'box')
# plt.plot(recall1, precision1, linewidth = 2,  color = 'tab:red', zorder = 10, label='heavy-occlusion')
# plt.plot(recall2, precision2, linewidth = 2,  color = 'tab:green', zorder = 10, label='mid-occlusion')
# plt.plot(recall3, precision3, linewidth = 2,  color = 'tab:blue', zorder = 10, label='light-occlusion')

plt.plot(recall1, precision1, linewidth = 2,  color = 'tab:red', zorder = 10, label='holdout category')
plt.plot(recall2, precision2, linewidth = 2,  color = 'tab:green', zorder = 10, label='holdout instance')
plt.plot(recall3, precision3, linewidth = 2,  color = 'tab:blue', zorder = 10, label='holdout view')
plt.legend(loc='upper right', fontsize=13)
plt.grid(True)
plt.savefig(savedir+"all.png")
# plt.savefig(savedir+"ycb-ref-test.pdf")
plt.show()
