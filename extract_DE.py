import scipy.io
import os
import numpy as np

extrac_path = 'F:/EEG/git/seed/DGCNN-main/session3/'
label_path = 'F:/EEG/git/seed/DGCNN-main/session3/label.mat'
save_path = 'F:/EEG/git/seed/DGCNN-main/DE/session3_4channel/'

# 需要选择的通道索引，例如选择第1,3,4,5,6,14个通道（索引从0开始）
selected_channels = [0,2,5,13]

dir_list = os.listdir(extrac_path)

label = scipy.io.loadmat(label_path)
label = label['label'][0]

for f in dir_list:
    if '_' not in f:
        continue

    S = scipy.io.loadmat(extrac_path + f)
    DE = []
    labelAll = []
    for i in range(15):
        data = S['de_LDS' + str(i + 1)]
        
        # 选择特定通道
        data = data[selected_channels, :]
        
        if len(DE):
            DE = np.concatenate((DE, data), axis=1)
        else:
            DE = data

        if len(labelAll):
            labelAll = np.concatenate((labelAll, np.zeros([data.shape[1], 1]) + label[i]), axis=0)
        else:
            labelAll = np.zeros([data.shape[1], 1]) + label[i]

    mdic = {"DE": DE, "labelAll": labelAll, "label": "experiment"}

    scipy.io.savemat(save_path + f, mdic)
    print(extrac_path + f, '->', save_path + f)