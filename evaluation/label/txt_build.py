import os
import random

# 设置图片路径和txt存放的路径
file_path = r'dataset_voc/SegmentationClass'
saveBasePath = r'dataset_voc/'

# 设置数据集比例，其中trainval是指train+val
train_percent = 0.8
val_percent = 0.2

# 求出数据总的数目
total_image = os.listdir(file_path)                  # 将文件名存为一个列表，这时是包含拓展名的
num = len(total_image)                               # 总的文件数
list = range(num)

# 求出各部分的数目
train_number = int(num * train_percent)
val_number = int(num * val_percent)
trainval_number = int(train_number + val_number)

# 各部分的样本
trainval = random.sample(list, trainval_number)      # 从总的数据集中，先挑train+val，再从train+val中，挑出train
train = random.sample(trainval, train_number)
print(trainval)
print(train)

# 确认数据集各部分的数目
print("train加val的数目", trainval_number)
print("train的数目", train_number)

# 建立每部分的txt
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
#ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

# 进行写文件名
for i in list:
    name = total_image[i][:-4] + '\n'               # 去掉拓展名
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

# 关闭txt文件
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
