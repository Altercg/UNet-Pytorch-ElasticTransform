# coding=utf-8
'''

'''
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision.transforms as transforms
from unet import Unet
from data_process import *
from tqdm import tqdm
import numpy as np
import skimage.io as io
from overlap_tile import rebuild_images
import torch.nn.functional as F

# 训练好的模型存储位置
PATH = 'model/unet_model.pt'

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.ToTensor()])

batch_size = 3
def train_model(model, criterion, optimizer, dataload, num_epochs=50):
    '''
        训练模型：其中包含输入图像们的重叠操作
    '''
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for imgs, y in dataload:     # 进度条库
            step += 1
            # 一张图片的patch存放处
            inputs = []
            outputs = []
            # 切分图像导入

            # mirror_img = overlap_tile(np.array(img_x), (696, 696), (92, 92))
            # 镜像图像切片
            # patches_img = extract_ordered_patches(mirror_img, (572, 572), (124, 124))

            for x in imgs:
                inputs.append(x.to(device))
            # 输入这张图片的
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # 一张图片的每一个patch输入网络
            # 输出的图片储存起来做拼接
            for i in inputs:
                out = model(i).cuda().data.cpu()
                outputs.append(out.numpy())
            # 拼接
            outputs = np.array(outputs)
            outputs = rebuild_images(outputs, (512, 512), (124, 124))
            # 转为tensor之后放入cuda
            outputs = torch.tensor(outputs, requires_grad=True)

            # 输出看看
            # img = torch.squeeze(outputs).detach().numpy()
            # # img_y = img[:, :, np.newaxis]
            # # img_y = img_y[:, :, 0]
            # io.imsave("dataset/train/0_predict.png", img[0])
            # io.imsave("dataset/train/1_predict.png", img[1])

            output = outputs.to(device)
            # 损失和更新
            loss = criterion(output, labels.long())
            # 反向更新参数
            loss.backward()     
            optimizer.step()
            epoch_loss += loss.item()
            # print(loss.item())    # loss不断抖动居高不下？？？
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
    torch.save(best_model.state_dict(), PATH)   # 保存模型
    return best_model


# 训练模型
def train():
    model = Unet(1, 2).to(device)
    
    # criterion = nn.BCEWithLogitsLoss()  # sigmoid函数+交叉熵的组合，二分类
    criterion = nn.CrossEntropyLoss() # softmax+交叉熵组合， 多类
    optimizer = optim.Adam(model.parameters())
    train_dataset = TrainDataset("dataset/train/image",
                                 "dataset/train/label",
                                 transform=x_transforms,
                                 target_transform=y_transforms)
    # 生成一个可迭代对象，实现每一次迭代加载一个batch的对象
    # dataloder迭代器每次可以从dataset基于某种原则取出一个batch的数据
    dataloaders = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 保存模型的输出结果
def test():
    model = Unet(1, 2)
    model.load_state_dict(torch.load(PATH))     # 加载模型
    test_dataset = TestDataset("dataset/test",
                               transform=x_transforms,
                               target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()    # 即使用模型到测试模式，不反向传播
    with torch.no_grad():   # 上下文管理器，被这部分圈起来的不会追踪梯度
        # enumerate可遍历的数组对象组合成一个(下标，数据)
        for index, imgs in enumerate(dataloaders):
            outputs = []
            for i in imgs:
                out = model(i)
                outputs.append(out.numpy())
            # 拼接
            outputs = np.array(outputs)
            outputs = torch.tensor(rebuild_images(outputs,(512, 512), (124, 124)))
            # 增加维度，（x*y*1维)
            outputs = torch.squeeze(outputs).numpy()
            img_y = outputs[:,:,np.newaxis]  
            img_y = img_y[:, :, 0]
            io.imsave("dataset/test/" + str(index) + "_predict.png", img_y)


if __name__ == '__main__':
    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("-"*20)
    print("开始预测")
    # test()
