'''
    数据处理文件
        训练与测试数据集的读取
        训练与测试数据集的镜像策略+图像切片
'''

from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import torchvision.transforms as transforms
from overlap_tile import overlap_tile, extract_ordered_patches
import torch


# 返回一系列文件的路径数组
def train_dataset(img_root, label_root):
    imgs = []
    # 返回指定文件夹包含的文件或者文件夹名字的列表
    n = len(os.listdir(img_root))   
    # 路径拼接
    for i in range(n):
        img = os.path.join(img_root, "%d.png" % i)  
        label = os.path.join(label_root, "%d.png" % i)
        imgs.append((img, label))
    return imgs


def test_dataset(img_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, "%d.png" % i)
        imgs.append(img)
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None,
                 target_transform=None):
        imgs = train_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        # 导入x图片
        img_x = Image.open(x_path)
        # 镜像策略
        # mirror_img = overlap_tile(np.array(img_x), (696, 696), (92, 92))
        # 镜像图像切片
        # patches_img = extract_ordered_patches(mirror_img, (572, 572), (124, 124))
        # 导入y图片
        img_y = Image.open(y_path)
        # 输入图片的处理
        if self.transform is not None:
            # img_x = []
            # for i in range(patches_img.shape[0]):
            #     img_x.append(self.transform(patches_img[i]))\
            img_x = self.target_transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            # img_y = torch.squeeze(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, img_root, transform=None, target_transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path)
        # mirror_img = overlap_tile(np.array(img_x), (696, 696), (92, 92))
        # patches_img = extract_ordered_patches(mirror_img, (572, 572), (124, 124))
        if self.transform is not None:
            # img_x = []
            # for i in range(patches_img.shape[0]):
                # img_x.append(self.transform(patches_img[i]))
            img_x = self.target_transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)
