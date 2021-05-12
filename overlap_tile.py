'''
    overlap_tile 镜像策略
    extract_ordered_patches 切片
    rebuild_images 重叠

    参考：https://www.jianshu.com/p/f5f5c9f0222e
'''

import numpy as np
from skimage.util.dtype import img_as_bool

def overlap_tile(imgs, patch_size, stride_size):
    """
        imgs （B,C,H,W）（3,1,512,512）
        patch_size: 镜像后图像块大小（696,696）
        stride_size: 间隔大小（92,92）
        return: numpy.narray
    """
    # 确定为灰度图像
    assert imgs.ndim > 2
    # batch size 为 1
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=0)
    # 获取宽高度
    b, c, h, w = imgs.shape
    # 镜像后图片的宽高度
    patch_h, patch_w = patch_size
    # 裁剪间隔
    stride_h, stride_w = stride_size
    # left_h, left_w获得0, 因为数据刚好裁剪完成
    left_h, left_w = (h - patch_h) % stride_h, (w - patch_w) % stride_w
    # pad_h和pad_w为0，表示裁剪完成不需要padding
    pad_h, pad_w = (stride_h - left_h) % stride_h, (stride_w - left_w) % stride_w
    if pad_h == 0:
        # 填充后的数组大小
        pad_imgs = np.empty((b, c, h + stride_h * 2, w), dtype=imgs.dtype)
        start_y = stride_h
        end_y = start_y + h
        # 多层切片,镜像
        for i, img in enumerate(imgs):
            pad_imgs[i, :, start_y:end_y, :] = img
            pad_imgs[i, :, :start_y, :] = img[:, :start_y, :][:, ::-1]  
            pad_imgs[i, :, end_y:, :] = img[:, h - stride_h:, :][:, ::-1]

        imgs = pad_imgs

    if pad_w == 0:
        # 可能垂直方向已经镜像操作导致高度改变
        h = imgs.shape[2]
        pad_imgs = np.empty((b, c, h, w + stride_w * 2), dtype=imgs.dtype)
        start_x = stride_w
        end_x = start_x + w
        
        for i, img in enumerate(imgs):
            pad_imgs[i, :, :, start_x:end_x] = img
            pad_imgs[i, :, :, :start_x] = img[:, :, :start_x][:, :, ::-1]
            pad_imgs[i, :, :, end_x:] = img[:, :, w - stride_w:][:, :, ::-1]

        imgs = pad_imgs

    return imgs


def extract_ordered_patches(img, patch_size, stride_size):
    """
        img: (696,696)
        patch_size: 裁剪后图像块大小（572，572）
        stride_size: 间隔大小（124，124）
        return: numpy,narray
    """
    assert img.ndim == 2

    h, w = img.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride_size
    assert (h - patch_h) % stride_h == 0 and (w-patch_w) % stride_w == 0

    # y方向上的切片数
    n_patches_y = (h - patch_h) // stride_h + 1
    # x方向上的切片数
    n_patches_x = (w - patch_w) // stride_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    # n_patches = n_patches_per_img * b
    patches = np.empty((n_patches_per_img, patch_h, patch_w), dtype=img.dtype)
    patche_idx = 0
    # 第i列
    for i in range(n_patches_y):
        # 第j行
        for j in range(n_patches_x):
            y1 = i * stride_h
            y2 = y1 + patch_h
            x1 = j * stride_w
            x2 = x1 + patch_w
            patches[patche_idx] = img[y1:y2, x1:x2]
            patche_idx += 1
    # [左上, 右上, 左下, 右下]
    return patches


def rebuild_images(patches, img_size, stride_size):
    """
        patches: 切片（4, 1, 2, 388, 388）
        img_size: 组合后的图像大小（512，512）
        stride_size:(124, 124)
    """
    assert patches.ndim == 5
    img_h, img_w = img_size
    stride_h, stride_w = stride_size
    n_patches = patches.shape[0] 
    patch_h = patches.shape[3]
    patch_w = patches.shape[4]

    assert (img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0
    n_patches_y = (img_h - patch_h) // stride_h + 1
    n_patches_x = (img_w - patch_w) // stride_w + 1

    imgs = np.zeros((1, 2, img_h, img_w))
    # 图像块之间存在重叠，需要处以重复的次数取平均
    weights = np.zeros_like(imgs)
    # 第img_idx号图
    img_idx = 0
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y1 = i * stride_h
            y2 = y1 + patch_h
            x1 = j * stride_w
            x2 = x1 + patch_w
            imgs[:, :, y1:y2, x1:x2] += patches[img_idx]
            weights[:, :, y1:y2, x1:x2] += 1
            img_idx += 1
    imgs /= weights
    return imgs.astype(patches.dtype)
