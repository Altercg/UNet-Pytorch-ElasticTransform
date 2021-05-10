'''
    弹性形变
'''

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torchvision.transforms as transforms
from PIL import Image


def elastic_transform(images, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described
       in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    image = images[0]
    label = images[1]
    if random_state is None:
        random_state = np.random.RandomState(None)  # 随机数生成器

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                      [center_square[0]+square_size,
                       center_square[1]-square_size],
                       center_square - square_size])
    # uniform(min,max) 随机数生成
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换 src原始图像的三个点坐标，dst仿射图像的这三个点坐标，M表示矩阵
    M = cv2.getAffineTransform(pts1, pts2)  
    image = cv2.warpAffine(image, M, shape_size[::-1],
                           borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, shape_size[::-1],
                           borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # dz = np.zeros_like(dx)

    # x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
    #                       np.arange(shape[2]))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)),
    #                       np.reshape(z, (-1, 1))

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), \
            map_coordinates(label, indices, order=1, mode='reflect').reshape(shape)


if __name__ == '__main__':
    transform = transforms.ToTensor()
    for i in range(30):
        img_path = 'U-net/dataset/train/image/'+str(i)+'.png'
        label_path = 'U-net/dataset/train/label/'+str(i)+'.png'
        img = Image.open(img_path)
        img = np.array(img)
        label = Image.open(label_path)
        label = np.array(label)
        imgs = [img, label]
        aug_img, aug_mask = elastic_transform(imgs, img.shape[1]*2, img.shape[1]*0.08, img.shape[1]*0.08)
        aug_img = Image.fromarray(aug_img)
        aug_mask = Image.fromarray(aug_mask)
        aug_img.save('U-net/dataset/train/image/'+str(i+30)+'.png')
        aug_mask.save('U-net/dataset/train/label/' + str(i+30) + '.png')
