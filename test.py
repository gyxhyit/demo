import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transform_image(image_path, shear_factor, scale_factor, rotation_angle):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return None, None

    # 获取图像的高度和宽度
    (height, width) = image.shape[:2]

    # 先旋转原始图像 180 度
    center = (width // 2, height // 2)
    M_rotation_orig = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated_original_image = cv2.warpAffine(image, M_rotation_orig, (width, height), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 1. 缩小图像
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    scaled_image = cv2.resize(rotated_original_image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # 2. 计算旋转矩阵
    center = (scaled_width // 2, scaled_height // 2)
    M_rotation = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 计算旋转后的图像尺寸
    abs_cos = abs(M_rotation[0, 0])
    abs_sin = abs(M_rotation[0, 1])
    new_width = int(scaled_height * abs_sin + scaled_width * abs_cos)
    new_height = int(scaled_height * abs_cos + scaled_width * abs_sin)

    # 更新旋转矩阵以考虑图像中心的偏移
    M_rotation[0, 2] += (new_width / 2) - center[0]
    M_rotation[1, 2] += (new_height / 2) - center[1]

    # 应用旋转变换，背景色设置为白色
    rotated_image = cv2.warpAffine(scaled_image, M_rotation, (new_width, new_height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 3. 计算剪切矩阵
    M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])

    # 应用剪切变换，背景色设置为白色
    sheared_image = cv2.warpAffine(rotated_image, M_shear,
                                   (new_width + int(abs(shear_factor) * new_height), new_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_original_image, sheared_image


def plot_3d_image(original_image, transformed_image):
    fig = plt.figure(figsize=(12, 6))

    # 原始图像
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    height, width = original_image.shape[:2]
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)

    # 归一化图像数据
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    r = original_image_rgb[:, :, 0] / 255.0
    g = original_image_rgb[:, :, 1] / 255.0
    b = original_image_rgb[:, :, 2] / 255.0

    # 绘制原始图像到三维空间
    ax1.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=np.stack((r, g, b), axis=-1), shade=False)
    ax1.set_title('Original Image in 3D')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')

    # 隐藏坐标轴数值
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=0)
    ax1.tick_params(axis='z', labelsize=0)

    # 变换后的图像
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    height, width = transformed_image.shape[:2]
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    # 在绘制图像时，将Z轴缩小
    z = np.zeros_like(x) + np.linspace(0, -0.2 * height, height)[:, np.newaxis]  # 使用更小的Z轴比例

    # 归一化图像数据并去除白色背景
    transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    r = transformed_image_rgb[:, :, 0] / 255.0
    g = transformed_image_rgb[:, :, 1] / 255.0
    b = transformed_image_rgb[:, :, 2] / 255.0

    # 创建透明度通道
    alpha_channel = np.ones_like(r)
    alpha_channel[np.all(transformed_image_rgb == [255, 255, 255], axis=-1)] = 0
    rgba_image = np.stack((r, g, b, alpha_channel), axis=-1)

    # 绘制变换后的图像到三维空间，并竖立起来
    ax2.plot_surface(x, z, y, rstride=1, cstride=1, facecolors=rgba_image, shade=False)
    ax2.set_title('Transformed Image in 3D (Vertical)')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_zlabel('')

    # 隐藏坐标轴数值
    ax2.tick_params(axis='x', labelsize=0)
    ax2.tick_params(axis='y', labelsize=0)
    ax2.tick_params(axis='z', labelsize=0)

    plt.show()


# 示例用法
image_path = 'C:/Users/admin/Desktop/321.jpg'
shear_factor = 0.5  # 剪切因子
scale_factor = 0.8  # 缩小因子
rotation_angle = 0  # 平面旋转角度

# 旋转原图180度并进行形变
original_image, transformed_image = transform_image(image_path, shear_factor, scale_factor, rotation_angle)

if original_image is not None and transformed_image is not None:
    plot_3d_image(original_image, transformed_image)
