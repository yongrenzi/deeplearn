import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def display_images(img_path, roi_path):
    # 加载并处理原始图像
    img = np.array(Image.open(img_path)) / 255.0  # 归一化到[0,1]

    # 加载并处理ROI掩码
    roi_img = np.array(Image.open(roi_path).convert('L'))  # 转换为灰度

    # 创建画布
    plt.figure(figsize=(12, 6))

    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 显示ROI掩码
    plt.subplot(1, 3, 2)
    plt.imshow(roi_img, cmap='gray')
    plt.title('ROI Mask')
    plt.axis('off')

    # 显示应用ROI后的图像
    plt.subplot(1, 3, 3)
    # 创建ROI掩码（255为有效区域）
    roi_mask = roi_img == 255
    # 应用掩码（非ROI区域设为黑色）
    masked_img = img.copy()
    masked_img[~roi_mask] = 0  # 非ROI区域置零
    plt.imshow(masked_img)
    plt.title('Masked Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 替换为实际路径
    img_path = r"D:\Code_python\deeplearn_data\DRIVE\training\images\22_training.tif"
    roi_path = r"D:\Code_python\deeplearn_data\DRIVE\training\mask\22_training_mask.gif"

    display_images(img_path, roi_path)