import os.path
from PIL import Image

import numpy as np


def main():
    img_channel = 3
    img_dir = r"D:\Code_python\deeplearn_data\DRIVE\training\images"
    roi_dir = r"D:\Code_python\deeplearn_data\DRIVE\training\mask"
    assert os.path.exists(img_dir), f"{img_dir} does not exist"
    assert os.path.exists(roi_dir), f"{roi_dir} does not exist"

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith("tif")]
    cumulative_mean = np.zeros(img_channel)
    cumulative_std = np.zeros(img_channel)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        roi_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(roi_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)
    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean:{mean}")
    print(f"std:{std}")


if __name__ == '__main__':
    main()
