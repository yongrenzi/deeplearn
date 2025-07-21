import os.path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DriveDataset(Dataset):
    def __init__(self, root, transforms, train=True):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path {data_root} is not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif") for i in img_names]
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif") for i in img_names]
        # ”D:\Code_python\deeplearn_data\DRIVE\training\images\21_training.tif“
        # ”D:\Code_python\deeplearn_data\DRIVE\training\1st_manual\21_manual1.gif“
        # "D:\Code_python\deeplearn_data\DRIVE\training\mask\21_training_mask.gif"
        # check file

        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file{i} is not found")
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file{i} is not found")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 转为PIL的原因是，Transforms中是对PIL数据进行处理的
        mask = Image.fromarray(mask)
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_img = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_img,batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



# train_dataset = DriveDataset(root=r"D:\Code_python\deeplearn_data",
#                              transforms=None,
#                              train=True)
# print(len(train_dataset))