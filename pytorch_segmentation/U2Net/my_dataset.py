import os.path

import cv2
import torch.utils.data as data
from PIL import Image


class DUTSDataset(data.Dataset):
    def __init__(self, duts_root: str, txt_name: str = "DUTS-TR", transforms=None):
        assert os.path.exists(duts_root), f" path {duts_root} not exist."
        root = os.path.join(duts_root, txt_name)
        assert os.path.exists(root), f" path {root} not exist."
        img_path = os.path.join(root, txt_name + "-Image")
        mask_path = os.path.join(root, txt_name + "-Mask")
        self.img_list = [os.path.join(img_path, p) for p in os.listdir(img_path)]
        self.mask_list = [os.path.join(mask_path, p) for p in os.listdir(mask_path)]
        self.transforms = transforms

    # def __getitem__(self, index):
    #     img = Image.open(self.img_list[index]).convert('RGB')
    #     mask = Image.open(self.mask_list[index]).convert('L')
    #
    #     if self.transforms is not None:
    #         img, mask = self.transforms(img, mask)
    #
    #     return img, mask

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape
        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)

        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        img, mask = list(zip(*batch))
        batched_img = cat_list(img, fill_value=0)
        batched_mask = cat_list(mask, fill_value=0)
        return batched_img, batched_mask


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = DUTSDataset("D:\Code_python\deeplearn_data\DUTS", "DUTS-TR")
    print(len(train_dataset))
    val_dataset = DUTSDataset("D:\Code_python\deeplearn_data\DUTS", "DUTS-TE")
    print(len(val_dataset))

    i, t = train_dataset[0]

# data = DUTSDataset("D:\Code_python\deeplearn_data\DUTS", "DUTS-TR")
# duts_root:  D:\Code_python\deeplearn_data\DUTS
# root: D:\Code_python\deeplearn_data\DUTS\DUTS-TR ,  D:\Code_python\deeplearn_data\DUTS\DUTS-TE
# img:  D:\Code_python\deeplearn_data\DUTS\DUTS-TR\DUTS-TR-Image
# mask: D:\Code_python\deeplearn_data\DUTS\DUTS-TR\DUTS-TR-Mask
