import os.path
import time

import numpy as np
from src.unet import UNet
from PIL import Image
from torchvision import transforms

import torch.cuda


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # 数据，模型
    classes = 1
    weight_path = "save_weights/best_model.pth"
    img_path = r"D:\Code_python\deeplearn_data\DRIVE\test\images\01_test.tif"
    roi_mask_path = r"D:\Code_python\deeplearn_data\DRIVE\test\mask\01_test_mask.gif"
    assert os.path.exists(weight_path), f"{weight_path} not found"
    assert os.path.exists(img_path), f"{img_path} not found"
    assert os.path.exists(roi_mask_path), f"{roi_mask_path} not found"

    device = torch.device("cuda" if torch.cuda.is_available() else None)
    model = UNet(base_c=3, num_classes=classes + 1)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    print("using {} device".format(device))
    model.to(device)

    # load roi mask
    roi_image = Image.open(roi_mask_path).convert('L')
    roi_image = np.array(roi_image)

    # load img
    original_img = Image.open(img_path).convert('RGB')

    # data transformer
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 进入验证模式
    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print(f"inference time:{t_end-t_start}")

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to('cpu').numpy().astype(np.uint8)

        #  将前景对应的像素改为255
        prediction[prediction == 1] = 255
        #  将roi_mask 对应的区域像素改为0
        prediction[roi_image == 0] = 0
        result = Image.fromarray(prediction)
        result.save('test_result.png')


if __name__ == '__main__':
    main()
