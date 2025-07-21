import os
import token

import torch.cuda
import transforms as T
from my_dataset import VOCSegmentation
from src.deeplabv3_model import deeplabv3_resnet50
from train_utils import evaluate

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform():
    base_size = 520
    return SegmentationPresetEval(base_size)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found"

    # segmentation num_classes+background
    num_classes = args.num_classes + 1
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(),
                                  txt_name="val.txt")
    num_workers = 8

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 pin_memory=True,
                                                 num_workers=num_workers,
                                                 collate_fn=val_dataset.collate_fn)


    model=deeplabv3_resnet50(aux=True,num_classes=num_classes,pretrain_backbone=False)
    model.load_state_dict(torch.load(args.weights,map_location=device)['model'])
    model.to(device)

    comfmat=evaluate(model,val_dataloader,device=device,num_classes=num_classes)
    print(comfmat)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="", help="")
    parser.add_argument("--weights", default="./save_weights/model_1.pth")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    return parser


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")
    main(args)
