import os
import json
import torch
from PIL import Image
import matplotlib
from torchvision import transforms
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用非交互式后端
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_path = r"D:\Code_python\deeplearn_data\image_data\flower_photos\tulips\10791227_7168491604.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    img=data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # load model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = './AlexNet.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path,weights_only=True))

    model.eval()
    with torch.no_grad():
        outputs = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(outputs, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
