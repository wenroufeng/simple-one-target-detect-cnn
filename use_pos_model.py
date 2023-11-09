import os
import torch
from PIL import Image
import torchvision
from torch import nn
from train_pos import Residual, ResNet
import fiftyone as fo
import random
from shape_map import cls_map

reverse_cls_map = dict(map(lambda x: (x[1], x[0]), cls_map.items()))

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((300, 300)),
                                            torchvision.transforms.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load("models/model_49.pth", map_location=torch.device(device))

circles = [f'data/circles/{i}' for i in os.listdir('data/circles')[-100: ]]
triangles = [f'data/triangles/{i}' for i in os.listdir('data/triangles')[-100: ]]

data = circles + triangles
random.shuffle(data)

dataset = fo.Dataset()
for im in data:
    dataset.add_sample(fo.Sample(im))

with torch.no_grad():
    j = 0
    for sample in dataset.iter_samples(autosave=True):
        try:
            image = Image.open(sample.filepath)
            image = image.convert('RGB')
            image = transform(image)
            image = torch.reshape(image, (1, 3, 300, 300)).to(device)
        except Exception as e:
            continue
        model.eval()
        output = model(image)
        left = output[0].argmax(1)
        top = output[1].argmax(1)
        width = (output[2].argmax(1) - left) * 15 / 300
        height = (output[3].argmax(1) - top) * 15 / 300
        left = left * 15 / 300
        top = top * 15 / 300
        sign = output[4].argmax(1).item()
        # print('***********out: ', output.softmax(dim=1))
        detections = [fo.Detection(label=reverse_cls_map[sign], bounding_box=[left, top, width, height])]
        sample['detect'] = fo.Detections(detections=detections)

session = fo.launch_app(dataset, address="10.158.99.23", port=8000)
session.wait()


