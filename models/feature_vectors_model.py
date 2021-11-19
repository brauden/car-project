import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.io import read_image
from glob import glob
from tqdm import tqdm


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_layers = 524
        self.resnet = resnet18()
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layers),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_layers, 196)
        )

    def forward(self, x):
        return self.resnet(x)


model = Model.load_from_checkpoint("../data/model/trained_model_2021-11-12 19:41:59.859689.ckpt")


class FeatureExtraction(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = model
        self.children_list = []
        for n, c in self.pretrained._modules['resnet'].named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x


f_model = FeatureExtraction("avgpool")
f_model.eval()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.)
    ])
    img = read_image("../data/cars_train/00001.jpg")
    img = transform(img)
    p1 = f_model(img.unsqueeze(0))
    assert p1.shape == torch.Size([1, 512, 1, 1])
    PATH = "../data/cars_test/*.jpg"
    all_images = glob(PATH)
    print("Num of images: ", len(all_images))
    images_list = []
    images_name = []
    for image in tqdm(all_images[:300]):
        img = read_image(image)
        img = transform(img)
        if img.shape == torch.Size([3, 256, 256]):
            p = f_model(img.unsqueeze(0)).squeeze(-1).squeeze(-1).detach().numpy()
            images_list.append(p)
            images_name.append(image)
    images_list = np.concatenate(images_list, axis=0)
    images_name = np.array(images_name)
    print(images_list.shape)

    with open("../data/test_vectors.npy", "wb") as f:
        np.save(f, images_list)

    with open("../data/test_names.npy", "wb") as f:
        np.save(f, images_name)
