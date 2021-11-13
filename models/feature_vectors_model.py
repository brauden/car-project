import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.io import read_image


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

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.)
    ])
    img = read_image("../data/cars_train/00001.jpg")
    img = transform(img)
    assert f_model(img.unsqueeze(0)).shape == torch.Size([1, 512, 1, 1])
    torch.save(f_model, "../data/model/feature_extraction.pth")
