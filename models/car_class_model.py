import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
import pytorch_lightning as pl
import pandas as pd

from torchvision.models import resnet50
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.img_df = df
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, item):
        img_path = self.img_df.loc[item, "fname"]
        image = read_image(img_path)
        label = self.img_df.loc[item, "class"]
        if self.transform:
            image = self.transform(image)
        return image, label


class CarClassification(pl.LightningModule):
    def __init__(self, num_of_classes,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 lr=1e-3, transfer=True,
                 tune_fc=True, hidden_layers=524,
                 optimizer=optim.Adam, batch_size=32):
        super(CarClassification, self).__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.resnet = resnet50(pretrained=transfer)
        self.in_features = self.resnet.fc.in_features
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        if tune_fc:
            for name, param in self.resnet.named_parameters():
                if "bn" not in name:
                    param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.in_features, hidden_layers),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_layers, num_of_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.Lambda(lambda x: x / 255.)
        ])
        img_train = CarDataset(self.train_df, transform=transform)
        return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        preds = self(x)
        loss = self.criterion(preds, y)
        max_vals, argmax = preds.max(-1)
        accuracy_metrics = tm.Accuracy()
        accuracy = accuracy_metrics(argmax, y).to(self.device)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def valid_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x / 255.)
        ])
        img_val = CarDataset(self.valid_df, transform=transform)
        return DataLoader(img_val, batch_size=1, shuffle=False)

    def valid_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        preds = self(x)
        max_vals, argmax = preds.max(-1)
        loss = self.criterion(preds, y)
        accuracy_metrics = tm.Accuracy()
        accuracy = accuracy_metrics(argmax, y).to(self.device)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == '__main__':
    annotation_df = pd.read_parquet("../data/annotation.parquet")
    nr_of_classes = annotation_df['class'].nunique()
    train_df, valid_df = train_test_split(annotation_df,
                                          test_size=0.2,
                                          random_state=123,
                                          shuffle=True,
                                          stratify=annotation_df['class'])
    train_df.reset_index(inplace=True)
    valid_df.reset_index(inplace=True)
    train_df.drop("index", axis=1, inplace=True)
    valid_df.drop("index", axis=1, inplace=True)
    model = CarClassification(nr_of_classes, train_df=train_df, valid_df=valid_df)
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
    trainer.fit(model)
