# Models

![Models](../Images/Models.svg)

## Description
1. Fine-tuning ResNet18 model using PyTorch Lightning using [Standford dataset](https://www.kaggle.com/jessicali9530/stanford-cars-dataset) from Kaggle: car_class_model.py
2. Extracting features from the images: feature_vectors_model.py
3. Training Nearest neighbors model to find similarities: similarity_model.py

## Car class model CLI
```bash
python car_class_model.py <num_epochs> [learning_rate] [batch_size]
```
Example:
```bash
python car_class_model.py 5 --lr 0.001 --bs 16
```
