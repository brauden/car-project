import pickle
import pandas as pd
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from torchvision.io import read_image

# Constants
DATA_PATH = Path("./data")
ANNOTATION_TRAIN = DATA_PATH / "cars_train_annos.mat"
TRAIN_PATH = DATA_PATH / "cars_train"


def annotations_preprocess(mat, path) -> pd.DataFrame:
    annotation = pd.DataFrame(mat["annotations"][0][['fname', 'class']])
    annotation['fname'] = annotation['fname'].apply(lambda x: x[0]).apply(lambda x: str(path / x))
    annotation['class'] = annotation['class'].astype(np.int16) - 1
    return annotation


if __name__ == '__main__':
    classes_data = [x[0] for x in loadmat(str(DATA_PATH / "devkit/cars_meta.mat"))["class_names"][0]]
    classes_data = {k: v for k, v in enumerate(classes_data)}

    with open(str(DATA_PATH / "classes_dict.pkl"), "wb") as f:
        pickle.dump(classes_data, f, pickle.HIGHEST_PROTOCOL)

    annotation_train = annotations_preprocess(loadmat(ANNOTATION_TRAIN), TRAIN_PATH)
    shapes = [read_image(p).shape for p in annotation_train['fname']]
    annotation_train['shapes'] = shapes
    annotation_train['shapes'] = annotation_train.shapes.apply(lambda x: x[0])
    annotation_train = annotation_train[annotation_train['shapes'] == 3]
    annotation_train.drop("shapes", axis=1, inplace=True)
    annotation_train.to_parquet(str(DATA_PATH / "annotation.parquet"))
