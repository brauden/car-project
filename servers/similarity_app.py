import os
import io
import pickle
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from models.feature_vectors_model import FeatureExtraction
from torchvision import transforms
from PIL import Image


PATH = "../data"
app = FastAPI()

with open(os.path.join(PATH, "similarity_model.pickle"), "rb") as m:
    similarity_model = pickle.load(m)

with open(os.path.join(PATH, "test_vectors.npy"), "rb") as f:
    vectors = np.load(f)

with open(os.path.join(PATH, "test_names.npy"), "rb") as f:
    names = np.load(f)

feature_extraction = FeatureExtraction("avgpool")
feature_extraction.eval()
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])


def transform_to_vec(img):
    image = transform(img)
    vec = feature_extraction(image.unsqueeze(0))
    vec = vec.squeeze(-1).squeeze(-1).detach().numpy()
    return vec


@app.get('/')
async def root():
    return "Use post method using /predict or get method with /docs"


@app.post('/predict')
async def prediction(img: UploadFile = File(...)):
    raw_file = img.file.read()
    raw_file = Image.open(io.BytesIO(raw_file))
    transformed_img = transform_to_vec(raw_file)
    distance, index = similarity_model.kneighbors(transformed_img.reshape(1, -1))
    result = {}
    for i in range(distance[0].shape[0]):
        result[names[index[0][i]]] = float(distance[0][i])
    return {"File names": result}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8234)
