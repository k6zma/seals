import torch
import clip

from PIL import Image

from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

def imgs_to_emb(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    class_name_train = []
    class_name_valid = []

    image_features = []
    image_names = []
    for img in tqdm(glob(img_path)):
        if 'train' in img_path:
            if 'train/seal' in img_path:
                image_names.append(img[19:])
                class_name_train.append(1)
            elif 'train/no_seal' in img_path:
                image_names.append(img[22:])
                class_name_train.append(0)
        elif 'valid' in img_path:
            if 'valid/seal' in img_path:
                image_names.append(img[19:])
                class_name_valid.append(1)
            elif 'valid/no_seal' in img_path:
                image_names.append(img[22:])
                class_name_valid.append(0)
        elif 'test' in img_path:
            image_names.append(img[13:])

        img = np.array(Image.open(img))
        img = img[:img.shape[0]//2]
        image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features.append(model.encode_image(image)[0].tolist())

    df = pd.DataFrame(np.array(image_features))
    df['Image name'] = image_names

    if 'train' in img_path:
        df['Target'] = class_name_train
    elif 'valid' in img_path:
        df['Target'] = class_name_valid

    return df