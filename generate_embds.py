# -*- encoding: utf-8 -*-
# 
import os, urllib, sys
from os.path import dirname, abspath

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import random, time
from datetime import datetime
import requests
from requests.exceptions import ProxyError

import time
import hashlib

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(1, BASE_DIR)
APP_DIR = dirname(BASE_DIR)
sys.path.append(APP_DIR)

import func # m-dv funcs

BASE_DIR = abspath(__file__)

DATA_DIR = "D:\\_data\\927-10Jun-SmartSocial\\mincult-train"
IMAGES_DIR = os.path.join(DATA_DIR, "train")
EMBDS_DIR = os.path.join(DATA_DIR, "embds")


from sentence_transformers import SentenceTransformer
from PIL import Image

model_name = 'clip-ViT-B-16'
st_model = SentenceTransformer(model_name)
def vectorize_img(img_path, model=st_model):
    img = Image.open(img_path)
    return st_model.encode(img)


# функция построения эмбеддингов
def create_images_db(object_id, img_name, model=st_model):
    image_path = os.path.join(IMAGES_DIR, str(object_id), str(img_name))
    if os.path.isfile(image_path):
        emb = vectorize_img(image_path, model)
    return emb


data_csv_path = os.path.join(DATA_DIR, "train.csv")
result_df_path = os.path.join(DATA_DIR, "result.json")

data_df = pd.read_csv(data_csv_path, sep=";", )
print(data_df.head())

tqdm.pandas()
data_df["Embedding"] = data_df.progress_apply(lambda x: create_images_db(x['object_id'], x['img_name']), axis=1)


with open(result_df_path, 'w', encoding='utf-8') as file:
    data_df.to_json(file, force_ascii=False)