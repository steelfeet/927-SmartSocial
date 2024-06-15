from turtle import title
from flask import render_template, request, redirect, flash, jsonify, Blueprint

from werkzeug.utils import secure_filename

from app import app

import os, sys, random, config
import csv, json, shutil
import glob
import requests

from ultralytics import YOLO, checks, hub
import base64

import cv2

from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os

SmartSocial927 = Blueprint('SmartSocial927', __name__)


from sentence_transformers import SentenceTransformer
from PIL import Image

model_name = 'clip-ViT-B-16'
st_model = SentenceTransformer(model_name)

# YaGPT, https://habr.com/ru/articles/780008/
service_acc = "aje0d6tpeavoi7p3f89a"
service_acc = "b1gqcld9kshv6apg0b3j"
api_key = "AQVNzjYBxggiibjSMmNAz256c8FOv1hDTlVhLGvT"


from nltk.corpus import stopwords
import pymorphy3

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
from string import punctuation


# функция очистки слов и лемматизации текста
# antispam
def preprocess_text(text):
	words = text.lower().split()
	
	# очистка от прилегающего к слову мусора из пунктуации (слово, -> слово / "или так" -> или так / круто!! -> круто)
	clear_words = []
	for word in words:
		clear_word = ""
		for s in word:
			if not s in punctuation:
				clear_word = clear_word + s
		clear_words.append(clear_word)
	
	# лемматизация, бомбу -> бомба 
	tokens = [morph.parse(token)[0].normal_form for token in clear_words if token not in russian_stopwords\
			and token != " " \
			and token.strip() not in punctuation]

	text = " ".join(tokens)
	
	return tokens

# функция сортировки 
def order_of_item(item):
    return int(item["stat"])



def get_df(df_path: str) -> pd.DataFrame:
    data_df = pd.read_json(df_path)
    data_df['Embedding'] = data_df['Embedding'].apply(lambda x: np.array(x))
    return data_df

def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
    result_distance = spatial.distance.cosine(emb_a, emb_b)
    return result_distance

def found_similar_images(input_img_path: str, images_db: pd.DataFrame, n: int=1) -> pd.DataFrame:
    input_vec = vectorize_img(input_img_path)
    result_df = copy.deepcopy(images_db)
    result_df['Distance_with_input'] = result_df.apply(lambda x: calculate_cos_dist(input_vec, x['Embedding']), axis=1)
    result_df_sorted =    result_df.sort_values('Distance_with_input').reset_index()
    return result_df_sorted.head(n)

def vectorize_img(img_path, model=st_model):
    img = Image.open(img_path)
    return model.encode(img)


########### Views:
@app.route('/927-smartsocial', methods=['post', 'get'])
def smartsocial927():
    DATA_DIR = "D:\\_data\\927-10Jun-SmartSocial\\mincult-train"
    IMAGES_DIR = os.path.join(DATA_DIR, "train")

    # очищаем config.UPLOAD_DIR
    files = glob.glob('config.UPLOAD_DIR')
    for f in files:
        os.remove(f)

    resp = {}
    files = request.files.to_dict()
    for file_key in files:
        file = files[file_key]
        # безопасно извлекаем оригинальное имя файла
        filename = secure_filename(file.filename)
        basename, file_extension = os.path.splitext(filename)
        # случайное число
        upload_basename = random.randrange(10, 10000)
        upload_filename = f"{upload_basename}{file_extension}"
        # сохраняем файл
        upload_path = os.path.join(config.UPLOAD_DIR, upload_filename)
        file.save(upload_path)
        
        if file_extension == ".jpg" or file_extension == ".png":
            result_df_path = os.path.join(DATA_DIR, "result.json")
            
            with open(upload_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            base64_string = encoded_string.decode('utf-8')
            resp["source_file"] = base64_string


            images_db = get_df(result_df_path)
            print("start_found")
            result_df = found_similar_images(upload_path, images_db, 10)
            
            resp["similar_files"] = []
            
            summ_desc = ""
            for i in range(10):
                resp_item = {}
                result_row = result_df.iloc[i]
                print(result_row["object_id"], result_row["name"], result_row["description"], result_row["group"], result_row["img_name"], result_row["Distance_with_input"])
                print()

                image_path = os.path.join(IMAGES_DIR, str(result_row["object_id"]), result_row["img_name"])
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                base64_string = encoded_string.decode('utf-8')    

                summ_desc += result_row["name"]
                if len(str(result_row["description"])) > 5:
                    summ_desc += result_row["description"]

                resp_item["name"] = result_row["name"]
                resp_item["description"] = result_row["description"]
                resp_item["group"] = result_row["group"]
                resp_item["Distance_with_input"] = result_row["Distance_with_input"]
                resp_item["url"] = f"https://goskatalog.ru/portal/#/collections?id={result_row['object_id']}"
                resp_item["img"] = base64_string

                resp["similar_files"].append(resp_item)


    prompt = {
    "modelUri": f"gpt://{service_acc}/yandexgpt-lite",
    "completionOptions": {
        "stream": False,
        "temperature": 0.6,
        "maxTokens": "2000"
    },
    "messages": [
        {
            "role": "system",
            "text": "Ты сотрудник музея, из нескольких описаний похожих предметов составь для одного предмета из трех предложений. "
        },
        {
            "role": "user",
            "text": "Прямоугольной формы желтого цвета. Состоит из двух частей наружней и внутренней. На наружней части вверху наклеена этикетка прямоугольной формы белого цвета на которой изображен девичий костюм Воронежской Губернии. Прямоугольной формы желтого цвета. Состоит из двух частей наружней и внутренней. На наружней части вверху наклеена этикетка прямоугольной формы белого цвета на которой изображена иллюстрация к произведению М. Шолохова Тихий Дон. "
        },
        {
            "role": "assistant",
            "text": "Прямоугольной формы желтого цвета. Состоит из двух частей наружней и внутренней. На наружней части вверху наклеена этикетка прямоугольной формы белого цвета."
        },
        {
            "role": "user",
            "text": f"{summ_desc}"
        }]
    }


    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}"
    }


    import json

    descs = []
    response = requests.post(url, headers=headers, json=prompt)
    result_dict = json.loads(response.text)
    try:
        referat = result_dict["result"]["alternatives"][0]["message"]["text"]
    except:
        referat = ""
    resp["referat_1"] = referat
    descs.append(referat)
    print()
    print(referat)
            
    response = requests.post(url, headers=headers, json=prompt)
    result_dict = json.loads(response.text)
    try:
        referat = result_dict["result"]["alternatives"][0]["message"]["text"]
    except:
        referat = ""
    resp["referat_2"] = referat
    descs.append(referat)
    print()
    print(referat)

    response = requests.post(url, headers=headers, json=prompt)
    result_dict = json.loads(response.text)
    try:
        referat = result_dict["result"]["alternatives"][0]["message"]["text"]
    except:
        referat = ""
    resp["referat_3"] = referat
    descs.append(referat)
    print()
    print(referat)




    # суммаризируем предложенные варианты
    sentences = []
    id = 0
    for desc in descs:
        # разбиваем на предложения
        sentence = str(desc).split(".")

        for sent in sentence:
            sentence_item = {}
            id += 1
            sentence_item["id"] = id
            sentence_item["stat"] = 0
            sentence_item["text"] = sent
            sentence_item["tokens"] = preprocess_text(sent)

            if len(sentence_item["tokens"]) > 0:
                sentences.append(sentence_item)

    # считаем статистику каждого токена в предложении и общую
    stat_sentences = []
    for sent_1 in sentences:
        current_stat = 0
        for sent_2 in sentences:
            if sent_1["id"] != sent_2["id"]:
                tokens_1 = sent_1["tokens"]
                tokens_2 = sent_1["tokens"]

                for token_1 in tokens_1:
                    if token_1 in tokens_2:
                        current_stat += 1

        sent_1["stat"] = current_stat
        stat_sentences.append(sent_1)

    sort_sentences = sorted(stat_sentences, key=order_of_item, reverse=True)

    summ_referat = ""
    for sort_sentence in sort_sentences[:5]:
        summ_referat += sort_sentence["text"] + ". "
    resp["summ_referat"] = summ_referat

    return jsonify(resp)