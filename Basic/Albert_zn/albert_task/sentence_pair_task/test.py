# -*- coding: utf-8 -*-
import json

from predict import Predictor


with open("/home/wangkehan/bert-for-task/albert_task/sentence_pair_task/config/lcqmc_config.json", "r") as fr:
    config = json.load(fr)


predictor = Predictor(config)

text_a = "询问是否认识借款人"
text_b = "不知道"
res = predictor.predict(text_a, text_b)
print(res)