#Auth:zhanglusheng@outlook.com
#Implementation of FastBERT, paper refer:https://arxiv.org/pdf/2004.02178.pdf
import argparse
import pickle as pkl
import numpy as np
import json
import logging

import torch
import torch.utils.data as data
from model_define.model_fastbert import FastBertModel, BertConfig
from data_utils import tokenization
from data_utils.dataset_preparing import PrepareDataset, TextCollate
from data_utils.dataset_preparing import read_from_tsv
from utils import load_json_config, load_saved_model

import time
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class ModelPredictor:
    def __init__(self, model_config_file, save_model_path, inference_speed, gpu_id):
        self.init_config(model_config_file)

        self.save_model_path = save_model_path
        self.max_seq_len = self.config.get('max_seq_len')
        self.inference_speed = inference_speed

        self.gpu_id = gpu_id
        self.use_cuda = gpu_id != -1

        self.init_model()
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.config.get("vocab_file"), do_lower_case=True)


    def init_config(self, config_file):
        logging.info("Loading HyperParameters".center(60, "="))
        self.config = load_json_config(config_file)
        logging.info(json.dumps(self.config, indent=2, sort_keys=True))
        logging.info("Load HyperParameters Done".center(60, "="))


    def init_model(self):
        bert_config = BertConfig.from_json_file(self.config.get("bert_config_path"))
        self.model = FastBertModel(bert_config, self.config)
        logging.info(self.model)
        logging.info("Initialize Model Done".center(60, "="))

        logging.info("Load saved model from: " + self.save_model_path)
        load_saved_model(self.model, self.save_model_path)
        logging.info("Load Saved Model Done".center(60, "="))

        if self.use_cuda:
            self.model = self.model.cuda(self.gpu_id) 

        self.model.eval()


    def preproc_text(self,  text):
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:(self.max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens 
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens))
        attn_masks  = [1] * (len(tokens))
        
        tokens = torch.LongTensor([tokens])
        segment_ids = torch.LongTensor([segment_ids])
        attn_masks = torch.LongTensor([attn_masks])
        return tokens, segment_ids, attn_masks


    def predict(self, text):
        tokens, segment_ids, attn_masks = self.preproc_text(text)
    
        if self.use_cuda:
            tokens = tokens.cuda(self.gpu_id) 
            segment_ids = segment_ids.cuda(self.gpu_id) 
            attn_masks = attn_masks.cuda(self.gpu_id) 
        with torch.no_grad():
            probs, layer_idxes, uncertain_infos = self.model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks,
                                            inference=True, inference_speed=self.inference_speed)
            top_probs, top_index = probs.topk(1)
            return probs.cpu().numpy()[0], top_index.cpu().numpy()[0]


if __name__=="__main__":
    model_config_file = './config/fastbert_cls_21.json' # './config/fastbert_cls.json'
    save_model_path =   'saved_model/fastbert_test_distill_21_08' #'saved_model/fastbert_test'
    inference_speed = 0.8
    gpu_id = 1

    vocab_file = "./pretrained_model/bert-chinese/vocab.txt"
    data_file = "./sample/project/dev.tsv"
    index = ["content","label"]
    test_data = read_from_tsv(data_file,index)

    model_predictor = ModelPredictor(model_config_file, save_model_path, inference_speed, gpu_id)

    acc = 0.0
    false = 0
    '''
    text = "是否认识借款人sep那小子保" # 8
    prob, pred = model_predictor.predict(text)
    print(pred, prob, text)
    '''




    random.shuffle(test_data)
    print(len(test_data))
    print(test_data[:10])  
    length = int(0.05*len(test_data))

    t0 = time.perf_counter()

    for item in range(length):
        text = test_data[item]["content"]
        y_label = int(test_data[item]["label"])
        prob, pred = model_predictor.predict(text)
        print(pred[0],y_label,prob)
        if pred[0] == y_label:
            acc += 1

    time_cost = time.perf_counter() - t0
    print(time_cost, "seconds process time",length)
    acc = acc / length #len(test_data)
    print("acc",acc,"seconds per time",time_cost/length)
    print("speed",save_model_path,inference_speed)
   


