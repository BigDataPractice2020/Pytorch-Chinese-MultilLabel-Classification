import csv
import pandas as pd
import numpy as np
import string
#import matplotlib.pyplot as plt


trains = ["/home/wangkehan/home/data_20000_node/ask_know_train.csv","/home/wangkehan/home/data_20000_node/ask_today_train.csv",
"/home/wangkehan/home/data_20000_node/ask_tomorrow_train.csv","/home/wangkehan/home/data_20000_node/identity1_train.csv",
"/home/wangkehan/home/data_20000_node/once1_train.csv","/home/wangkehan/home/data_20000_node/request_train.csv",
"/home/wangkehan/home/data_20000_node/twice_train.csv"]

tests=["/home/wangkehan/home/data_20000_node/ask_know_test.csv","/home/wangkehan/home/data_20000_node/ask_today_test.csv",
"/home/wangkehan/home/data_20000_node/ask_tomorrow_test.csv","/home/wangkehan/home/data_20000_node/identity1_test.csv",
"/home/wangkehan/home/data_20000_node/once1_test.csv","/home/wangkehan/home/data_20000_node/request_test.csv",
"/home/wangkehan/home/data_20000_node/twice_test.csv"]


def processor(lists):
    for file_name in lists:
        d = pd.read_csv(file_name, usecols=['type_robot', 'msg', 'type_combine'])

        d['type_number'] = ''
        '''
        Length = []
        max_length = 0
        max_i = 0
        '''
        for i in range(d.shape[0]):
            d.msg[i] = d.msg[i].strip(string.punctuation)
            '''
            length_i = len(d.msg[i])
            Length.append(length_i)
            if length_i > max_length:
                max_length = length_i
                max_i = i
            '''
            if d.type_combine[i] == 'invalid':
                d.type_number[i] = 0
            elif d.type_combine[i] == 'yes':
                d.type_number[i] = 1
            else:
                d.type_number[i] = 2
        del d['type_combine']
        print(d)
        #d.to_csv('test' + '.txt', sep='\t', header=None, index=False, mode='a+')
        d.iloc[:int(d.shape[0]*0.8)].to_csv('train' + '.txt', sep='\t', header=None, index=False, mode='a+')
        d.iloc[int(d.shape[0]*0.8):].to_csv('dev' + '.txt', sep='\t', header=None, index=False, mode='a+')

#processor(tests)
processor(trains)





