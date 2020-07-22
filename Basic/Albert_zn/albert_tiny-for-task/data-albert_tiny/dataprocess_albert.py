# coding: UTF-8
# Preprocessor for train and test files to desired formats

import csv
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import math

def dataprocess_single1():
    file_name = "./data/csvdata/ask_tomorrow_test"

    #d = pd.read_csv(file_name + '.csv' ,encoding='utf8',usecols=['type_robot', 'msg', 'type_combine'])
    d = pd.read_csv(file_name + '.csv', encoding='utf8', usecols=['msg', 'type_combine'])
    d['type_number'] = ''

    Length = []
    max_length = 0
    max_i = 0

    for i in range((d.shape[0])):
        d.msg[i] = d.msg[i].strip(string.punctuation)
        length_i = len(d.msg[i])
        Length.append(length_i)
        if length_i > max_length:
            max_length = length_i
            max_i = i
        if d.type_combine[i] == 'invalid':
            d.type_number[i] = 18
        elif d.type_combine[i] == 'yes':
            d.type_number[i] = 19
        else:
            d.type_number[i] = 20

    merge_d = d.drop(columns=['type_combine'])
    merge_d_new = d.drop(columns=['type_combine', 'type_number'])

    for i in range(merge_d.shape[0]):
        #merge_d_new.type_robot[i] = str(merge_d.type_robot[i]) + '<SEP>' + str(merge_d.msg[i]) \
        #                            + '<SEP>' + str(merge_d.type_number[i])
        merge_d_new.msg[i] = str(merge_d.msg[i]) + '<SEP>' + str(merge_d.type_number[i])

    merge_d_new.to_csv(file_name + '.txt', sep='\t', header=None, index=False)

def txtcombine():
    # 合并一个文件夹下的多个txt文件
    # coding=utf-8
    import os
    # 获取目标文件夹的路径
    filedir = os.getcwd() + '/data/txttest'
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(filedir)
    # 打开当前目录下的result.txt文件，如果没有则创建
    f = open('test_result.txt', 'w')
    i = 0
    # 先遍历文件名
    for filename in filenames:
        i += 1
        print(i)
        if i > 0:
            filepath = filedir + '\\' + filename
            print(filepath)
            # 遍历单个文件，读取行数
            for line in open(filepath,encoding='utf-8',errors='ignore'):
                # print(str(line))
                f.writelines(line)
                # f.write('\n')
    # 关闭文件
    f.close()

def train_split():
    file_name = "./data/datanew/train_result.txt"
    lines = []

    with open(file_name, 'r',encoding='utf8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    lines = np.array(lines)

    np.random.seed(10)
    a = np.random.randn(lines.shape[0])  # 正态分布
    b = np.argsort(a)
    line_new = lines[b]

    train_n = math.ceil(0.99 * lines.shape[0])
    train = line_new[:train_n]
    valid = line_new[train_n:]
    np.savetxt("data/datanew/train_all_1.txt",train,fmt='%s')
    np.savetxt("data/datanew/valid_all_1.txt", valid,fmt='%s')



#dataprocess_single1()
#txtcombine()
train_split()





'''
train_n = math.ceil(0.8*merge_d_new.shape[0])
train = merge_d_new.iloc[:train_n]
valid = merge_d_new.iloc[train_n:]
print(train.shape)
print(valid.shape)
#train_train.to_csv(file_name+'.txt', sep='\t',header=None,index=False)
train.to_csv('./data/train.txt', sep='\t',header=None,index=False)
valid.to_csv('./data/valid.txt', sep='\t',header=None,index=False)


print(Length)
Length = sorted(Length)
category = list(merge_d.drop(columns=['msg']).values[:,0])


def plot_distribution(data,plot_type):

    #统计出现的元素有哪些
    unique_data = np.unique(data)
    print(unique_data)
    #统计某个元素出现的次数
    resdata = []
    for ii in unique_data:
        resdata.append(data.count(ii))
    print(resdata)

    #fig = plt.figure()
    if plot_type == 0:
        plt.bar(unique_data, resdata)
        plt.title('Sentence length distribution')
        plt.xlabel('length')
        plt.ylabel('times')
    elif plot_type == 1:
        plt.pie(resdata,labels=unique_data)
        plt.title('Labels frequency distribution')
    plt.show()

fig1 = plt.figure()
plot_distribution(Length,0)
fig2 = plt.figure()
plot_distribution(category,1)
'''