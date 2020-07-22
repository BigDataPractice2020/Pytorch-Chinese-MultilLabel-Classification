import pandas as pd 
import numpy as np
import copy 

frame = pd.read_table('train.tsv')

data =copy.deepcopy(frame.values)
print(type(data[1,0]))

for i in range(frame.shape[0]):
    if (frame.label[i])//3 == 0 :
        frame.content[i] = str("身份确认<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 1 :
        frame.content[i] = str("一次确认<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 2 :
        frame.content[i] = str("是否认识借款人<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 3 :
        frame.content[i] = str("今日能否还款<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 4 :
        frame.content[i] = str("二次确认<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 5 :
        frame.content[i] = str("请求转告<SEP>")+ str(data[i,0])
    elif (frame.label[i])//3 == 6 :
        frame.content[i] = str("明日能否还款<SEP>")+ str(data[i,0])



frame.to_csv("train.csv",sep=',',header=False,index=False)


