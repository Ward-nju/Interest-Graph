# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import pandas as pd
import json

word_index=pd.read_csv('/home/hungfei/桌面/paperdata/word_index.csv',header=None,names=['index','word'])
num_seq=11

filename="/home/hungfei/桌面/paperdata/result/lda-seq/topic-002-var-e-log-prob.dat"
f=open(filename,'r')
x=[]
for line in f.readlines():
    x.append(float(line.strip()))
f.close()
x=np.array(x)
x=x.reshape(len(x)/num_seq,num_seq)
df=pd.DataFrame(x)

words=[]
#按照每个时间列排序，获取该主题在该时间下的top10词语的index
for i in range(0,num_seq):
    words.append(list(df.sort_values(by=i,ascending=False).index)[:10])
    
result=[]
for i in range(num_seq):
    tmp=[]
    for j in range(10):
        index=words[i][j]
        word=word_index[word_index['index']==index]['word'].values[0]
        prob="%.5f"% np.exp(x[index,i])
        tmp.append((word,prob))
    result.append([i+2004,tmp])
    
topic_dict={}
topic_dict['tid']='002'
topic_dict['topics']=result
fo = open('/home/hungfei/桌面/paperdata/test.json', 'a', encoding='utf-8')
topic_json = json.dumps(topic_dict, ensure_ascii=False)
fo.write('%s\n' % topic_json)
