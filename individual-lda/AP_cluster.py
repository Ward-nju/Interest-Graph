# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:19:25 2017

@author: hungfei
"""

import json
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import lil_matrix
from sklearn.cluster import AffinityPropagation

def load_author_topics_distribution():
    topics=[]
    f=open('D:/TopicInterestGraph/data/lda_person/model_lda/author_topics_distribution_10items.json','r',encoding='utf-8')
    for line in f.readlines():
        temp=json.loads(line)
        topics.append([temp['aid'],temp['topics']])
    f.close() 
    return topics

topics=load_author_topics_distribution()
topics=[[x[0],y[0],y[1]] for x in topics for y in x[1]]
"""
word_list=[]
i=0
for record in topics:
    print(i)
    i+=1
    for item in record[2]:
        if item[0] not in word_list:
            word_list.append(item[0])
with open('D:/TopicInterestGraph/data/lda_person/word_list_20.pkl','wb') as f:
    pickle.dump(word_list,f)

"""

"""
f=open('D:/TopicInterestGraph/data/lda_person/word_list_20.pkl','rb')
word_list=pickle.load(f)
f.close()

def fun(topics,word_list):
    #n_sample * n_words
    X=lil_matrix((len(topics),len(word_list)))
    row=0
    for record in topics:
        print(row)
        topic=record[2]
        words=[item[0] for item in topic]
        i=0
        for word in words:
            col=word_list.index(word)
            X[row,col]=topic[i][1]
            i+=1
        row+=1
    return X

X=fun(topics,word_list)

"""

#计算各主题TOP20的Jaccard系数，作为主题之间的相似度，然后用AP聚类
"""
X=np.zeros((len(topics),len(topics)))
for i in range(len(topics)-1):
    print(i)
    for j in range(i+1,len(topics)):
        list1=topics[i][2]
        list2=topics[j][2]
        topic_word1=[item[0] for item in list1]
        topic_word2=[item[0] for item in list2]
        a=len(set(topic_word1) & set(topic_word2))
        b=len(set(topic_word1) | set(topic_word2))
        X[i,j]=a/b
with open('D:/TopicInterestGraph/data/lda_person/model_lda/X_10_Jaccard.pkl','wb') as f:
    pickle.dump(X,f)
"""
"""
f=open('D:/TopicInterestGraph/data/lda_person/model_lda/X_10_JS.pkl','rb')
X=pickle.load(f)
f.close()
clf=AffinityPropagation(affinity='precomputed')
X=X.T+X
X=-1*X
clf.fit(X)
labels=clf.labels_
"""