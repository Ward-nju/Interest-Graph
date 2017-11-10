# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:43:10 2017

@author: hungfei
"""

import numpy as np
import pandas as pd
import json

num_topic=60
num_seq=11
seq=[887,2047,2081,1963,2000,2092,2056,2289,2093,2080,1076]
filename='D:/TopicInterestGraph/dtm/paperdata/result/lda-seq/gam.dat'
f=open(filename,'r')
x=[]
for line in f.readlines():
    x.append(float(line.strip()))
f.close()
x=np.array(x)
x=x.reshape(int(len(x)/num_topic),num_topic)
"""
for i in range(len(x)):
    x[i]=x[i]/np.sum(x,axis=1)[i]
   
   
seq=[0,887,2934,5015,6978,8978,11070,13126,15415,17508,19588,20664]

topic_intensity=np.array([[0.0 for y in range(num_topic)] for x in range(num_seq)])
for i in range(len(seq)-1):
    temp=x[seq[i]:seq[i+1]]
    topic_intensity[i]=np.sum(temp,axis=0)/len(temp)
""" 