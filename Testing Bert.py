#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:49:16 2019

@author: yatingupta
"""
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
bc = BertClient()
x = bc.encode(['Tiger', 'Cat'])
a = x[0]
b = x[1]

a = a.reshape(1,768)
b = b.reshape(1,768)
q = cosine_similarity(a,b)
q = q[0][0]
#Macintosh HD⁩ ▸ ⁨Users⁩ ▸ ⁨yatingupta⁩ ▸ ⁨Documents⁩ ▸ ⁨NLP Datasets⁩

# bert-serving-start -model_dir Documents/NLP_Datasets/bert/ -num_worker=4
