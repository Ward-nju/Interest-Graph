#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: ake
@time: 2017/3/21
主要训练：
    个体学者研究兴趣+AT
    个体学者研究兴趣+LDA
"""
import json
from collections import OrderedDict
from topic import get_corpus_by_aid, phrase_extraction
from author import author_paper
from gensim import corpora, models
from topic import lda_topic
from author import author_sort
import time
import numpy as np
import scipy.stats

def paper_author_name_list():
    '''获得文章对应的作者信息'''
    fo = open('D:/TopicInterestGraph/data/at/paper_authorName_info.json', 'a', encoding='utf-8')
    paper_authors = author_paper.get_paper_authors()
    paper_author_dict = {}
    for paper in paper_authors:
        author_list = []
        for author in paper_authors[paper]:
            if isinstance(author, dict):
                author_list.append(author['name'])
            else:
                author_list.append(author)
        paper_author_dict[paper] = author_list
    paper_author_json = json.dumps(paper_author_dict, ensure_ascii=False)
    fo.write('%s\n' % paper_author_json)
    fo.close()


def generate_doc2author(aid):
    '''为指定作者的AT模型所需的doc2author字典'''
    doc2author = OrderedDict()
    paper_list = get_corpus_by_aid.get_author_paper_list(aid)
    fo = open('D:/TopicInterestGraph/data/at/paper_authorName_info.json', 'r', encoding='utf-8')
    paper_authors = json.loads(fo.readline())
    fo.close()
    i = 0
    for paper in paper_list:
        author_list = paper_authors[paper]
        doc2author[i] = author_list
        i += 1
    return doc2author



def get_corpus(aid):
    '''获得指定aid的AT模型所需的训练数据、字典'''
    corpus = get_corpus_by_aid.get_author_corpus(aid)
    phrase_list = phrase_extraction.get_noun_phrase(corpus, 3, 3)
    corpus = phrase_extraction.replace_phrase(corpus, phrase_list)
    noun_corpus = get_corpus_by_aid.get_noun_corpus_list(corpus)
    dict_train, corpus_train = lda_topic.dict_corpora_from_list(noun_corpus)
    tfidf_train = models.TfidfModel(corpus_train)
    corpus_tfidf_train = list(tfidf_train[corpus_train])
    return corpus_tfidf_train, dict_train


def train_at_model(my_corpus, my_dict, topic_num, doc_author):
    '''训练AT模型'''
    at = models.AuthorTopicModel(
        my_corpus,
        num_topics=topic_num,
        doc2author=doc_author,
        id2word=my_dict,
        chunksize=5,
        update_every=1,
        passes=5)
    return at

"""
def find_topic_number(aid, corpus):
    '''为指定aid训练的个人AT模型寻找合适的主题个数（仅根据余弦相似度选择）'''
    topic_number = range(2, 20)
    sim_list = []
    # perplexity_list = []
    train_data = corpus[0]
    # test_data = corpus[0][0:5]
    my_dict = corpus[1]
    doc_author = generate_doc2author(aid)
    for num in topic_number:
        try:
            my_model = train_at_model(train_data, my_dict, num, doc_author)
            topics = my_model.show_topics(num, len(my_dict), formatted=False)
            topic_sim = lda_topic.average_sim(topics)
            sim_list.append(topic_sim)
        except:
            return False
        # perplexity = my_model.log_perplexity(test_data, [0, 1, 2, 3, 4])
        # perplexity_list.append(perplexity)

        # 使用困惑度除以预先相似度作者发现主题个数的指标
        # lda_topic.plot_topic_number_trend(topic_number, sim_list, 'Topic number',
        #                                   'Consine similarity')
        # lda_topic.plot_topic_number_trend(topic_number, perplexity_list,
        #                                   'Topic number', 'Perplexity')
        # per_cos_list = list(
        #     map(lambda x: x[0] / x[1], zip(perplexity_list, sim_list)))

    num_cos_dict = {}
    i = 0
    for num in topic_number:
        num_cos_dict[num] = sim_list[i]
        i += 1
    num_cos_dict = sorted(num_cos_dict.items(), key=lambda x: x[1])
    return num_cos_dict[0][0]
"""
def average_jsd(topics_list):
    numbers = len(topics_list)
    all_sim = 0
    for j in range(numbers - 1):
        for m in range(j + 1, numbers):
            li1 = [
                num[1] for num in sorted(
                    topics_list[j][1], key=lambda li: li[0])
            ]
            li2 = [
                num[1] for num in sorted(
                    topics_list[m][1], key=lambda li: li[0])
            ]
            p=np.array(li1)
            q=np.array(li2)
            M=(p+q)/2
            
            js=0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q,M)
            all_sim += js
    return all_sim / (numbers * (numbers - 1) / 2)

def find_topic_number(aid, corpus):
    '''为指定aid训练的个人LDA模型寻找合适的主题个数（仅根据余弦相似度选择）'''
    topic_number = range(2, 20)
    sim_list = []
    # perplexity_list = []
    train_data = corpus[0]
    # test_data = corpus[0][0:5]
    my_dict = corpus[1]
    for num in topic_number:
        try:
            my_model = lda_topic.lda_model(train_data, num,my_dict)
            topics = my_model.show_topics(num, len(my_dict), formatted=False)
            #topic_sim = lda_topic.average_sim(topics)
            topic_sim=average_jsd(topics)
            sim_list.append(topic_sim)
        except:
            return False
    num_cos_dict = {}
    i = 0
    for num in topic_number:
        num_cos_dict[num] = sim_list[i]
        i += 1
    num_cos_dict = sorted(num_cos_dict.items(), key=lambda x: x[1])
    return num_cos_dict[0][0]

def get_author_at_topics(aid, name, topic_number, corpus):
    '''获取指定作者的AT模型'''
    train_data = corpus[0]
    my_dict = corpus[1]
    doc_author = generate_doc2author(aid)

    my_model = train_at_model(train_data, my_dict, topic_number, doc_author)
    model_name = 'D:/TopicInterestGraph/data/at/model/aid-%d.gensim' % aid
    my_model.save(model_name)

    topic_distribute = my_model.get_author_topics(name)
    topics = []
    for topic in topic_distribute:
        topics.append(my_model.show_topic(topic[0], topn=10))
    author_dict = OrderedDict()
    author_dict['aid'] = aid
    author_dict['name'] = name
    author_dict['topic distribute'] = topic_distribute
    author_dict['topics list'] = topics
    return author_dict


def get_aid_list():
    '''获取实验所需的aid列表'''
    fo = open('D:/TopicInterestGraph/data/lda/author_topic_distribution.json', 'r', encoding='utf-8')
    aid_list = []
    for line in fo.readlines():
        temp = json.loads(line)
        aid_list.append(list(temp.keys())[0])
    fo.close()
    return aid_list


def at_result():
    '''获得每个作者的AT模型中的主题分布'''
    aid_list = get_aid_list()
    aid_name = author_sort.get_aid_name(aid_list)
    fo = open('D:/TopicInterestGraph/data/at/author_topics.json', 'a', encoding='utf-8')
    for aid in aid_list:
        name = aid_name[aid]
        corpus = get_corpus(int(aid))
        topic_number = find_topic_number(int(aid), corpus)
        if topic_number == False: continue
        print(name + 'has' + str(topic_number) + 'topics.')

        author_dict = get_author_at_topics(
            int(aid), name, topic_number, corpus)
        author_json = json.dumps(author_dict, ensure_ascii=False)
        fo.write('%s\n' % author_json)
        print(aid)
    fo.close()

    


def get_author_lda_topics(aid, name, topic_number, corpus):
    '''获得指定作者的LDA主题模型下的主题分布'''
    train_data = corpus[0]
    my_dict = corpus[1]

    my_model = lda_topic.lda_model(train_data, topic_number, my_dict)
    model_name = 'D:/TopicInterestGraph/data/lda_person/model_new/aid-%d.gensim' % aid
    my_model.save(model_name)

    topic_list = my_model.show_topics(topic_number, my_model.num_terms, formatted=False)
    topics = []
    for topic in topic_list:
        topics.append([topic[0],topic[1]])
        #topics.append((topic[0], '、'.join([word[0] for word in topic[1]])))
    author_dict = OrderedDict()
    author_dict['aid'] = aid
    author_dict['name'] = name
    author_dict['number_of_topic'] = topic_number
    author_dict['topics'] = topics
    return author_dict
"""

def get_author_lda_topics(aid, name, topic_number):
    model_name = 'D:/TopicInterestGraph/data/lda_person/model/aid-%d.gensim' % aid
    my_model=models.LdaModel.load(model_name)
    #topic_list = my_model.show_topics(topic_number, 10, formatted=False)
    topic_list = my_model.show_topics(topic_number, my_model.num_terms, formatted=False)
    topics=[]
    for topic in topic_list:
        topics.append([topic[0],topic[1]])
        
    fo=open('D:/TopicInterestGraph/data/author_unit_topic_paper.json', 'r',encoding='utf-8')
    for line in fo.readlines():
        temp = json.loads(line)
        if temp['aid']==int(aid):
            unit=temp['unit']
            number_of_paper=temp['number_of_paper']
            papers=temp['papers']
            break
    fo.close()
    
    author_dict = OrderedDict()
    author_dict['aid'] = aid
    author_dict['name'] = name
    author_dict['unit']=unit
    author_dict['number_of_paper']=number_of_paper
    author_dict['papers']=papers
    author_dict['number_of_topic'] = topic_number
    author_dict['topics'] = topics
    return author_dict
"""

def lda_result():
    '''获取每位作者的LDA模型主题分布结果'''
    aid_list = get_aid_list()
    aid_name = author_sort.get_aid_name(aid_list)
    #fw = open('D:/TopicInterestGraph/data/lda_person/author_topics.json', 'a', encoding='utf-8')
    fw = open('D:/TopicInterestGraph/data/lda_person/model_lda/author_topics_distribution.json', 'a', encoding='utf-8')
    for aid in aid_list:
        name = aid_name[aid]
        corpus = get_corpus(int(aid))
        topic_number = find_topic_number(int(aid), corpus)
        if topic_number == False: continue
        print(aid)
        print(name + ' has ' + str(topic_number) + ' topics.')
        author_dict = get_author_lda_topics(
            int(aid), name, topic_number, corpus)
        author_json = json.dumps(author_dict, ensure_ascii=False)
        fw.write('%s\n' % author_json)
    fw.close()
"""
def lda_result():
    '''获取每位作者的LDA模型主题分布结果'''
    aid_list = get_aid_list()
    aid_name = author_sort.get_aid_name(aid_list)
    #fw = open('D:/TopicInterestGraph/data/lda_person/author_topics.json', 'a', encoding='utf-8')
    fw = open('D:/TopicInterestGraph/data/lda_person/author_topics_allitems.json', 'a', encoding='utf-8')
    for aid in aid_list:
        name = aid_name[aid]
        #corpus = get_corpus(int(aid))
        #topic_number = find_topic_number(int(aid), corpus)
        topic_number=0
        fo=open('D:/TopicInterestGraph/data/lda_person/author_topics_new.json', 'r',encoding='utf-8')
        for line in fo.readlines():
            temp = json.loads(line)
            if temp['aid']==int(aid):
                topic_number=temp['number_of_topic']
                break
        fo.close()
    
        if topic_number == 0: continue
        print(aid)
        print(name + ' has ' + str(topic_number) + ' topics.')
        #print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        author_dict = get_author_lda_topics(
            int(aid), name, topic_number)
        author_json = json.dumps(author_dict, ensure_ascii=False)
        fw.write('%s\n' % author_json)
    fw.close()
"""    
    
def main():
    pass


if __name__ == '__main__':
    main()