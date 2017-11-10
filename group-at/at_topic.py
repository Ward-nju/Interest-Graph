#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: ake
@time: 2017/5/17
主要用来训练：
    集体学者研究兴趣网络+AT
"""
import json
import pickle
from topic import person_topic, lda_topic
from collections import OrderedDict
from topic import get_corpus_by_aid, phrase_extraction
from gensim import corpora, models
import time



aid_list = person_topic.get_aid_list()


def get_aid_name_paperlist(aidList):
    '''获得作者编号列表对应的作者名称、对应的文章列表'''
    aid_name = {}
    fo = open('D:/TopicInterestGraph/data/author_unit_paper.json', 'r', encoding='utf-8')
    for line in fo.readlines():
        temp = json.loads(line)
        aid = temp['aid']
        if str(aid) in aidList:
            aid_name[str(aid)] = [temp['name'], temp['paper_list']]
    return aid_name


def generate_new_doc2author():
    '''生成1500位作者的所有文章的doc2author字典'''
    doc2author = OrderedDict()
    all_papers = {}
    papers_list = []
    fo = open('D:/TopicInterestGraph/data/at/paper_authorName_info.json', 'r', encoding='utf-8')
    paper_authors = json.loads(fo.readline())
    aid_name_papers = get_aid_name_paperlist(aid_list)
    i = 0
    for aid in aid_list:
        papers = aid_name_papers[aid][1]
        for paper in papers:
            if paper not in all_papers:
                all_papers[paper] = i
                papers_list.append(paper)
                authors = paper_authors[paper]
                authors = [
                    aid if x == aid_name_papers[aid][0] else x for x in authors
                ]
                doc2author[i] = authors
                i += 1
            else:
                authors = doc2author[all_papers[paper]]
                authors = [
                    aid if x == aid_name_papers[aid][0] else x for x in authors
                ]
                doc2author[all_papers[paper]] = authors
    return doc2author, papers_list


doc2author, papers = generate_new_doc2author()
for i in range(len(doc2author)):
    doc2author[i]=[item for item in doc2author[i] if item.isdigit()]

def generate_corpus():
    '''为集体AT模型生成语料'''
    # corpus = get_corpus_by_aid.get_papers_corpus(papers)
    # with open('data/at/one_at_origina_corpus.pickle', 'wb') as f:
    #     pickle.dump(corpus, f)
    fo = open('D:/TopicInterestGraph/data/at/one_at_origina_corpus.pickle', 'rb')
    corpus = pickle.load(fo)
    print('corpus')
    fo.close()

    # phrase_list = phrase_extraction.get_noun_phrase(corpus, 5, 5)
    # with open('data/at/one_at_phrase_list.pickle', 'wb') as f:
    #     pickle.dump(phrase_list, f)
    fo1 = open('D:/TopicInterestGraph/data/at/one_at_phrase_list.pickle', 'rb')
    phrase_list = pickle.load(fo1)
    print('phrase_list')
    fo1.close()
    
    corpus=corpus
    phrase_list=phrase_list
    
    corpus = phrase_extraction.replace_phrase(corpus, phrase_list)
    noun_corpus = get_corpus_by_aid.get_noun_corpus_list(corpus)
    dict_train, corpus_train = lda_topic.dict_corpora_from_list(noun_corpus)
    tfidf_train = models.TfidfModel(corpus_train)
    corpus_tfidf_train = list(tfidf_train[corpus_train])
    return corpus_tfidf_train, dict_train


def find_topic_number(corpus):
    '''集体AT模型寻找合适的主题个数（仅根据余弦相似度选择）'''
    topic_number = range(80, 100, 20)
    sim_list = []
    train_data = corpus[0]
    my_dict = corpus[1]
    for num in topic_number:
        print(num,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        try:
            my_model = person_topic.train_at_model(train_data, my_dict, num, doc2author)
            model_name = 'D:/TopicInterestGraph/data/at/model_all/%dtopics-at.gensim' % num
            my_model.save(model_name)
            print('model finish!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            topics = my_model.show_topics(num, len(my_dict), formatted=False)
            topic_sim = lda_topic.average_sim(topics)
            print('topic_sim:',topic_sim)
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

    
def main():    
    fo = open('D:/TopicInterestGraph/corpus.pickle', 'rb')
    corpus = pickle.load(fo)
    fo.close()
    #corpus = generate_corpus()
    print('Then find proper topic number:')
    topic_number = find_topic_number(corpus)
    print('topic number is:' + str(topic_number))
    #train_data = corpus[0]
    #my_dict = corpus[1]
    #my_model = person_topic.train_at_model(train_data, my_dict, topic_number, doc2author)
    
    
if __name__ == '__main__':
    #main()
    pass