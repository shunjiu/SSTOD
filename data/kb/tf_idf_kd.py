# coding:utf-8
import heapq
import json
import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypinyin import lazy_pinyin
from tqdm import tqdm


class Tf_idf():
    def __init__(self, type='word'):
        db = json.load(open('knowledge_db.json', 'r', encoding='utf-8'))

        if type == 'word':
            db_for_tfidf = [' '.join(value) for value in db]
        else:
            db_for_tfidf = [' '.join(lazy_pinyin(value)) for value in db]

        analyzer = 'char' if type=='word' else 'word'
        vectorizer = CountVectorizer(analyzer=analyzer, lowercase=False)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(db_for_tfidf))
        weight = tfidf.toarray()
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.weight = weight
        self.type = type

        filename = './db_words.pkl' if type == 'word' else './db_pinyin.pkl'
        pickle.dump([vectorizer, transformer, weight], open(filename, 'wb'))


    def search(self, sen, n, db):
        if self.type == 'word':
            x_test = [' '.join(sen)]
        else:
            x_test = [' '.join(lazy_pinyin(sen))]
        tf_idf = self.transformer.transform(self.vectorizer.transform(x_test))
        x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
        result = list(cosine_similarity(x_test_weight, self.weight)[0])
        re2 = map(result.index, heapq.nlargest(n, result))
        all_ = [[db[i], result[i]] for i in list(re2)]
        return all_


def search_2_idf(sen, n, word_idf, pinyin_idf, rate, db):
    x_word = [' '.join(sen)]
    tf_idf = word_idf[1].transform(word_idf[0].transform(x_word))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
    result_word = list(cosine_similarity(x_test_weight, word_idf[2])[0])

    x_pinyin = [' '.join(lazy_pinyin(i)[0] for i in sen)]
    tf_idf = pinyin_idf[1].transform(pinyin_idf[0].transform(x_pinyin))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
    result_pinyin = list(cosine_similarity(x_test_weight, pinyin_idf[2])[0])

    result = (np.array(result_word) * rate + np.array(result_pinyin) * (1-rate)).tolist()
    re2 = map(result.index, heapq.nlargest(n, result))
    # all_ = [[list(db.keys())[i], result[i]] for i in list(re2)]
    all_ = [db[i] for i in list(re2)]
    return all_

def search_1_idf(sen, n, word_idf, db):
    x_word = [' '.join(sen)]
    tf_idf = word_idf[1].transform(word_idf[0].transform(x_word))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
    result_word = list(cosine_similarity(x_test_weight, word_idf[2])[0])

    re2 = map(result_word.index, heapq.nlargest(n, result_word))
    all_ = [[db[i], result_word[i]] for i in list(re2)]
    # all_ = [db[i] for i in list(re2)]
    return all_


if __name__ == '__main__':
    tfidf = Tf_idf(type='pinyin')
    tfidf2 = Tf_idf(type='word')