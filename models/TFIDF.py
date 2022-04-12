import heapq
import json
import pickle
import re

import numpy as np
from pypinyin import lazy_pinyin
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm


class TfIdf(object):
    def __init__(self):
        self.word_vectorizer, self.word_transformer, self.word_weight = pickle.load(open('./data/kb/db_words.pkl', 'rb'))
        self.pinyin_vectorizer, self.pinyin_transformer, self.pinyin_weight = pickle.load(
            open('./data/kb/db_pinyin.pkl', 'rb'))
        self.db = json.load(open('./data/kb/db_key.json', 'r', encoding='utf-8'))

    def search(self, sen, n: int = 1, rate=0.09):
        x_word = [' '.join(sen)]
        tf_idf = self.word_transformer.transform(self.word_vectorizer.transform(x_word))
        x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
        result_word = cosine_similarity(x_test_weight, self.word_weight)

        x_pinyin = [' '.join(lazy_pinyin(sen))]
        tf_idf = self.pinyin_transformer.transform(self.pinyin_vectorizer.transform(x_pinyin))
        x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
        result_pinyin = cosine_similarity(x_test_weight, self.pinyin_weight)

        if n == 1:
            result = result_word * rate + result_pinyin * (1 - rate)
            re2 = np.argmax(result, axis=1)[0]
            all_ = [[self.db[re2], result[0][re2]]]
        else:
            result = (result_word * rate + result_pinyin * (1 - rate)).tolist()[0]
            re2 = map(result.index, heapq.nlargest(n, result))
            all_ = [[self.db[i], result[idx][i]] for idx, i in enumerate(re2)]
            # all_ = [self.db[i] for i in list(re2)]
        return all_

    def search_batch(self, batch, n: int = 1, rate=0.09):
        x_word = [' '.join(sen) for sen in batch]
        tf_idf = self.word_transformer.transform(self.word_vectorizer.transform(x_word))
        x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
        result_word = cosine_similarity(x_test_weight, self.word_weight)

        x_pinyin = [' '.join(lazy_pinyin(sen)) for sen in batch]
        tf_idf = self.pinyin_transformer.transform(self.pinyin_vectorizer.transform(x_pinyin))
        x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
        result_pinyin = cosine_similarity(x_test_weight, self.pinyin_weight)

        result = result_word * rate + result_pinyin * (1 - rate)
        re2 = torch.topk(torch.tensor(result), n).indices.tolist()
        all_ = [[[self.db[i], result[idx][i]] for i in t_k] for idx, t_k in enumerate(re2)]
        candidates_word = [word[0][0] for word in all_]
        return ' '.join(candidates_word)