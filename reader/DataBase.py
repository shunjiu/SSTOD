import json
import logging
import os
import pickle
import random
import re
import pypinyin

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import ontology


class DB(object):
    def __init__(self, path, tfidf_path):
        self.db = []
        self.read_db(path)
        # self.vectorizer, self.weight = self.load_tf_idf_vector(tfidf_path)

    def read_db(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            db_file = json.load(f)
        for word, descs in db_file['nameDict'].items():
            word_descs = descs['word']
            struct_descs = descs['pack']
            for desc in word_descs:
                self.db.append((word, 'word', desc))

            for desc in struct_descs:
                self.db.append((word, 'pack', desc))

        self.word2kd = {}
        for kd in self.db:
            if kd[0] in self.word2kd:
                self.word2kd[kd[0]].append(kd[2])
            else:
                self.word2kd[kd[0]] = [kd[2]]

    # def _save_tf_idf_vector(self, file_path):
    #     logging.info('saving tfidf vector')
    #     all_data = [k[2] for k in self.db]
    #     pinyin_list = [' '.join(pypinyin.lazy_pinyin(s)) for s in all_data]
    #
    #     vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
    #     tfidf = vectorizer.fit_transform(pinyin_list)
    #     weight = tfidf.toarray()
    #     pickle.dump([vectorizer, weight], open(file_path, 'wb'))
    #
    #     return vectorizer, weight
    #
    # def _load_tf_idf_vector(self, file_path):
    #     logging.info('loading tfidf vector')
    #     vectorizer, weight = pickle.load(open(file_path, 'rb'))
    #     return vectorizer, weight
    #
    # def load_tf_idf_vector(self, save_path):
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     file_path = os.path.join(save_path, 'tfidf.pkl')
    #     if os.path.exists(file_path):
    #         vectorizer, weight = self._load_tf_idf_vector(file_path)
    #     else:
    #         vectorizer, weight = self._save_tf_idf_vector(file_path)
    #     return vectorizer, weight
    #
    # # TOP N
    # def research(self, batch, n: int=10):
    #     x_word = [' '.join(pypinyin.lazy_pinyin(sen)) for sen in batch]
    #     tf_idf = self.vectorizer.transform(x_word)
    #     cosine_similarities = linear_kernel(tf_idf, self.weight)
    #     related_docs_indices = cosine_similarities.argsort()[:, :-n-1:-1]
    #
    #     related_docs = [[self.db[ind] for ind in related_docs_indices[b]] for b in range(len(batch))]
    #     return related_docs

    def act_to_DBPointer(self, action):
        """
        Select a knowledge for a sub-slot.
        """
        if '[' in action:
            act_s_idx = action.index('[')
        else:
            act_s_idx = 0
        if ']' in action:
            act_e_idx = action.index(']')
        else:
            act_e_idx = len(action) - 1
        act = action[act_s_idx + 1:act_e_idx]
        param = action[act_e_idx + 1:].strip()
        if action not in ontology.action:
            return ''
        db_pointer = []
        if act == 'explicit_confirm':
            for w in param:
                if w in self.word2kd:
                    db = random.choice(self.word2kd[w])
                    db_pointer.append(db)
        db_results = ';'.join(db_pointer)
        if len(db_results) > 40:
            db_results = db_results[:40]
        return db_results


if __name__ == '__main__':
    class KdDataset(Dataset):
        def __init__(self, filename):
            super(KdDataset, self).__init__()
            self.kd, self.c_char = self.read_file(filename)
            assert len(self.kd) == len(self.c_char)

        def __getitem__(self, item):
            return self.kd[item], self.c_char[item]

        def __len__(self):
            return len(self.kd)

        @staticmethod
        def read_file(filename):
            dialogs = json.load(open(filename, 'r', encoding='utf-8'))
            all_kd = []
            all_char = []
            for dialog in dialogs.values():
                for turn in dialog['turns']:
                    if 'staff' in turn:
                        continue

                    kds = turn['user_label'][0]['param-knowledge']
                    for kd in kds:
                        if kd['knowledge']:
                            if re.search('[A-Za-z]]', kd['knowledge'][0]['string']):
                                continue
                            all_kd.append(kd['knowledge'][0]['string'])
                            all_char.append(kd['correct_char'])
            return all_kd, all_char
    db = DB('../../data/database.json', '../utils/tfidf/')
    dataset = KdDataset('../../data/test.json')
    kd_loader = DataLoader(dataset, batch_size=4000, shuffle=True)
    succ_count = 0
    total_count = 0
    for batch in tqdm(kd_loader):
        x, y_target = batch
        y_predict = db.research(x, 45)
        y_predict_char = [[kd[0] for kd in sample] for sample in y_predict]
        for i in range(len(x)):
            succ_count += (y_target[i] in y_predict_char[i])
            # if y_target[i] not in y_predict_char[i]:
            #     print(x[i], y_target[i])
            total_count += 1
    print('succ rate: ', succ_count / total_count)
