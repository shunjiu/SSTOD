import copy
import json
import logging
import os
import random
from collections import OrderedDict

from torch.utils.data import Dataset

import ontology
from reader.data_util import read_name_dialog_from_file
from utils import utils


class _BaseDataset(Dataset):
    def __init__(self, args):
        super(_BaseDataset, self).__init__()
        self.vocab = None
        self.db = None
        self.args = args
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data, batch_size):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key=='dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def collate_fn(self, data_batch):
        pass


class UBARDataset(_BaseDataset):
    def __init__(self, args, tokenizer, data_path, mode, kb):
        super(UBARDataset, self).__init__(args)
        self.args = args
        self.kb = kb
        self.tokenizer = tokenizer
        args.mode = mode
        self.pad_token_id = 0
        self._load_data(data_path, mode)

    def _load_data(self, data_path, mode):
        encoded_file_list = {'train': 'train.encoded.UBARdata.json', 'dev': 'dev.encoded.UBARdata.json',
                             'test': 'test.encoded.UBARdata.json'}
        encoded_file = encoded_file_list[mode]
        encoded_file = os.path.join(data_path, encoded_file)
        if not os.path.exists(encoded_file):
            logging.info('Encoding data and save the encoded data in {}'.format(encoded_file))
            raw_data_path = os.path.join(data_path, mode + '.json')
            dialogs = self.read_data(raw_data_path)
            encoded_data = self._get_seqs(dialogs)
            json.dump(encoded_data, open(encoded_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            self.data = copy.deepcopy(encoded_data)
            random.shuffle(self.data)
        else:
            logging.info('loading encoded data from {}'.format(encoded_file))
            encoded_data = json.load(open(encoded_file, 'r', encoding='utf-8'))
            self.data = copy.deepcopy(encoded_data)
            random.shuffle(self.data)

    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        data_file = read_name_dialog_from_file(raw_data)
        return data_file

    def _get_seqs(self, dialogs):
        data_list = []
        for fn, dial in dialogs.items():
            encoded_dial = []
            for idx, t in enumerate(dial['log']):
                enc = {}
                enc['dial_id'] = fn
                enc['user'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_u> ' +
                    t['user'] + ' <eos_u>'))
                enc['resp'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_r> ' +
                    t['sys'] + ' <eos_r>'))
                enc['bspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_b> ' + t['bs'] + ' <eos_b>'))
                enc['aspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_a> ' + '[' +
                    t['sys_act'] + ']' + t['sys_act_param'] + ' <eos_a>'))
                enc['turn_num'] = t['turn_num']
                if 'db' in t:
                    enc['db'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                        '<sos_db> ' + '[SEP]'.join(t['db']) + ' <eos_db>'))
                else:
                    enc['db'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                        '<sos_db> ' + ' <eos_db>'))
                enc['cc'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_c> ' + t['correct_char'] + ' <eos_c>'))
                enc['kdpn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                    '<sos_k>' + t['kdpn'] + '<eos_k>'))

                encoded_dial.append(enc)
            data_list.append(encoded_dial)
        return data_list

    def convert_batch_session(self, dial_batch):
        """
        convert the whole session for training
        concat [U_0, K_0, c_0, B_0, A_0, R_0, ... , U_n, K_n, c_n, B_n, A_n, R_n]
        """

        inputs, outputs = {}, {}
        contexts = []

        cell_list = ['user', 'kdpn', 'cc', 'bspn', 'aspn', 'db', 'resp']
        for idx, dial in enumerate(dial_batch):
            context = []
            for turn_num, turn in enumerate(dial):
                for cell in cell_list:
                    context.extend(turn[cell])
            contexts.append(context)

        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], self.pad_token_id)

        return inputs

    def get_batches(self, batch_size):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''
        dial = self.data
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k], batch_size)
            log_str += "turn num:%d, batch num: %d last batch len: %d\n" % (
                k, len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))

        return all_batches

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
            firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        # predict_list = []
        prompt = ''
        # predict bspn aspn resp. db are not predicted. this part tbd.
        context_list = ['user']
        # predict_list = ['bspn', 'aspn','db', 'resp']
        prompt = '<sos_k>'

        if first_turn:
            context = []
            for c in context_list:
                context += turn[c]

            inputs['context'] = context + self.tokenizer.encode(prompt, add_special_tokens=False)
            inputs['labels'] = context

        else:
            context = []
            for c in context_list:
                context += turn[c]

            pv_context = pv_turn['labels'] + pv_turn['kdpn'] + pv_turn['cc'] + pv_turn['bspn'] + pv_turn['aspn'] + pv_turn['db'] + pv_turn['resp']

            # prompt response, add sos_r
            inputs['context'] = pv_context + context + self.tokenizer.encode(prompt, add_special_tokens=False)

            inputs['labels'] = pv_context + context  # use all previous ubar history

        if len(inputs['context']) > 900:
            # print('len exceeds 900')
            inputs['context'] = inputs['context'][-900:]

        return inputs

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                 'bspn', 'kdpn', 'kdpn_gen', 'cc', 'cc_gen']
        for dial_id, turns in result_dict.items():

            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])

                        v = " ".join(v)
                    else:
                        pass # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

    def __len__(self):
        return len(self.data)