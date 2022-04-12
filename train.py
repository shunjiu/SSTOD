import sys

from torch import nn
from transformers import AutoTokenizer

import ontology
from models.model import UBAR_plus

sys.path.append('..')
from evaluate import validate_metric, validation_metric_gpt

import argparse
import json
import logging
import os
import random
import time

from torch.utils.data import RandomSampler, DistributedSampler, BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import parser, BIO_TAG

import numpy as np
import torch
import torch.nn.functional as F

from reader import data_util
from reader.DataBase import DB
from utils.optim import Optim
from utils.utils import log_first_inputs, maskedNll
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer):
    """
        add special tokens to gpt tokenizer
        serves a similar role of Vocab.construt()
        make a dict of special tokens
    """

    special_tokens = []
    action = ontology.all_acts
    for word in action:
        word = '[' + word + ']'
        special_tokens.append(word)

    special_tokens_list = ontology.special_tokens
    special_tokens.extend(special_tokens_list)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def add_torch_input(inputs, device):
    # to tensor and to device
    contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
    contexts_tensor = contexts_tensor.to(device)
    inputs['contexts_tensor'] = contexts_tensor
    return inputs

def add_torch_input_eval(inputs, device):
    # inputs: context
    inputs['context_tensor'] = torch.tensor(
        [inputs['context']]).to(device)
    return inputs


def calculate_loss_and_accuracy(outputs, labels):
    lm_logits = outputs[0]

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    pad_id = 0
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # avg loss
    not_ignore = shift_labels.ne(pad_id)
    num_targets = not_ignore.long().sum().item()

    loss /= num_targets
    return loss


def train(args, dataloader, dev_dataloader, model, tokenizer, writer, optimizer, scheduler, num_train_steps, device):
    log_inputs = 2
    global_step = 0
    all_batches = dataloader.get_batches(args.train_batch_size)
    for epoch in range(args.num_train_epochs):
        epoch_step = 0
        tr_loss = 0.0
        logging_loss = 0.0
        btm = time.time()
        oom_time = 0
        optimizer.zero_grad()

        train_iterator = dataloader.get_nontranspose_data_iterator(all_batches)
        for batch_idx, batch in enumerate(train_iterator):
            # train
            inputs = dataloader.convert_batch_session(batch)
            model.train()
            if log_inputs > 0:
                log_first_inputs({'input': tokenizer.decode(inputs['contexts'][0])})
                log_inputs -= 1

            # to tensor
            inputs = add_torch_input(inputs, device)
            outputs = model(inputs['contexts_tensor'])
            loss = calculate_loss_and_accuracy(
                outputs, labels=inputs['contexts_tensor'])
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)
            epoch_step += 1

            # step, wrt gradient_accumulation_steps, clip grad norm
            if (epoch_step + 1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1 == num_train_steps):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                # global_step: actual step the optimizer took
                global_step += 1

                logs = {}
                # logging: loss, lr... after certain amount of steps
                if args.report_interval > 0 and global_step % args.report_interval == 0:
                    loss_scalar = (tr_loss - logging_loss) / args. report_interval
                    logging_loss = tr_loss
                    logs['loss'] = loss_scalar
                    logging.info(
                        'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                            global_step, epoch_step, loss_scalar
                        ))

        logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
            (time.time()-btm) / 60, tr_loss))

        # save model after every epoch
        save_model(args.exp_path, epoch, tr_loss/epoch_step, model, tokenizer)


def validate(args, dataloader, model, tokenizer, db, device):
    logging.info("**** Running Evaluation ****")

    eval_data = dataloader.data
    btm = time.time()
    result_collection = {}
    with torch.no_grad():
        for dial_idx, dialog in enumerate(tqdm(eval_data)):
            pv_turn = {}
            for turn_idx, turn in enumerate(dialog):
                first_turn = (turn_idx == 0)
                inputs = dataloader.convert_turn_eval(turn, pv_turn, first_turn)
                inputs = add_torch_input_eval(inputs, device)

                context_length = len(inputs['context'])

                # generate kd_snippets
                if args.use_true_curr_kdpn:
                    outputs = turn['kdpn']
                    kdpn_gen, decoded_kdpn = decode_generated_kdpn(tokenizer, outputs)
                else:
                    max_len = 40
                    outputs = model.generate(input_ids=inputs['context_tensor'],
                                                  max_length=context_length + max_len, temperature=0.7,
                                                  pad_token_id=0,
                                                  eos_token_id=tokenizer.encode('<eos_k>')[1])
                    generated = outputs[0].cpu().numpy().tolist()
                    kdpn_gen, decoded_kdpn = decode_generated_kdpn(tokenizer, generated[context_length - 1:])

                # generate correct_char and score
                if args.use_true_curr_kp:
                    outputs = turn['cc']
                    decoded_cc = decode_outputs_cc(tokenizer, outputs)
                else:
                    decoded_cc = model.generate_correct_char(decoded_kdpn) if len(decoded_kdpn) else ''

                ccs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                    '<sos_c>' + decoded_cc + '<eos_c>'))
                inputs['context_tensor'] = torch.tensor([inputs['context'][:-1] + kdpn_gen + ccs
                                                         + tokenizer.encode('<sos_b>',
                                                                                 add_special_tokens=False)]).to(device)
                context_length = len(inputs['context_tensor'][0])
                # generate bspn, act, response
                outputs = model.generate(input_ids=inputs['context_tensor'],
                                              max_length=context_length + 80, temperature=0.7,
                                              # top_p=0.9, num_beams=4,
                                              pad_token_id=0,
                                              eos_token_id=tokenizer.encode('<eos_a>')[1])
                generated_bsa = outputs[0].cpu().numpy().tolist()
                generated_bsa = generated_bsa[context_length - 1:]
                try:
                    bspn_gen, aspn_gen = decode_generated_bspn_act(tokenizer, generated_bsa)
                except ValueError as exception:
                    logging.info(str(exception))
                    logging.info(tokenizer.decode(generated_bsa))
                    aspn_gen, bspn_gen = [], []

                # query DB
                if args.use_true_db_pointer:
                    db = turn['db']
                else:

                    db_results = dataloader.kb.act_to_DBPointer(tokenizer.decode(aspn_gen[1:-1]))
                    db = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize('<sos_db> ' + db_results + ' <eos_db>'))
                inputs['context_tensor_db'] = torch.tensor(
                    [inputs['context'][:-1] + kdpn_gen + ccs + generated_bsa + db + tokenizer.encode(
                        '<sos_r>', add_special_tokens=False)]).to(
                    device)

                context_length = len(inputs['context_tensor_db'][0])
                outputs_db = model.generate(input_ids=inputs['context_tensor_db'],
                                                 max_length=context_length + 40, temperature=0.7,
                                                 # top_p=0.9, num_beams=4,
                                                 pad_token_id=0,
                                                 eos_token_id=tokenizer.encode('<eos_r>')[1])

                generated_r = outputs_db[0].cpu().numpy().tolist()
                generated_r = generated_r[context_length - 1:]
                try:
                    resp_gen = decode_generated_resp(tokenizer, generated_r)
                    decoded = {'kdpn': kdpn_gen, 'cc': ccs, 'bspn': bspn_gen, 'aspn': aspn_gen, 'resp': resp_gen}
                except ValueError as exception:
                    logging.info(str(exception))
                    logging.info(tokenizer.decode(generated_r))
                    decoded = {'kdpn': [], 'resp': [], 'cc': [], 'bspn': [], 'aspn': []}

                turn['resp_gen'] = decoded['resp']
                turn['kdpn_gen'] = turn['kdpn'] if args.use_true_curr_kdpn else decoded['kdpn']
                turn['cc_gen'] = turn['cc'] if args.use_true_curr_kp else decoded['cc']
                turn['bspn_gen'] = decoded['bspn']
                turn['aspn_gen'] = decoded['aspn']

                pv_turn['labels'] = inputs['labels']  # all true previous context
                pv_turn['resp'] = decoded['resp']
                pv_turn['bspn'] = decoded['bspn']
                pv_turn['kdpn'] = turn['kdpn'] if args.use_true_curr_kdpn else decoded['kdpn']
                pv_turn['cc'] = turn['cc'] if args.use_true_curr_kp else decoded['cc']
                pv_turn['db'] = turn['db'] if args.use_true_db_pointer else db
                pv_turn['aspn'] = decoded['aspn']

                result_collection.update(
                    dataloader.inverse_transpose_turn(dialog))

    results, _ = dataloader.wrap_result_lm(result_collection)
    data_path = os.path.join(args.data_dir, 'test.json')
    joint_acc, slot_acc, success = validation_metric_gpt(data_path, dataloader, results)
    logging.info('test' + ' results: joint_acc: {:.2f}\tslot_acc: {:.2f}\tsuccess: {:.2f}'
                 .format(joint_acc * 100, slot_acc * 100, success * 100))


def decode_generated_kdpn(tokenizer, generated):
    sos_k_id = tokenizer.encode('<sos_k>')[1]
    eos_k_id = tokenizer.encode('<eos_k>')[1]
    kd_id = tokenizer.encode('<kd>')[1]
    if sos_k_id in generated:
        sos_k_idx = generated.index(sos_k_id)
    else:
        sos_k_idx = 1
    if eos_k_id in generated:
        eos_k_idx = generated.index(eos_k_id)
    else:
        eos_k_idx = len(generated) - 1
    kspn_gen = generated[sos_k_idx: eos_k_idx + 1]
    decoded_gen = tokenizer.decode(kspn_gen[1: -1])

    decoded_gen = decoded_gen.strip().split('<kd>')
    if '' in decoded_gen:
        decoded_gen.remove('')
    return kspn_gen, decoded_gen


def decode_outputs_cc(tokenizer, outputs):
    cc = []
    for output in outputs:
        char = tokenizer.decode(output)
        if char == '<sos_c>' or char =='<eos_c>':
            continue
        cc.append(char)
    return ' '.join(cc)


def decode_generated_bspn_act(tokenizer, generated):
    """decode generated"""
    eos_b_id = tokenizer.encode('<eos_b>')[1]
    eos_a_id = tokenizer.encode('<eos_a>')[1]

    # eos_a may not exists if gpt2 generated repetitive words
    if eos_a_id in generated:
        eos_a_idx = generated.index(eos_a_id)
    else:
        eos_a_idx = len(generated) - 1
        logging.info('eos_a not in generated: ' + tokenizer.decode(generated))

    eos_b_idx = generated.index(eos_b_id)
    bspn_gen = generated[: eos_b_idx + 1]
    aspn_gen = generated[eos_b_idx + 1: eos_a_idx + 1]
    return bspn_gen, aspn_gen


def decode_generated_resp(tokenizer, generated):
    eos_r_id = tokenizer.encode('<eos_r>')[1]
    if eos_r_id in generated:
        eos_r_idx = generated.index(eos_r_id)
    else:
        eos_r_idx = len(generated) - 1
    return generated[: eos_r_idx + 1]

def save_model(exp_path, epoch, loss, model, tokenizer):
    save_path = os.path.join(
        exp_path, 'epoch{}_trloss{:.2f}'.format(epoch+1, loss))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logging.info('Saving model checkpoint to %s', save_path)

    # save model
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(save_path, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), output_model_file)
    # save tokenizer
    # tokenizer.save(os.path.join(save_path, 'tokenizer.json'))

def load_model(model, save_path):
    ckpt_path = os.path.join(save_path, 'pytorch_model.bin')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

def main():
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            n_gpu = 0
        else:
            device = torch.device(args.device)
            n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.mode not in ['train', 'test']:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.exp_path) and os.listdir(args.exp_path):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.exp_path))
    # os.makedirs(args.exp_path, exist_ok=True)
    if args.mode == 'train' and not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)

    db = DB(args.db_path, args.tfidf_path)

    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    add_special_tokens(tokenizer)
    if args.mode == 'train':
        model = UBAR_plus(args, tokenizer, device)
    else:
        model = UBAR_plus(args, tokenizer, device)
        load_model(model, args.eval_load_path)
    model.to(device)
    optim = Optim(learning_rate=args.learning_rate)
    writer = SummaryWriter(log_dir='./log')

    # prepare dataset
    train_dataset = None
    num_train_steps = None
    if args.mode == 'train':
        from reader.Dataset import UBARDataset
        train_dataset = UBARDataset(args, tokenizer, args.data_dir, 'train', db)
        dev_dataset = UBARDataset(args, tokenizer, args.data_dir, 'dev', db)
        num_train_steps = int(len(train_dataset) * args.num_train_epochs / args.train_batch_size / args.gradient_accumulation_steps)
        optimizer, scheduler = optim.get_optimizer_scheduler(model, num_train_steps)

        logger.info('start training...')

        # train
        model.train()
        train(args,
              dataloader=train_dataset,
              dev_dataloader=dev_dataset,
              model=model,
              tokenizer=tokenizer,
              writer=writer,
              optimizer=optimizer,
              scheduler=scheduler,
              num_train_steps=num_train_steps,
              device=device)

    elif args.mode == 'test':
        from reader.Dataset import UBARDataset
        test_data = UBARDataset(args, tokenizer, args.data_dir, 'test', db)
        model.eval()
        validate(args, test_data, model, tokenizer, db, device)

if __name__ == '__main__':
    main()
