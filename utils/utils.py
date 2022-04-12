import logging

import numpy as np
import torch
from torch.autograd import Variable

from config import BIO_TAG


def log_first_inputs(log_dict):
    logging.info("**** Input Examples ****")
    for key, context in log_dict.items():
        logging.info(key + ': ' + context)


def padSeqs_gpt(sequences, pad_id, maxlen=None):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)

    # maxlen = 1024
    if seq_mexlen > 1024:  # gpt2.n_ctx
        # print('maxlen exceeds 1024')
        maxlen = 1024
    else:
        maxlen = seq_mexlen

    # tokenizer.encode('<|endoftext|>') = ['50256']
    # All labels set to ``-100`` are ignored (masked), the loss is only
    # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)

    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        # trunc method = 'pre'
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)

        # pad method = 'post'
        x[idx, :len(trunc)] = trunc

    return x, lengths


def padSeqs(sequences, maxlen=None, truncated = False, pad_method='post',
                     trunc_method='pre', dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    if maxlen is not None and truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x

def maskedNll(seq, target, pad_id=0):
    """
    Compute the Cross Entropy Loss of ground truth (target) sentence given the model
    S: <START>, E: <END>, W: word token, 1: padding token, P(*): logProb
    Teacher forced logProbs (seq):
        [P(W1) P(W2) P(E) -   -   -]
    Required gtSeq (target):
        [  W1    W2    E  1   1   1]
    Mask (non-zero tokens in target):
        [  1     1     1  0   0   0]

    """
    # Generator a mask of non-padding (non-zero) tokens
    mask = target.data.ne(pad_id)
    loss = 0
    assert isinstance(target, Variable)
    if isinstance(target, Variable):
        mask = Variable(mask, volatile=target.volatile)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    maskedNLL = torch.masked_select(gtLogProbs, mask)
    nll_loss = -torch.sum(maskedNLL) / seq.size(1)
    return nll_loss