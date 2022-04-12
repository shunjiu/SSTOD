import argparse

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--data_dir",
                    default='./data/',
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_path", default='uer/gpt2-chinese-cluecorpussmall', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--mode",
                    default=None,
                    type=str,
                    # required=True,
                    help="The name of the task to train.")
parser.add_argument("--exp_path",
                    default=None,
                    type=str,
                    # required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--db_path",
                    type=str,
                    default='./data/database.json',
                    help="The dictionary of database"
                    )
parser.add_argument("--tfidf_path",
                    type=str,
                    default='./utils/tfidf/',
                    help="The dictionary where the vector of knowledge tuple will be written"
                    )
parser.add_argument("--eval_load_path",
                    type=str,
                    help="The path of saved model")

# setting
parser.add_argument("--embedding_dim", type=int, default=768)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument('--hop', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--use_true_curr_kdpn', type=bool, default=False)
parser.add_argument('--use_true_curr_kp', type=bool, default=False)
parser.add_argument('--use_true_db_pointer', type=bool, default=True)
parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cuda:2', 'cpu'], default='cuda:0')

# Other parameters
parser.add_argument("--n_best", type=int, default=60)
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_lower_case",
                    default=False,
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=2,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=1e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=60,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--report_interval",
                    default=400,
                    type=int,
                    help="Step of reporting train loss")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=10,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=16,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--clip_grad_norm',
                    type=float,
                    default=5.0,
                    help="Parameter of clip grad norm")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--evaluate_during_training',
                    type=bool, default=True)



BIO_TAG = {
    'B': 1,
    'I': 2,
    'O': 3,
    '<pad>': 0
}