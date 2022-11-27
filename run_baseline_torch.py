import argparse
import os
import logging
from attrdict import AttrDict

# bert
from transformers import AutoTokenizer
from transformers import BertConfig,RobertaConfig

from src.model.model import BertForSequenceClassification, RobertaForSequenceClassification
#from src.model.model_multi import BertForSequenceClassification, RobertaForSequenceClassification

from src.model.main_functions import train, evaluate, predict
#from src.model.main_functions_multi import train, evaluate, predict

from src.functions.utils import init_logger, set_seed

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def create_model(args):

    if args.model_name_or_path.split("/")[-2] == "bert":

        # 모델 파라미터 Load
        config = BertConfig.from_pretrained(
            args.model_name_or_path#'bert/Bert-base'
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
        )

        config.num_coarse_labels = 3
        config.num_labels = 9

        # roberta attention 추출하기
        config.output_attentions=True

        # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
        # BertTokenizerFast로 되어있음
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path#'bert/init_weight'
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,
        )

        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path#'bert/init_weight'
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
            config=config,
            max_sentence_length=args.max_sentence_length,
            # from_tf= True if args.from_init_weight else False
        )
        args.model_name_or_path = args.cache_dir
        # print(tokenizer.convert_tokens_to_ids("<WORD>"))

    elif args.model_name_or_path.split("/")[-2] == "roberta":

        # 모델 파라미터 Load
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
        )

        config.num_coarse_labels = 3
        config.num_labels = 9

        # roberta attention 추출하기
        config.output_attentions=True

        # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
        # BertTokenizerFast로 되어있음
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,
        )

        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path
            if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            cache_dir=args.cache_dir,
            config=config,
            max_sentence_length=args.max_sentence_length,
            # from_tf= True if args.from_init_weight else False
        )
        args.model_name_or_path = args.cache_dir
        # print(tokenizer.convert_tokens_to_ids("<WORD>"))



    # vocab 추가
    # 중요 단어의 UNK 방지 및 tokenize를 방지해야하는 경우(HTML 태그 등)에 활용
    # "세종대왕"이 OOV인 경우 ['세종대왕'] --tokenize-->  ['UNK'] (X)
    # html tag인 [td]는 tokenize가 되지 않아야 함. (완전한 tag의 형태를 갖췄을 때, 의미를 갖기 때문)
    #                             ['[td]'] --tokenize-->  ['[', 't', 'd', ']'] (X)

    if args.from_init_weight and args.add_vocab:
        if args.from_init_weight:
            add_token = {
                "additional_special_tokens": ["[td]", "추가 단어 1", "추가 단어 2"]}
            # 추가된 단어는 tokenize 되지 않음
            # ex
            # '[td]' vocab 추가 전 -> ['[', 't', 'd', ']']
            # '[td]' vocab 추가 후 -> ['[td]']
            tokenizer.add_special_tokens(add_token)
            model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    return model, tokenizer

def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    model, tokenizer = create_model(args)

    # Running mode에 따른 실행
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        evaluate(args, model, tokenizer, logger)
    elif args.do_predict:
        predict(args, model, tokenizer)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # Directory

    #------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--data_dir", type=str, default="./data")
    cli_parser.add_argument("--data_dir", type=str, default="./data/origin/merge_origin_preprocess")
    #cli_parser.add_argument("--data_dir", type=str, default="./data/origin/merge_origin_preprocess_woCNJ")

    cli_parser.add_argument("--train_file", type=str, default='origin_train.json')
    #cli_parser.add_argument("--train_file", type=str, default='sample.json')
    cli_parser.add_argument("--eval_file", type=str, default='origin_test.json')
    cli_parser.add_argument("--predict_file", type=str, default='origin_test.json')

    # ------------------------------------------------------------------------------------------------

    # roberta
    cli_parser.add_argument("--model_name_or_path", type=str, default="./roberta/init_weight")
    cli_parser.add_argument("--cache_dir", type=str, default="./roberta/init_weight")

    # # bert
    # cli_parser.add_argument("--model_name_or_path", type=str, default="./bert/init_weight")
    # cli_parser.add_argument("--cache_dir", type=str, default="./bert/init_weight")

    #------------------------------------------------------------------------------------------------------------

    # cli_parser.add_argument("--output_dir", type=str, default="./bert/biaffine_model/origin/baseline/wCNJ")   # checkout-5  f1-score: 89.62 acc: 89.38
    cli_parser.add_argument("--output_dir", type=str, default="./bert/biaffine_model/origin/baseline/woCNJ")  # checkout-4  f1-score: 89.61 acc: 89.39

    # cli_parser.add_argument("--output_dir", type=str, default="./bert/baseline/wCNJ")

    #cli_parser.add_argument("--output_dir", type=str, default="./bert/biaffine_model/origin/multi/together/wCNJ")  # checkout-5 f1-score:89.62  acc:89.39
    #cli_parser.add_argument("--output_dir", type=str, default="./bert/biaffine_model/origin/multi/together/woCNJ") # checkout-5 f1-score: 89.53 acc: 89.29

    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/baseline/")
    #cli_parser.add_argument("--output_dir", type=str, default="./roberta/biaffine_model/multi/")

    # ------------------------------------------------------------------------------------------------------------

    cli_parser.add_argument("--max_sentence_length", type=int, default=110)

    # https://github.com/KLUE-benchmark/KLUE-baseline/blob/main/run_all.sh
    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=1e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default =16)
    cli_parser.add_argument("--eval_batch_size", type=int, default = 32)
    cli_parser.add_argument("--num_train_epochs", type=int, default=6)

    #cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--logging_steps", type=int, default=100)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default= False)#True)
    cli_parser.add_argument("--add_vocab", type=bool, default=False)

    cli_parser.add_argument("--checkpoint", type=str, default="6")

    cli_parser.add_argument("--do_train", type=bool, default=False)#True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)#True)#False)
    cli_parser.add_argument("--do_predict", type=bool, default=True)#False)

    cli_args = cli_parser.parse_args()

    main(cli_args)
