import logging
import random
import torch
import numpy as np
import os

from src.functions.processor import (
    CLASSIFIERProcessor,
    convert_examples_to_features
)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# tensor를 list 형으로 변환하기위한 함수
def to_list(tensor):
    return tensor.detach().cpu().tolist()


# dataset을 load 하는 함수
def load_examples(args, tokenizer, evaluate=False, output_examples=False, do_predict=False, input_dict=None):
    '''

    :param args: 하이퍼 파라미터
    :param tokenizer: tokenization에 사용되는 tokenizer
    :param evaluate: 평가나 open test시, True
    :param output_examples: 평가나 open test 시, True / True 일 경우, examples와 features를 같이 return
    :param do_predict: open test시, True
    :param input_dict: open test시 입력되는 문서와 질문으로 이루어진 dictionary
    :return:
    examples : max_length 상관 없이, 원문으로 각 데이터를 저장한 리스트
    features : max_length에 따라 분할 및 tokenize된 원문 리스트
    dataset : max_length에 따라 분할 및 학습에 직접적으로 사용되는 tensor 형태로 변환된 입력 ids
    '''
    input_dir = args.data_dir
    print("Creating features from dataset file at {}".format(input_dir))

    # processor 선언
    ## json으로 된 train과 dev data_file명
    processor = CLASSIFIERProcessor()

    # open test 시
    if do_predict:
        ## input_dict: guid, premise, hypothesis로 이루어진 dictionary
        # examples = processor.get_example_from_input(input_dict)
        examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                              filename=args.predict_file)
    # 평가 시
    elif evaluate:
        examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                              filename=args.eval_file)
    # 학습 시
    else:
        examples = processor.get_train_examples(os.path.join(args.data_dir),
                                                filename=args.train_file)

    # features = (prem_features, hypo_features)
    features, dataset = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
        max_sentence_length = args.max_sentence_length,
        language = args.model_name_or_path.split("/")[-2]
    )
    if output_examples:
        ## example == feature == dataset
        return dataset, examples, features
    return dataset
