import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import transformers
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.data.processors.utils import DataProcessor

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)



def convert_example_to_features(example, max_seq_length, is_training, max_sentence_length, language):

    # 데이터의 유효성 검사를 위한 부분
    # ========================================================
    label = None
    coarse_label = None
    if is_training:
        # Get label
        label = example.label
        coarse_label = example.coarse_label

        # label_dictionary에 주어진 label이 존재하지 않으면 None을 feature로 출력
        # If the label cannot be found in the text, then skip this example.
        ## kind_of_label: label의 종류
        kind_of_label = ["문제 정의", "가설 설정", "기술 정의", "제안 방법", "대상 데이터", "데이터처리", "이론/모형", "성능/효과", "후속연구"]#, "기타"]
        actual_text = kind_of_label[label] if label<=len(kind_of_label) else label
        if actual_text not in kind_of_label:
            logger.warning("Could not find label: '%s' \n not in label list", actual_text)
            return None

        kind_of_coarse_label = ["연구 목적", "연구 방법", "연구 결과"]#, "기타"]
        actual_text = kind_of_coarse_label[coarse_label] if coarse_label <= len(kind_of_coarse_label) else coarse_label
        if actual_text not in kind_of_coarse_label:
            logger.warning("Could not find coarse_label: '%s' \n not in coarse_label list", actual_text)
            return None

    # ========================================================

    # 단어(어절;word)와 토큰 간의 위치 정보 확인
    tok_to_orig_index = {"sentence": []}  # token 개수만큼 # token에 대한 word의 위치
    orig_to_tok_index = {"sentence": []} # origin 개수만큼 # word를 토큰화하여 나온 첫번째 token의 위치
    all_doc_tokens = {"sentence": []} # origin text를 tokenization
    token_to_orig_map = {"sentence": []}

    for case in example.merge.keys():
        new_merge = []
        new_word = []
        idx = 0
        for merge_idx in example.merge[case]:
            for m_idx in merge_idx:
                new_word.append(example.doc_tokens[case][m_idx])
            new_word.append("<WORD>")
            merge_idx = [m_idx+idx for m_idx in range(0,len(merge_idx))]
            new_merge.append(merge_idx)
            idx = max(merge_idx)+1
            new_merge.append([idx])
            idx+=1
        example.merge[case] = new_merge
        example.doc_tokens[case] = new_word

    for case in example.merge.keys():
        for merge_idx in example.merge[case]:
            for word_idx in merge_idx:
                # word를 토큰화하여 나온 첫번째 token의 위치
                orig_to_tok_index[case].append(len(tok_to_orig_index[case]))
                if (example.doc_tokens[case][word_idx] == "<WORD>"):
                    sub_tokens = ["<WORD>"]
                else: sub_tokens = tokenizer.tokenize(example.doc_tokens[case][word_idx])
                for sub_token in sub_tokens:
                    # token 저장
                    all_doc_tokens[case].append(sub_token)
                    # token에 대한 word의 위치
                    tok_to_orig_index[case].append(word_idx)
                    # token_to_orig_map: {token:word}
                    #token_to_orig_map[case][len(tok_to_orig_index[case]) - 1] = len(orig_to_tok_index[case]) - 1
                    token_to_orig_map[case].append(len(orig_to_tok_index[case]) - 1)

    # print("tok_to_orig_index\n"+str(tok_to_orig_index))
    # print("orig_to_tok_index\n"+str(orig_to_tok_index))
    # print("all_doc_tokens\n"+str(all_doc_tokens))
    # print("token_to_orig_map\n\tindex of token : index of word\n\t"+str(token_to_orig_map))

    # =========================================================
    if language == "bert":
        ## 최대 길이 넘는지 확인
        if int(transformers.__version__[0]) <= 3:
            assert len(all_doc_tokens["sentence"]) + 2 <= tokenizer.max_len
        else:
            assert len(all_doc_tokens["sentence"]) + 2 <= tokenizer.model_max_length

    input_ids = [tokenizer.cls_token_id]
    if language == "KorSciBERT":
        input_ids += sum([tokenizer.convert_tokens_to_ids([token]) for token in all_doc_tokens["sentence"]], [])
        word_idxs = [0] + list(filter(lambda x: input_ids[x] == tokenizer.convert_tokens_to_ids(["<WORD>"])[0], range(len(input_ids))))
    else:
        input_ids += [tokenizer.convert_tokens_to_ids(token) for token in all_doc_tokens["sentence"]]
        word_idxs = [0] + list(filter(lambda x: input_ids[x] == tokenizer.convert_tokens_to_ids("<WORD>"), range(len(input_ids))))

    input_ids += [tokenizer.sep_token_id]

    token_type_ids = [0] * len(input_ids)

    position_ids = list(range(0, len(input_ids)))

    # non_padded_ids: padding을 제외한 토큰의 index 번호
    non_padded_ids = [i for i in input_ids]

    # tokens: padding을 제외한 토큰
    non_padded_tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

    attention_mask = [1]*len(input_ids)

    paddings = [tokenizer.pad_token_id]*(max_seq_length - len(input_ids))

    if tokenizer.padding_side == "right":
        input_ids += paddings
        attention_mask += [0]*len(paddings)
        token_type_ids += paddings
        position_ids += paddings
    else:
        input_ids  = paddings + input_ids
        attention_mask = [0]*len(paddings) + attention_mask
        token_type_ids = paddings + token_type_ids
        position_ids = paddings + position_ids

        word_idxs = [x+len(paddings) for x in word_idxs]

    # """
    # mean pooling
    not_word_list = []
    for k, p_idx in enumerate(word_idxs[1:]):
        not_word_idxs = [0] * len(input_ids);
        for j in range(word_idxs[k] + 1, p_idx):
            not_word_idxs[j] = 1 / (p_idx - word_idxs[k] - 1)
        not_word_list.append(not_word_idxs)
    not_word_list = not_word_list + [[0] * len(input_ids)] * (
                max_sentence_length - len(not_word_list))


    """
    # (a,b, |a-b|, a*b)
    not_word_list = [[], []]
    for k, p_idx in enumerate(word_idxs[1:]):
        not_word_list[0].append(word_idxs[k] + 1)
        not_word_list[1].append(p_idx - 1)
    not_word_list[0] = not_word_list[0] + [int(word_idxs[-1]+i+2) for i in range(0, (max_sentence_length - len(not_word_list)))]
    not_word_list[1] = not_word_list[1] + [int(word_idxs[-1] + i + 2) for i in range(0, (max_sentence_length - len(pnot_word_list)))]
    """

    # p_mask: mask with 0 for token which belong premise and hypothesis including CLS TOKEN
    #           and with 1 otherwise.
    # Original TF implem also keep the classification token (set to 0)
    p_mask = np.ones_like(token_type_ids)
    if tokenizer.padding_side == "right":
        # [CLS] P [SEP] H [SEP] PADDING
        p_mask[:len(all_doc_tokens["sentence"]) + 1] = 0
    else:
        p_mask[-(len(all_doc_tokens["sentence"])  + 1): ] = 0

    # pad_token_indices: input_ids에서 padding된 위치
    pad_token_indices = np.array(range(len(non_padded_ids), len(input_ids)))
    # special_token_indices: special token의 위치
    special_token_indices = np.asarray(
        tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    ).nonzero()

    p_mask[pad_token_indices] = 1
    p_mask[special_token_indices] = 1

    # Set the cls index to 0: the CLS index can be used for impossible answers
    # Identify the position of the CLS token
    cls_index = input_ids.index(tokenizer.cls_token_id)

    p_mask[cls_index] = 0

    # dependency  = [[tail, head, dependency], [], ...]
    if example.dependency["sentence"] == [[]]:
        example.dependency["sentence"] = [[max_sentence_length-1,max_sentence_length-1,0] for _ in range(0,max_sentence_length)]
    else:
        example.dependency["sentence"] = example.dependency["sentence"] + [[max_sentence_length-1,max_sentence_length-1,0] for i in range(0, abs(max_sentence_length-len(example.dependency["sentence"])))]

    dependency = example.dependency["sentence"]

    return  CLASSIFIERFeatures(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            cls_index,
            p_mask.tolist(),
            example_index=0,
            tokens=non_padded_tokens,
            token_to_orig_map=token_to_orig_map,
            label = label,
            coarse_label = coarse_label,
            doc_id = example.doc_id,
            language = language,
            dependency = dependency,
            not_word_list = not_word_list,
        )



def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        is_training,
        return_dataset=False,
        threads=1,
        max_sentence_length = 0,
        tqdm_enabled=True,
        language = None,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:

        # annotate_ = 하나의 example에 대한 여러 feature를 리스트로 모은 것
        # annotate_ = list(feature1, feature2, ...)
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            max_sentence_length=max_sentence_length,
            is_training=is_training,
            language = language,
        )

        # examples에 대한 annotate_
        # features = list( feature1, feature2, feature3, ... )
        ## len(features) == len(examples)
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert bert examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    example_index = 0  # example의 id  ## len(features) == len(examples)
    for example_feature in tqdm(
            features, total=len(features), desc="add example index", disable=not tqdm_enabled
    ):
        if not example_feature:
            continue

        example_feature.example_index = example_index
        new_features.append(example_feature)
        example_index += 1

    features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        ## RoBERTa doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment.
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)

        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # 전체 feature의 개별 index

        # all_dependency  = [[[premise_tail, premise_head, dependency], [], ...],[[hypothesis_tail, hypothesis_head, dependency], [], ...]], [[],[]], ... ]
        all_dependency = torch.tensor([f.dependency for f in features], dtype=torch.long)

        all_not_word_list = torch.tensor([f.not_word_list for f in features], dtype=torch.float)

        if not is_training:
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks, all_token_type_ids, all_position_ids,
                all_cls_index, all_p_mask, all_feature_index,
                all_dependency,
                all_not_word_list
            )
        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            all_coarse_labels = torch.tensor([f.coarse_label for f in features], dtype=torch.long)
            # label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
            # all_labels = torch.tensor([label_dict[f.label] for f in features], dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_position_ids,
                all_labels,
                all_coarse_labels,
                all_cls_index,
                all_p_mask,
                all_example_indices,
                all_feature_index,
                all_dependency,
                all_not_word_list
            )

        return features, dataset
    else:
        return features

class CLASSIFIERProcessor(DataProcessor):
    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            gold_label = None
            gold_coarse_label = None
            label = tensor_dict["tag"].numpy().decode("utf-8")
            coarse_label = tensor_dict["coarse_tag"].numpy().decode("utf-8")
        else:
            gold_label = tensor_dict["tag"].numpy().decode("utf-8")
            gold_coarse_label = tensor_dict["coarse_tag"].numpy().decode("utf-8")
            label = None
            coarse_label = None

        return CLASSIFIERExample(
            doc_id=tensor_dict["doc_id"].numpy().decode("utf-8"),
            # sentid=tensor_dict["sentid"].numpy().decode("utf-8"),
            sentence=tensor_dict["sentence"].numpy().decode("utf-8"),
            preprocess=tensor_dict["preprocess"].numpy().decode("utf-8"),
            parsing=tensor_dict["merge"]["parsing"].numpy().decode("utf-8"),
            keysentnece=tensor_dict["keysentence"].numpy().decode("utf-8"),
            label=label,
            coarse_label=coarse_label,
            gold_label=gold_label,
            gold_coarse_label = gold_coarse_label
            )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.CLASSIFIERExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of CLASSIFIERExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.

        """
        if data_dir is None:
            data_dir = ""

        #if self.train_file is None:
        #    raise ValueError("CLASSIFIERProcessor should be instantiated via CLASSIFIERV1Processor.")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, 'train', self.train_file if filename is None else filename)

    def get_dev_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.
        """
        if data_dir is None:
            data_dir = ""

        #if self.dev_file is None:
        #    raise ValueError("CLASSIFIERProcessor should be instantiated via CLASSIFIERV1Processor.")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev", self.dev_file if filename is None else filename)

    def get_example_from_input(self, input_dictionary):

        doc_id = input_dictionary["doc_id"]
        keysentnece = input_dictionary["keysentnece"]
        # sentid = input_dictionary["sentid"]
        sentence = input_dictionary["sentence"]

        label = None
        coarse_label = None
        gold_label = None
        gold_coarse_label = None

        examples = [CLASSIFIERExample(
            doc_id=doc_id,
            # sentid=sentid,
            keysentence = keysentnece,
            sentence=sentence,
            gold_label=gold_label,
            gold_coarse_label=gold_coarse_label,
            label=label,
            coarse_label=coarse_label,
        )]
        return examples

    def _create_examples(self, input_data, set_type, data_file):
        is_training = set_type == "train"
        num = 0
        examples = []
        for entry in tqdm(input_data):

            doc_id = entry["doc_id"]
            # sentid = entry["sentid"]
            sentence = entry["sentence"]
            preprocess = entry["preprocess"]
            merge = entry["merge"]["origin"]
            parsing = entry["merge"]["parsing"]
            keysentence= entry["keysentence"]

            label = None
            coarse_label = None
            gold_label = None
            gold_coarse_label = None
            if is_training:
                label = entry["tag"]
                coarse_label = entry["coarse_tag"]
            else:
                gold_label = entry["tag"]
                gold_coarse_label = entry["coarse_tag"]



            example = CLASSIFIERExample(
                doc_id=doc_id,
                # sentid=sentid,
                keysentence=keysentence,
                sentence=sentence,
                preprocess=preprocess,
                parsing=parsing,
                merge=merge,
                gold_label=gold_label,
                gold_coarse_label=gold_coarse_label,
                label=label,
                coarse_label=coarse_label,
            )
            examples.append(example)
        # len(examples) == len(input_data)
        return examples


class CLASSIFIERV1Processor(CLASSIFIERProcessor):
    train_file = "train.json"
    dev_file = "dev.json"


class CLASSIFIERExample(object):
    def __init__(
        self,
            doc_id,
            # sentid,
            sentence,
            preprocess,
            parsing,
            merge,
            keysentence,
            gold_label=None,
            gold_coarse_label=None,
            label=None,
            coarse_label=None,
    ):
        self.doc_id = doc_id
        # self.sentid = sentid
        self.keysentence = keysentence
        self.sentence = sentence
        self.preprocess = preprocess
        self.parsing = parsing
        self.merge = merge

        label_dict = {'문제 정의': 0, '가설 설정': 1, '기술 정의': 2, '제안 방법': 3, '대상 데이터': 4, '데이터처리': 5, '이론/모형': 6, '성능/효과': 7, '후속연구': 8, '기타': 9}
        coarse_label_dict = {'연구 목적': 0, '연구 방법': 1, '연구 결과': 2, '기타': 3}
        if gold_label in label_dict.keys():
            gold_label = label_dict[gold_label]
        if gold_coarse_label in coarse_label_dict.keys():
            gold_coarse_label = coarse_label_dict[gold_coarse_label]
        self.gold_label = gold_label
        self.gold_coarse_label = gold_coarse_label

        if coarse_label in coarse_label_dict.keys():
            coarse_label = coarse_label_dict[coarse_label]
        if label in label_dict.keys():
            label = label_dict[label]
        self.label = label
        self.coarse_label = coarse_label

        # doct_tokens : 띄어쓰기 기준으로 나누어진 어절(word)로 만들어진 리스트
        ##      sentence1                   sentence2
        self.doc_tokens = {"sentence":self.preprocess.strip().split()}

        # merge: 말뭉치의 시작위치를 어절 기준으로 만든 리스트
        merge_word = []; check_merge_word = []
        merge_index = []
        for merge in self.merge:
            if merge != []: merge_index.append(merge[1])

        # 구문구조 종류
        depend2idx = {"None":0}; idx2depend ={0:"None"}
        for depend1 in ['DP', 'L', 'NP', 'IP', 'PAD', 'VP', 'VNP', 'X', 'AP']:
            for depend2 in ['MOD', 'OBJ', 'CNJ', 'CMP', 'SBJ', 'None', 'AJT']:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2

        if ([words for words in self.parsing if words[2][0] != words[2][1]] == []): merge_word.append([])
        else:
            for words in self.parsing:
                if words[2][0] != words[2][1]:
                    w1 = merge_index.index(words[1][0])
                    w2 = merge_index.index(words[1][1])
                    dep = depend2idx["-".join(words[2])]
                    if [w1,w2] not in check_merge_word:
                        check_merge_word.append([w1, w2])
                        merge_word.append([w1,w2,dep])
                    else:
                        check_index = check_merge_word.index([w1,w2])
                        now_dep = idx2depend[merge_word[check_index][2]].split("-")[1]
                        if (words[2][1] in ['SBJ', 'CNJ', 'OBJ']) and(now_dep in ['CMP', 'MOD', 'AJT', 'None', "UNDEF"]):
                            merge_word[check_index][2] = dep

        del check_merge_word
        self.merge = {"sentence":merge_index}
        self.dependency = {"sentence":merge_word}

class CLASSIFIERFeatures(object):
    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            cls_index,
            p_mask,
            example_index,
            token_to_orig_map,
            doc_id,
            tokens,
            label,
            coarse_label,
            language,
            dependency,
            not_word_list,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.token_to_orig_map = token_to_orig_map
        self.doc_id = doc_id
        self.tokens = tokens

        self.label = label
        self.coarse_label = coarse_label

        self.dependency = dependency

        self.not_word_list = not_word_list


class KLUEResult(object):
    def __init__(self, example_index, label_logits, gold_label=None, cls_logits=None):
        self.label_logits = label_logits
        self.example_index = example_index

        if gold_label:
            self.gold_label = gold_label
            self.cls_logits = cls_logits
