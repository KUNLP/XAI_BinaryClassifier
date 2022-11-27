import os
import numpy as np
import pandas as pd
import torch
import timeit
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.file_utils import is_torch_available

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src.functions.utils import load_examples, set_seed, to_list
from src.functions.metric import get_score, get_ai_score, get_sklearn_score

from sklearn.metrics import confusion_matrix
from functools import partial

def train(args, model, tokenizer, logger):
    max_f1 =0.89
    max_acc = 0.89
    # 학습에 사용하기 위한 dataset Load
    ## dataset: tensor형태의 데이터셋
    ## all_input_ids,
        # all_attention_masks,
        # all_labels,
        # all_cls_index,
        # all_p_mask,
        # all_example_indices,
        # all_feature_index

    train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # t_total: total optimization step
    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",  args.train_batch_size  * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    if not args.from_init_weight: global_step += int(args.checkpoint)

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    epoch_idx=0
    if not args.from_init_weight: epoch_idx += int(args.checkpoint)

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # train 모드로 설정
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # 모델에 입력할 입력 tensor 저장
            inputs_list = ["input_ids", "attention_mask","token_type_ids","position_ids"]
            inputs_list.append("labels")
            inputs_list.append("coarse_labels")
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]
            inputs_list2 = ['word_idxs', 'span']
            for m, input in enumerate(inputs_list2):
                inputs[input] = batch[-(m+1)]

            # Loss 계산 및 저장
            ## outputs = (total_loss,) + outputs
            outputs = model(**inputs)
            loss = outputs[0]

            # 높은 batch size는 학습이 진행하는 중에 발생하는 noisy gradient가 경감되어 불안정한 학습을 안정적이게 되도록 해줌
            # 높은 batch size 효과를 주기위한 "gradient_accumulation_step"
            ## batch size *= gradient_accumulation_step
            # batch size: 16
            # gradient_accumulation_step: 2 라고 가정
            # 실제 batch size 32의 효과와 동일하진 않지만 비슷한 효과를 보임
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                

            ## batch_size의 개수만큼의 데이터를 입력으로 받아 만들어진 모델의 loss는
            ## 입력 데이터들에 대한 특징을 보유하고 있다(loss를 어떻게 만드느냐에 따라 달라)
            ### loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction = ?)
            ### reduction = mean : 입력 데이터에 대한 평균
            loss.backward()
            tr_loss += loss.item()


            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step+1),loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        epoch_idx += 1
        logger.info("***** Eval results *****")
        results = evaluate(args, model, tokenizer, logger, epoch_idx = str(epoch_idx), tr_loss = loss.item())

        # model save
        if ((max_acc < float(results["accuracy"])) or (
                max_f1 < float(results["macro_f1_score" if "macro_f1_score" in results.keys() else "macro_f1"]))):
            if max_acc < float(results["accuracy"]): max_acc = float(results["accuracy"])
            if max_f1 < float(
                results["macro_f1_score" if "macro_f1_score" in results.keys() else "macro_f1"]): max_f1 = float(
                results["macro_f1_score" if "macro_f1_score" in results.keys() else "macro_f1"])

            # 모델 저장 디렉토리 생성
            output_dir = os.path.join(args.output_dir, "model/checkpoint-{}".format(epoch_idx))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 학습된 가중치 및 vocab 저장
            ## pretrained 모델같은 경우 model.save_pretrained(...)로 저장
            ## nn.Module로 만들어진 모델일 경우 model.save(...)로 저장
            ### 두개가 모두 사용되는 모델일 경우 이 두가지 방법으로 저장을 해야한다!!!!
            model.save_pretrained(output_dir)
            if (args.model_name_or_path.split("/")[-2] != "KorSciBERT"): tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

        mb.write("Epoch {} done".format(epoch + 1))

    return global_step, tr_loss / global_step

# 정답이 사전부착된 데이터로부터 평가하기 위한 함수
def evaluate(args, model, tokenizer, logger, epoch_idx = "", tr_loss = 1):
    # 데이터셋 Load
    ## dataset: tensor형태의 데이터셋
    ## example: json형태의 origin 데이터셋
    ## features: index번호가 추가된 list형태의 examples 데이터셋
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(epoch_idx))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()

    # 예측 라벨
    pred_logits = torch.tensor([], dtype = torch.long).to(args.device)
    pred_coarse_logits = torch.tensor([], dtype = torch.long).to(args.device)
    for batch in progress_bar(eval_dataloader):
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs_list = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            inputs_list2 = ['word_idxs', 'span']
            for m, input in enumerate(inputs_list2): inputs[input] = batch[-(m + 1)]

            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits,outputs[0][1]], dim = 0)
        pred_coarse_logits = torch.cat([pred_coarse_logits, outputs[0][0]], dim=0)

    # pred_label과 gold_label 비교
    pred_logits= pred_logits.detach().cpu().numpy()
    pred_coarse_logits= pred_coarse_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    pred_coarse_labels = np.argmax(pred_coarse_logits, axis=-1)
    ## gold_labels
    gold_labels = [example.gold_label for example in examples]
    gold_coarse_labels = [example.gold_coarse_label for example in examples]

    # print('\n\n=====================outputs=====================')
    # for g,p in zip(gold_labels, pred_labels):
    #     print(str(g)+"\t"+str(p))
    # print('===========================================================')

    # 평가 시간 측정을 위한 time 변수
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # 최종 예측값과 원문이 저장된 example로 부터 성능 평가
    ## results  = {"macro_precision":round(macro_precision, 4), "macro_recall":round(macro_recall, 4), "macro_f1_score":round(macro_f1_score, 4), \
    ##        "accuracy":round(total_accuracy, 4), \
    ##       "micro_precision":round(micro_precision, 4), "micro_recall":round(micro_recall, 4), "micro_f1":round(micro_f1_score, 4)}
    idx2label = {0: '문제 정의', 1: '가설 설정', 2: '기술 정의', 3: '제안 방법', 4: '대상 데이터', 5: '데이터처리', 6: '이론/모형', 7: '성능/효과', 8: '후속연구'}# , 9: '기타'}
    idx2coarse_label = {0: '연구 목적', 1: '연구 방법', 2: '연구 결과'}  # , 3: '기타'}

    # results = get_score(pred_labels, gold_labels, idx2label)
    results = get_sklearn_score(pred_labels, gold_labels, idx2label)
    coarse_results = get_sklearn_score(pred_coarse_labels, gold_coarse_labels, idx2coarse_label)

    output_dir = os.path.join( args.output_dir, 'eval')

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type ='w'

    # 평가 스크립트 기반 성능 저장을 위한 파일 생성
    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else:
        eval_file_name = "init_weight"
    output_eval_file = os.path.join(output_dir, "eval_result_{}.txt".format(eval_file_name))

    with open(output_eval_file, out_file_type, encoding='utf-8') as f:
        f.write("train loss: {}\n".format(tr_loss))
        f.write("epoch: {}\n".format(epoch_idx))
        f.write("세부분류 성능\n")
        for k in results.keys():
            f.write("{} : {}\n".format(k, results[k]))
        f.write("\n대분류 성능\n")
        for k in coarse_results.keys():
            f.write("{} : {}\n".format(k, coarse_results[k]))

        confusion_m = confusion_matrix(pred_labels, gold_labels)
        confusion_list = [[], [0 for i in range(0, len(confusion_m))], []]
        for i in range(0, len(confusion_m)):
            for j in range(0, len(confusion_m[i])):
                if (i == j): confusion_list[0].append(confusion_m[i][j])
        all_cnt = sum([sum(i) for i in confusion_m])
        f.write("micro_accuracy: " + str(round((sum(confusion_list[0]) / all_cnt), 4)) +"\n")
        print("micro_accuracy: " + str(round((sum(confusion_list[0]) / all_cnt), 4)))

        f.write("=======================================\n\n")
    return results

def predict(args, model, tokenizer):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True, do_predict=True)

    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizing 된 데이터를 batch size만큼 가져오기 위한 random sampler 및 DataLoader
    ## RandomSampler: 데이터 index를 무작위로 선택하여 조정
    ## SequentialSampler: 데이터 index를 항상 같은 순서로 조정
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    print("***** Running Prediction *****")
    print("  Num examples = %d", len(dataset))

    # 예측 라벨
    pred_coarse_logits = torch.tensor([], dtype=torch.long).to(args.device)
    pred_logits = torch.tensor([], dtype=torch.long).to(args.device)
    for batch in progress_bar(eval_dataloader):
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs_list = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]
            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            inputs_list2 = ['word_idxs', 'span']
            for m, input in enumerate(inputs_list2): inputs[input] = batch[-(m + 1)]

            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits, outputs[0][1]], dim=0)
        pred_coarse_logits = torch.cat([pred_coarse_logits, outputs[0][0]], dim=0)

    # pred_label과 gold_label 비교
    pred_logits = pred_logits.detach().cpu().numpy()
    pred_coarse_logits = pred_coarse_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    pred_coarse_labels = np.argmax(pred_coarse_logits, axis=-1)
    ## gold_labels
    gold_labels = [example.gold_label for example in examples]
    gold_coarse_labels = [example.gold_coarse_label for example in examples]

    idx2label = {0: '문제 정의', 1: '가설 설정', 2: '기술 정의', 3: '제안 방법', 4: '대상 데이터', 5: '데이터처리', 6: '이론/모형', 7: '성능/효과', 8: '후속연구'}#, 9: '기타'}
    idx2coarse_label = {0: '연구 목적', 1: '연구 방법', 2: '연구 결과'}  # , 3: '기타'}

    # results = get_score(pred_labels, gold_labels, idx2label)
    results = get_sklearn_score(pred_labels, gold_labels, idx2label)
    coarse_results = get_sklearn_score(pred_coarse_labels, gold_coarse_labels, idx2coarse_label)

    print("result of get_ai_score")
    for k in coarse_results.keys():
        print("{} : {}\n".format(k, coarse_results[k]))
    for k in results.keys():
        print("{} : {}\n".format(k, results[k]))

    print("result of get_sklearn_score")
    sk_results = get_sklearn_score(pred_labels, gold_labels, idx2label)
    sk_coarse_results = get_sklearn_score(pred_coarse_labels, gold_coarse_labels, idx2coarse_label)
    for k in sk_coarse_results.keys():
        print("{} : {}\n".format(k, sk_coarse_results[k]))
    for k in sk_results.keys():
        print("{} : {}\n".format(k, sk_results[k]))


    # 검증 스크립트 기반 성능 저장
    output_dir = os.path.join(args.output_dir, 'test')

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type = 'w'

    ## 검증 스크립트 기반 성능 저장을 위한 파일 생성
    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else:
        eval_file_name = "init_weight"

    ## 대분류 세부분류
    print("===== 대분류 세부분류 =====")
    coarse_new_output_1 = torch.zeros([9,9])
    coarse_new_output_2 = torch.zeros([9,9])
    coarse_new_output_3 = torch.zeros([9,9])

    for i,(cg,cp, g,p) in enumerate(zip(pred_coarse_labels, gold_coarse_labels, gold_labels, pred_labels)):
       if cp == 0:
           coarse_new_output_1[p][g] += torch.tensor(1)
       elif cp == 1:
           coarse_new_output_2[p][g] += torch.tensor(1)
       elif cp == 2:
           coarse_new_output_3[p][g] += torch.tensor(1)


    print("============대분류 결과====================")
    for co in zip(coarse_new_output_1, coarse_new_output_2, coarse_new_output_3):
        print(co)


    ### incorrect data 저장
    out_incorrect = {"sentence": [], "correct": [], "predict": []}
    print('\n\n=====================outputs=====================')
    for i,(g,p) in enumerate(zip(gold_labels, pred_labels)):
        if g != p:
            out_incorrect["sentence"].append(examples[i].sentence)
            out_incorrect["correct"].append(idx2label[g])
            out_incorrect["predict"].append(idx2label[p])
    df_incorrect = pd.DataFrame(out_incorrect)
    df_incorrect.to_csv(os.path.join(output_dir, "test_result_{}_incorrect.csv".format(eval_file_name)), index=False)

    ### 전체 data 저장
    out = {"sentence":[], "correct":[], "predict":[]}
    for i,(g,p) in enumerate(zip(gold_labels, pred_labels)):
        for k,v in zip(out.keys(),[examples[i].sentence, idx2label[g], idx2label[p]]):
            out[k].append(v)
    for k, v in zip(out.keys(), [examples[i].sentence, idx2label[g], idx2label[p]]):
        out[k].append(v)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(output_dir, "test_result_{}.csv".format(eval_file_name)), index=False)

    return results
