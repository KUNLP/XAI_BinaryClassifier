import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

def get_score(predicts, corrects, idx2label):
    predicts = [idx2label[predict] for predict in predicts]
    corrects = [idx2label[correct] for correct in corrects]
    result = {}

    def get_score_one_class(predicts, corrects, value):
        TP, FP, FN, TN = 0, 0, 0, 0
        for correct, predict in zip(corrects, predicts):
            if(correct == value and predict == value):
                TP += 1
            elif(correct != value and predict == value):
                FP += 1
            elif(correct == value and predict != value):
                FN += 1
            elif(correct != value and predict != value):
                TN += 1

        if(TP == 0):
            precision, recall, f1_score, accuracy = 0, 0, 0, 0
        else:
            precision = float(TP)/(TP+FP)
            recall = float(TP)/(TP+FN)
            f1_score = (2*precision*recall)/(precision+recall)
            accuracy = float(TP+TN)/(TP+FN+FP+TN)

        return precision, recall, f1_score, accuracy, TP, FP, FN, TN

    values = list(idx2label.values())
    for value in values:
        precision, recall, f1_score, accuracy, TP, FP, FN, TN = get_score_one_class(predicts, corrects, value)
        result[value] = {"precision":precision, "recall":recall, "f1_score":f1_score, "accuracy":accuracy,
                         "TP":TP, "FP":FP, "FN":FN, "TN":TN}

    macro_precision = np.sum([result[value]["precision"] for value in values]) / len(values)
    macro_recall = np.sum([result[value]["recall"] for value in values]) / len(values)
    macro_f1_score = np.sum([result[value]["f1_score"] for value in values]) / len(values)
    total_accuracy = np.sum([result[value]["accuracy"] for value in values]) / len(values)

    total_TP = np.sum([result[value]["TP"] for value in values])
    total_FP = np.sum([result[value]["FP"] for value in values])
    total_FN = np.sum([result[value]["FN"] for value in values])
    total_TN = np.sum([result[value]["TN"] for value in values])

    if (total_TP == 0):
        micro_precision, micro_recall, micro_f1_score, accuracy = 0, 0, 0, 0
    else:
        micro_precision = float(total_TP) / (total_TP + total_FP)
        micro_recall = float(total_TP) / (total_TP + total_FN)
        micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    for value in values:
        precision, recall, f1_score = result[value]["precision"], result[value]["recall"], result[value]["f1_score"]
        TP, FP, FN, TN = result[value]["TP"], result[value]["FP"], result[value]["FN"], result[value]["TN"]

        print("Precision from {} : ".format(value) + str(round(precision, 4)))
        print("Recall from {} : ".format(value) + str(round(recall, 4)))
        print("F1_score from {} : ".format(value) + str(round(f1_score, 4)))

        print("True Positive from {} : ".format(value) + str(TP))
        print("False Positive from {} : ".format(value) + str(FP))
        print("False Negative from {} : ".format(value) + str(FN))
        print("True Negative from {} : ".format(value) + str(TN) + "\n")


    print("macro_precision: "+ str(round(macro_precision, 4)))
    print("macro_recall: "+ str(round(macro_recall, 4)))
    print("macro_f1_score: "+ str(round(macro_f1_score, 4)))
    print("macro_Accuracy: " + str(round(total_accuracy, 4)))

    return {"macro_precision":round(macro_precision, 4), "macro_recall":round(macro_recall, 4), "macro_f1_score":round(macro_f1_score, 4), \
            "macro_accuracy":round(total_accuracy, 4), \
           "micro_precision":round(micro_precision, 4), "micro_recall":round(micro_recall, 4), "micro_f1":round(micro_f1_score, 4)}


def get_sklearn_score(predicts, corrects, idx2label):
    predicts = [idx2label[predict] for predict in predicts]
    corrects = [idx2label[correct] for correct in corrects]
    result = {"accuracy": accuracy_score(corrects, predicts),
              "macro_precision": precision_score(corrects, predicts, average="macro"),
              "micro_precision": precision_score(corrects, predicts, average="micro"),
              "macro_f1": f1_score(corrects, predicts, average="macro"),
              "micro_f1": f1_score(corrects, predicts, average="micro"),
              "macro_recall": recall_score(corrects, predicts, average="macro"),
              "micro_recall": recall_score(corrects, predicts, average="micro"),
              }

    for k, v in result.items():
        result[k] = round(v, 3)
        print(k + ": " + str(v))
    return result



def get_ai_score(predicts, corrects, idx2label):
    predicts = [idx2label[predict] for predict in predicts]
    corrects = [idx2label[correct] for correct in corrects]
    result = {}

    def get_score_one_class(predicts, corrects, value):
        TP, FP, FN, TN = 0, 0, 0, 0
        for correct, predict in zip(corrects, predicts):
            if(correct == value and predict == value):
                TP += 1
            elif(correct != value and predict == value):
                FP += 1
            elif(correct == value and predict != value):
                FN += 1
            elif(correct != value and predict != value):
                TN += 1

        if(TP == 0):
            precision, recall, f1_score, accuracy = 0, 0, 0, 0
        else:
            precision = float(TP)/(TP+FP)
            recall = float(TP)/(TP+FN)
            f1_score = (2*precision*recall)/(precision+recall)
            accuracy = float(TP+TN)/(TP+FN+FP+TN)

        return precision, recall, f1_score, accuracy, TP, FP, FN, TN

    values = list(idx2label.values())
    for value in values:
        precision, recall, f1_score, accuracy, TP, FP, FN, TN = get_score_one_class(predicts, corrects, value)
        result[value] = {"precision":precision, "recall":recall, "f1_score":f1_score, "accuracy":accuracy,
                         "TP":TP, "FP":FP, "FN":FN, "TN":TN}

    macro_precision = np.sum([result[value]["precision"] for value in values]) / len(values)
    macro_recall = np.sum([result[value]["recall"] for value in values]) / len(values)
    macro_f1_score = np.sum([result[value]["f1_score"] for value in values]) / len(values)
    total_accuracy = np.sum([result[value]["accuracy"] for value in values]) / len(values)

    total_TP = np.sum([result[value]["TP"] for value in values])
    total_FP = np.sum([result[value]["FP"] for value in values])
    total_FN = np.sum([result[value]["FN"] for value in values])
    total_TN = np.sum([result[value]["TN"] for value in values])

    if (total_TP == 0):
        micro_precision, micro_recall, micro_f1_score, accuracy = 0, 0, 0, 0
    else:
        micro_precision = float(total_TP) / (total_TP + total_FP)
        micro_recall = float(total_TP) / (total_TP + total_FN)
        micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        micro_accuracy = float(total_TP + total_TN) / (total_TP + total_TN + total_FP+ total_FN)

    for value in values:
        precision, recall, f1_score = result[value]["precision"], result[value]["recall"], result[value]["f1_score"]
        TP, FP, FN, TN = result[value]["TP"], result[value]["FP"], result[value]["FN"], result[value]["TN"]

        print("Precision from {} : ".format(value) + str(round(precision, 4)))
        print("Recall from {} : ".format(value) + str(round(recall, 4)))
        print("F1_score from {} : ".format(value) + str(round(f1_score, 4)))

        print("True Positive from {} : ".format(value) + str(TP))
        print("False Positive from {} : ".format(value) + str(FP))
        print("False Negative from {} : ".format(value) + str(FN))
        print("True Negative from {} : ".format(value) + str(TN) + "\n")


    print("macro_precision: "+ str(round(macro_precision, 4)))
    print("macro_recall: "+ str(round(macro_recall, 4)))
    print("macro_f1_score: "+ str(round(macro_f1_score, 4)))
    print("accuracy: " + str(round(total_accuracy, 4)))
    print("micro_Accuracy: " + str(round(micro_accuracy, 4)))

    return {"macro_precision":round(macro_precision, 4), "macro_recall":round(macro_recall, 4), "macro_f1_score":round(macro_f1_score, 4), \
            "macro_accuracy":round(total_accuracy, 4), "micro_accuracy":round(micro_accuracy, 4), \
           "micro_precision":round(micro_precision, 4), "micro_recall":round(micro_recall, 4), "micro_f1":round(micro_f1_score, 4)}