import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


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


