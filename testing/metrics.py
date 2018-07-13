import numpy as np
import csv
import operator


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items, tested_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / (len(relevant_items)*tested_items)

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluateSubmission(testSet, submitted):
    """:parameter test: submission file path
       :parameter submission: testset file path or dictionary"""


    if type(submitted) is dict:
        submission = submitted
    else:
        reader = csv.reader(open(submitted), delimiter=',')
        next(reader)  # hop header line

        submission = {}

        for line in reader:
            submission[line[0]] = [i for i in line[1].split('\t') if len(i) > 0]

    reader = csv.reader(open(testSet), delimiter=',')
    next(reader)  # hop header line

    test = {}

    for line in reader:
        test[line[0]] = [i for i in line[1].split('\t') if len(i) > 0]

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for playlist in test.keys():
        relevant_items = np.array(test[playlist])
        recommended_items = np.array(submission[playlist])

        num_eval += 1
        cumulative_precision += precision(recommended_items, relevant_items)
        cumulative_recall += recall(recommended_items, relevant_items, len(test))
        cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    return {'precision' : cumulative_precision, 'recall' : cumulative_recall, 'MAP' : cumulative_MAP}

