import math

def dcg_at_k(recommended, ground_truth, k):

    dcg = 0

    for i, item in enumerate(recommended[:k]):

        if item in ground_truth:
            dcg += 1 / math.log2(i + 2)

    return dcg


def ndcg_at_k(recommended, ground_truth, k):

    dcg = dcg_at_k(recommended, ground_truth, k)

    ideal = dcg_at_k(ground_truth, ground_truth, k)

    if ideal == 0:
        return 0

    return dcg / ideal