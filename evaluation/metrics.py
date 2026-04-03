from precision import precision_at_k
from recall import recall_at_k
from ndcg import ndcg_at_k


def evaluate(recommended, ground_truth, k=10):

    precision = precision_at_k(recommended, ground_truth, k)
    recall = recall_at_k(recommended, ground_truth, k)
    ndcg = ndcg_at_k(recommended, ground_truth, k)

    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg
    }