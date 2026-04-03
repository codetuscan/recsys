from evaluation.precision import precision_at_k
from evaluation.recall import recall_at_k
from evaluation.ndcg import ndcg_at_k

recommended = [10,20,30,40,50]
ground_truth = [30]

print("Precision:", precision_at_k(recommended, ground_truth, 5))
print("Recall:", recall_at_k(recommended, ground_truth, 5))
print("NDCG:", ndcg_at_k(recommended, ground_truth, 5))