def recall_at_k(recommended, ground_truth, k):

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(ground_truth))

    return hits / len(ground_truth)