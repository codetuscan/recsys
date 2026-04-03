def precision_at_k(recommended, ground_truth, k):
    """
    recommended : list of recommended items
    ground_truth : items actually interacted by user
    k : top k
    """

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(ground_truth))

    return hits / k