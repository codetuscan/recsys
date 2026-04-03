import pandas as pd
from collections import defaultdict
from models.bpr_mf import BPRMF
from data.train_test_split import train_test_split
from evaluation.precision import precision_at_k
from evaluation.recall import recall_at_k
from evaluation.ndcg import ndcg_at_k


ratings = pd.read_csv("data/raw/ml-32m/ratings.csv")

train, test = train_test_split(ratings)

# encode ids
train["user_idx"] = train["userId"].astype("category").cat.codes
train["item_idx"] = train["movieId"].astype("category").cat.codes

test["user_idx"] = test["userId"].astype("category").cat.codes
test["item_idx"] = test["movieId"].astype("category").cat.codes

num_users = train["user_idx"].nunique()
num_items = train["item_idx"].nunique()

# build user → items dictionary
user_items = defaultdict(set)

for _, row in train.iterrows():
    user_items[row["user_idx"]].add(row["item_idx"])

model = BPRMF(num_users, num_items)

model.train(user_items, epochs=5)

precision_scores = []
recall_scores = []
ndcg_scores = []

for _, row in test.iterrows():

    user = row["user_idx"]
    ground_truth = [row["item_idx"]]

    recs = model.recommend(user, k=10)

    precision_scores.append(
        precision_at_k(recs, ground_truth, 10)
    )

    recall_scores.append(
        recall_at_k(recs, ground_truth, 10)
    )

    ndcg_scores.append(
        ndcg_at_k(recs, ground_truth, 10)
    )


print("Precision@10:", sum(precision_scores)/len(precision_scores))
print("Recall@10:", sum(recall_scores)/len(recall_scores))
print("NDCG@10:", sum(ndcg_scores)/len(ndcg_scores))