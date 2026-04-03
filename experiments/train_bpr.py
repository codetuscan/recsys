import pandas as pd
from collections import defaultdict
from models.bpr_mf import BPRMF


ratings = pd.read_csv("data/raw/ml-32m/ratings.csv")

# encode ids
ratings["user_idx"] = ratings["userId"].astype("category").cat.codes
ratings["item_idx"] = ratings["movieId"].astype("category").cat.codes

num_users = ratings["user_idx"].nunique()
num_items = ratings["item_idx"].nunique()

# build user → items dictionary
user_items = defaultdict(set)

for _, row in ratings.iterrows():
    user_items[row["user_idx"]].add(row["item_idx"])

model = BPRMF(num_users, num_items)

model.train(user_items, epochs=5)

print(model.recommend(0))