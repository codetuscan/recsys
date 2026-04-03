import pandas as pd

def train_test_split(ratings):

    ratings = ratings.sort_values("timestamp")

    train = ratings.groupby("userId").head(-1)
    test = ratings.groupby("userId").tail(1)

    return train, test