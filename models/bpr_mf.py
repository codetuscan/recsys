import numpy as np
import random


class BPRMF:

    def __init__(self, num_users, num_items, factors=50, lr=0.01, reg=0.01):

        self.num_users = num_users
        self.num_items = num_items
        self.factors = factors
        self.lr = lr
        self.reg = reg

        self.user_factors = np.random.normal(0, 0.1, (num_users, factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, factors))

    def predict(self, u, i):
        return np.dot(self.user_factors[u], self.item_factors[i])

    def sample_negative(self, user, user_items):

        while True:
            j = np.random.randint(self.num_items)
            if j not in user_items[user]:
                return j

    def train(self, user_items, epochs=10):

        users = list(user_items.keys())

        for epoch in range(epochs):

            random.shuffle(users)

            for u in users:

                positive_items = list(user_items[u])

                for i in positive_items:

                    j = self.sample_negative(u, user_items)

                    x_ui = self.predict(u, i)
                    x_uj = self.predict(u, j)

                    x_uij = x_ui - x_uj

                    sigmoid = 1 / (1 + np.exp(-x_uij))

                    grad = 1 - sigmoid

                    self.user_factors[u] += self.lr * (
                        grad * (self.item_factors[i] - self.item_factors[j])
                        - self.reg * self.user_factors[u]
                    )

                    self.item_factors[i] += self.lr * (
                        grad * self.user_factors[u]
                        - self.reg * self.item_factors[i]
                    )

                    self.item_factors[j] += self.lr * (
                        -grad * self.user_factors[u]
                        - self.reg * self.item_factors[j]
                    )

            print(f"epoch {epoch} completed")

    def recommend(self, user_id, k=10):

        scores = np.dot(self.item_factors, self.user_factors[user_id])

        return np.argsort(scores)[::-1][:k]