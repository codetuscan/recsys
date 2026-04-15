"""Run public-style PURS training with Kaggle-compatible TensorFlow v1 APIs.

This script preserves the public training/evaluation protocol and reads run
parameters from the existing YAML config (epochs, batch_size, learning_rate,
history_length, seed, threshold).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _import_tensorflow_with_jax_fallback():
    """Import TensorFlow and auto-recover from JAX/ml_dtypes conflicts on Kaggle."""
    try:
        import tensorflow as tf_mod

        return tf_mod
    except Exception as exc:
        message = str(exc)
        if "JAX requires ml_dtypes version" not in message:
            raise

        print(
            "Detected JAX/ml_dtypes conflict while importing TensorFlow. "
            "Removing optional JAX packages and retrying..."
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "jax", "jaxlib"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for module_name in list(sys.modules):
            if module_name.startswith("jax") or module_name.startswith("tensorflow"):
                sys.modules.pop(module_name, None)

        import tensorflow as tf_mod

        return tf_mod


tf = _import_tensorflow_with_jax_fallback()
tf1 = tf.compat.v1

from model_public_tf_compat import Model


class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[
            self.i * self.batch_size : min((self.i + 1) * self.batch_size, len(self.data))
        ]
        self.i += 1
        u, hist, it, y = [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            it.append(t[2])
            y.append(t[3])
        return self.i, (u, hist, it, y)


# Public metric functions (kept intentionally aligned).
def test_auc(sess, model, test_set, batch_size):
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    arr = []

    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        _ = user
        for index in range(len(score)):
            if label[index] > 0:
                arr.append([0, 1, score[index]])
            elif label[index] == 0:
                arr.append([1, 0, score[index]])

    arr = sorted(arr, key=lambda d: d[2])
    for record in arr:
        fp2 += record[0]
        tp2 += record[1]
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:
        return 1.0 - auc / (2.0 * tp2 * fp2)
    return None


def hit_rate(sess, model, test_set, batch_size):
    hit, arr = [], []
    userid = list(set([x[0] for x in test_set]))

    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                arr.append([label[index], 1, user[index]])
            else:
                arr.append([label[index], 0, user[index]])

    for user in userid:
        arr_user = [x for x in arr if x[2] == user and x[1] == 1]
        hit.append(sum([x[0] for x in arr_user]) / len(arr_user))

    return np.mean(hit)


def coverage(sess, model, test_set, batch_size, itemid):
    rec_item = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, _ = model.test(sess, uij)
        _ = (label, user)
        for index in range(len(score)):
            if score[index] > 0.5:
                rec_item.append(item[index])
    return len(set(rec_item)) / len(itemid)


def unexpectedness(sess, model, test_set, batch_size):
    unexp_list = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, unexp = model.test(sess, uij)
        _ = (score, label, user, item)
        for index in range(len(unexp)):
            unexp_list.append(unexp[index])
    return np.mean(unexp_list)


def _normalize_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if key in {"movielens-1m", "ml-1m", "1m", "movielens1m"}:
        return "movielens-1m"
    if key in {"movielens-32m", "ml-32m", "32m", "movielens32m"}:
        return "movielens-32m"
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _resolve_ratings_file(dataset_name: str, raw_dir: Path | None) -> Path:
    dataset_name = _normalize_dataset_name(dataset_name)

    if raw_dir is None:
        raw_dir = Path("/kaggle/input")

    raw_dir = Path(raw_dir)

    if dataset_name == "movielens-1m":
        candidates = [
            raw_dir / "ratings.dat",
            raw_dir / "ml-1m" / "ratings.dat",
            raw_dir / "movielens-1m" / "ratings.dat",
            raw_dir / "movielens-1m" / "ml-1m" / "ratings.dat",
            Path("/kaggle/input/movielens-1m/ml-1m/ratings.dat"),
            Path("/kaggle/input/movielens-1m/ratings.dat"),
            Path("/kaggle/input/ml-1m/ml-1m/ratings.dat"),
            Path("/kaggle/input/ml-1m/ratings.dat"),
        ]
    else:
        candidates = [
            raw_dir / "ratings.csv",
            raw_dir / "ml-32m" / "ratings.csv",
            raw_dir / "movielens-32m" / "ratings.csv",
            raw_dir / "movielens-32m" / "ml-32m" / "ratings.csv",
            Path("/kaggle/input/movielens-32m/ml-32m/ratings.csv"),
            Path("/kaggle/input/movielens-32m/ratings.csv"),
            Path("/kaggle/input/ml-32m/ml-32m/ratings.csv"),
            Path("/kaggle/input/ml-32m/ratings.csv"),
        ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find ratings file for {dataset_name}. Checked: {candidates}"
    )


def build_public_input_file(
    dataset_name: str,
    output_file: Path,
    threshold: float,
    seed: int,
    raw_dir: Path | None = None,
    data_subset: float | None = None,
) -> Path:
    ratings_file = _resolve_ratings_file(dataset_name=dataset_name, raw_dir=raw_dir)

    if _normalize_dataset_name(dataset_name) == "movielens-1m":
        df = pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["utdid", "vdo_id", "rating", "hour"],
        )
    else:
        raw = pd.read_csv(ratings_file)
        df = raw.rename(
            columns={
                "userId": "utdid",
                "movieId": "vdo_id",
                "rating": "rating",
                "timestamp": "hour",
            }
        )[["utdid", "vdo_id", "rating", "hour"]]

    df["click"] = (df["rating"] >= float(threshold)).astype(float)
    df = df[["utdid", "vdo_id", "click", "hour"]]

    if data_subset is not None:
        frac = float(data_subset)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"data_subset must be in (0, 1], got {data_subset}")
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, header=False)
    return output_file


def build_public_train_test(data_path: Path, batch_size: int, history_length: int):
    data = pd.read_csv(data_path, names=["utdid", "vdo_id", "click", "hour"])

    user_id = data[["utdid"]].drop_duplicates().reindex()
    user_id["user_id"] = np.arange(len(user_id))
    data = pd.merge(data, user_id, on=["utdid"], how="left")

    item_id = data[["vdo_id"]].drop_duplicates().reindex()
    item_id["video_id"] = np.arange(len(item_id))
    data = pd.merge(data, item_id, on=["vdo_id"], how="left")

    data = data[["user_id", "video_id", "click", "hour"]]
    userid = list(set(data["user_id"]))
    itemid = list(set(data["video_id"]))
    user_count = len(userid)
    item_count = len(itemid)

    validate = 4 * len(data) // 5
    train_data = data.loc[:validate, :]
    test_data = data.loc[validate:, :]
    train_set, test_set = [], []

    for user in userid:
        train_user = train_data.loc[train_data["user_id"] == user]
        train_user = train_user.sort_values(["hour"])
        length = len(train_user)
        train_user.index = range(length)
        if length > history_length:
            for i in range(length - history_length):
                target_idx = i + history_length - 1
                train_set.append(
                    (
                        train_user.loc[target_idx, "user_id"],
                        list(train_user.loc[i:target_idx, "video_id"]),
                        train_user.loc[target_idx, "video_id"],
                        float(train_user.loc[target_idx, "click"]),
                    )
                )

        test_user = test_data.loc[test_data["user_id"] == user]
        test_user = test_user.sort_values(["hour"])
        length = len(test_user)
        test_user.index = range(length)
        if length > history_length:
            for i in range(length - history_length):
                target_idx = i + history_length - 1
                test_set.append(
                    (
                        test_user.loc[target_idx, "user_id"],
                        list(test_user.loc[i:target_idx, "video_id"]),
                        test_user.loc[target_idx, "video_id"],
                        float(test_user.loc[target_idx, "click"]),
                    )
                )

    random.shuffle(train_set)
    random.shuffle(test_set)
    train_set = train_set[: len(train_set) // batch_size * batch_size]
    test_set = test_set[: len(test_set) // batch_size * batch_size]

    return train_set, test_set, user_count, item_count, itemid


def parse_args():
    parser = argparse.ArgumentParser(description="Public-style PURS training for Kaggle")
    parser.add_argument(
        "--config",
        type=str,
        default="config/kaggle_config.yaml",
        help="YAML config path (uses model/experiment/data fields)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Existing public-format data file (utdid,vdo_id,click,hour). If empty, auto-build.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="",
        help="Optional root directory to search for MovieLens ratings file",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Override dataset name (movielens-1m or movielens-32m)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/legacy_public_purs",
        help="Directory for generated input + metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    exp_cfg = config.get("experiment", {})

    seed = int(exp_cfg.get("seed", 625))
    batch_size = int(model_cfg.get("batch_size", 32))
    epochs = int(model_cfg.get("epochs", 1000))
    learning_rate = float(model_cfg.get("learning_rate", 1.0))
    history_length = int(model_cfg.get("history_length", 10))
    hidden_size = int(model_cfg.get("gru_hidden_dim", 128))
    positive_threshold = float(data_cfg.get("positive_rating_threshold", 3.5))
    data_subset = data_cfg.get("data_subset", None)

    dataset_name = args.dataset_name or data_cfg.get("dataset_name", "movielens-1m")
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    random.seed(seed)
    np.random.seed(seed)
    tf1.set_random_seed(seed)
    tf1.disable_eager_execution()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        public_input_path = Path(args.data_path)
    else:
        public_input_path = output_dir / "test.txt"
        public_input_path = build_public_input_file(
            dataset_name=dataset_name,
            output_file=public_input_path,
            threshold=positive_threshold,
            seed=seed,
            raw_dir=raw_dir,
            data_subset=data_subset,
        )

    print(f"Public input file: {public_input_path}")
    print(
        f"Config -> dataset={dataset_name}, batch_size={batch_size}, epochs={epochs}, "
        f"learning_rate={learning_rate}, history_length={history_length}, seed={seed}"
    )

    train_set, test_set, user_count, item_count, itemid = build_public_train_test(
        data_path=public_input_path,
        batch_size=batch_size,
        history_length=history_length,
    )

    if len(train_set) == 0 or len(test_set) == 0:
        raise ValueError(
            "Empty train/test set after public preprocessing. "
            "Try larger data_subset or lower batch_size/history_length."
        )

    gpu_options = tf1.GPUOptions(allow_growth=True)

    with tf1.Session(config=tf1.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Model(
            user_count=user_count,
            item_count=item_count,
            batch_size=batch_size,
            hidden_size=hidden_size,
            long_memory_window=history_length,
            short_memory_window=min(3, history_length),
        )

        sess.run(tf1.global_variables_initializer())
        sess.run(tf1.local_variables_initializer())

        initial_test_auc = test_auc(sess, model, test_set, batch_size)
        print("test_auc: %.4f" % initial_test_auc)
        sys.stdout.flush()

        lr = learning_rate
        start_time = time.time()
        last_auc = 0.0

        epoch_metrics = []

        for _ in range(epochs):
            random.shuffle(train_set)
            loss_sum = 0.0
            train_auc = None
            eval_auc = None

            for _, uij in DataInput(train_set, batch_size):
                loss = model.train(sess, uij, lr)
                loss_sum += loss

                if model.global_step.eval() % 100 == 0:
                    eval_auc = test_auc(sess, model, test_set, batch_size)
                    train_auc = test_auc(sess, model, train_set, batch_size)
                    print(
                        "Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f\tTrain_AUC: %.4F"
                        % (
                            model.global_epoch_step.eval(),
                            model.global_step.eval(),
                            loss_sum / 1000,
                            eval_auc,
                            train_auc,
                        )
                    )
                    sys.stdout.flush()
                    loss_sum = 0.0

            print(
                "Epoch %d DONE\tCost time: %.2f"
                % (model.global_epoch_step.eval(), time.time() - start_time)
            )

            if train_auc is None:
                train_auc = test_auc(sess, model, train_set, batch_size)
            if eval_auc is None:
                eval_auc = test_auc(sess, model, test_set, batch_size)

            if abs(train_auc - last_auc) < 0.001:
                lr = lr / 2
            last_auc = train_auc
            sys.stdout.flush()

            model.global_epoch_step_op.eval()
            hit = hit_rate(sess, model, test_set, batch_size)
            cov = coverage(sess, model, test_set, batch_size, itemid)
            unexp = unexpectedness(sess, model, test_set, batch_size)

            current_epoch = int(model.global_epoch_step.eval())
            print("Epoch %d Eval_Hit_Rate: %.4f" % (current_epoch, hit))
            print("Epoch %d Eval_Coverage: %.4f" % (current_epoch, cov))
            print("Epoch %d Eval_Unexpectedness: %.4f" % (current_epoch, unexp))

            epoch_metrics.append(
                {
                    "epoch": current_epoch,
                    "train_auc": None if train_auc is None else float(train_auc),
                    "eval_auc": None if eval_auc is None else float(eval_auc),
                    "hit_rate": float(hit),
                    "coverage": float(cov),
                    "unexpectedness": float(unexp),
                    "lr": float(lr),
                }
            )

    summary = {
        "config_path": str(config_path),
        "public_input_path": str(public_input_path),
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate_initial": learning_rate,
        "history_length": history_length,
        "seed": seed,
        "user_count": user_count,
        "item_count": item_count,
        "epoch_metrics": epoch_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
