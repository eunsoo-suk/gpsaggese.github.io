"""
clip_utils.py

Utility functions supporting `clip.API.ipynb` and `clip.example.ipynb`.

The notebooks call these functions instead of writing raw logic inline.

Import as:

import clip_utils as cliputil
"""

import logging
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)

# Sentiment classes in a fixed order, used as integer class ids.
LABELS = ["negative", "neutral", "positive"]


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Label processing
# #############################################################################


def _majority_vote(votes: List[str]) -> Optional[str]:
    """
    Return the label chosen by at least 2 of the 3 annotators.

    :param votes: labels from the 3 annotators
    :return: majority label, or `None` if all 3 annotators disagree
    """
    label, count = Counter(votes).most_common(1)[0]
    return label if count >= 2 else None


def _combine_text_image(
    text_label: Optional[str], image_label: Optional[str]
) -> Optional[str]:
    """
    Merge the text and image labels into one post label.

    Follow the rule of Xu & Mao (2017):
    - Same label: keep it.
    - One side neutral: take the non-neutral side.
    - Positive vs negative: drop the post.

    :return: post label, or `None` if the post is dropped
    """
    if text_label is None or image_label is None:
        return None
    if text_label == image_label:
        return text_label
    if text_label == "neutral":
        return image_label
    if image_label == "neutral":
        return text_label
    return None


def load_mvsa_labels(label_path: str) -> pd.DataFrame:
    """
    Parse `labelResultAll.txt` and aggregate the annotator votes.

    Each row of the file is `ID  t,i  t,i  t,i`, i.e., 3 annotators each
    giving a (text, image) sentiment pair.

    :param label_path: path to `labelResultAll.txt`
    :return: one row per post with columns `id`, `text_label`,
        `image_label`, `label` (`None` where the post is dropped)
    """
    rows = []
    with open(label_path, encoding="utf-8") as f:
        # Skip the header.
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            post_id = int(parts[0])
            votes = [p.split(",") for p in parts[1:4]]
            text_votes = [v[0] for v in votes]
            image_votes = [v[1] for v in votes]
            rows.append((post_id, text_votes, image_votes))
    df = pd.DataFrame(rows, columns=["id", "text_votes", "image_votes"])
    df["text_label"] = df["text_votes"].map(_majority_vote)
    df["image_label"] = df["image_votes"].map(_majority_vote)
    df["label"] = [
        _combine_text_image(t, i)
        for t, i in zip(df["text_label"], df["image_label"])
    ]
    df = df.drop(columns=["text_votes", "image_votes"])
    _LOG.info(
        "Loaded %d posts, %d kept after label aggregation",
        len(df),
        df["label"].notna().sum(),
    )
    return df


# #############################################################################
# Text and image files
# #############################################################################


def clean_tweet(text: str) -> str:
    """
    Normalize tweet text before feeding it to the CLIP text encoder.

    - Remove URLs and the `RT` prefix.
    - Replace user mentions with `@user`.
    - Keep hashtag words but drop the `#` symbol.
    """
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"^RT\s+", " ", text)
    text = re.sub(r"@\w+", "@user", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def _read_text(path: str) -> Optional[str]:
    """
    Read a tweet text file, returning `None` if it is missing or empty.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return text or None


def _is_valid_image(path: str) -> bool:
    """
    Check that an image file exists and can be decoded.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def attach_files(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Attach the tweet text and image path of each post, dropping broken posts.

    :param df: output of `load_mvsa_labels()` with the dropped posts removed
    :param data_dir: directory with `<id>.txt` and `<id>.jpg` files
    :return: `df` with `text_raw`, `text`, `image_path` columns
    """
    df = df.copy()
    texts, images, reasons = [], [], []
    for post_id in tqdm(df["id"], desc="Checking files"):
        text = _read_text(os.path.join(data_dir, f"{post_id}.txt"))
        image_path = os.path.join(data_dir, f"{post_id}.jpg")
        reason = None
        if text is None:
            reason = "missing_or_empty_text"
        elif not _is_valid_image(image_path):
            reason = "missing_or_broken_image"
        texts.append(text)
        images.append(image_path)
        reasons.append(reason)
    df["text_raw"] = texts
    df["image_path"] = images
    df["drop_reason"] = reasons
    n_bad = df["drop_reason"].notna().sum()
    if n_bad > 0:
        _LOG.warning(
            "Dropping %d posts:\n%s",
            n_bad,
            df["drop_reason"].value_counts().to_string(),
        )
    df = df[df["drop_reason"].isna()].drop(columns=["drop_reason"])
    df["text"] = df["text_raw"].map(clean_tweet)
    return df.reset_index(drop=True)


# #############################################################################
# Train / validation / test split
# #############################################################################


def split_dataset(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add a `split` column with a stratified train / val / test split.

    Stratify on `label` so that each split keeps the class imbalance.
    """
    df = df.copy()
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    val_size = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def build_mvsa_table(
    raw_dir: str,
    out_path: str,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full data preparation and save the result as a CSV file.

    :param raw_dir: directory containing `labelResultAll.txt` and `data/`
    :param out_path: output CSV path, e.g. `data/processed/labels.csv`
    :return: one row per usable post with columns `id`, `text`,
        `text_raw`, `image_path`, `text_label`, `image_label`, `label`,
        `label_id`, `split`
    """
    df = load_mvsa_labels(os.path.join(raw_dir, "labelResultAll.txt"))
    df = df[df["label"].notna()]
    df = attach_files(df, os.path.join(raw_dir, "data"))
    df["label_id"] = df["label"].map(LABELS.index)
    df = split_dataset(df, seed=seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    _LOG.info("Saved %d posts to %s", len(df), out_path)
    return df


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count posts per split and label, with the label share in each split.
    """
    counts = pd.crosstab(df["split"], df["label"])[LABELS]
    counts["total"] = counts.sum(axis=1)
    return counts.loc[["train", "val", "test"]]
