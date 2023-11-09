import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["generate_tf_embeddings"]


def get_tf_embeddings_BOW(
    df: pd.DataFrame,
    feature: str,
    max_features: int,
    max_len: int,
    col_name: str,
):
    """
    Generates embeddings based on Bag-of-Words approach.

    feature: List of strings with fixed sequence of items e.g. ['123 24 15 0 0 0', '54 35 0 0 0 0']
    max_features: Maximum vocab size.
    max_len: Sequence length to pad the outputs to.
    col_name: Name of columns in output DataFrame with embeddings/
    """
    text_dataset = tf.data.Dataset.from_tensor_slices(df[feature].tolist())
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorize_layer.adapt(text_dataset.batch(64))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)

    emb = model.predict(df[feature])
    emb = pd.DataFrame(
        emb, index=df.index, columns=[f"{col_name}_{i}" for i in range(max_len)]
    )

    return emb, model


def fit_transform_embeddings(
    df: pd.DataFrame,
    feature: str,
    output_dim: int,
):
    vocab = df[feature].astype(str).unique().tolist()
    str_lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    lookup_and_embed = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),
            str_lookup_layer,
            tf.keras.layers.Embedding(
                input_dim=str_lookup_layer.vocabulary_size(),
                output_dim=output_dim,
            ),
        ]
    )
    emb = lookup_and_embed(tf.constant(df[feature].astype(str))).numpy()
    cols = [f"{feature}_{i}" for i in range(output_dim)]

    return pd.DataFrame(emb, index=df.index, columns=cols), lookup_and_embed


def transform_embeddings(
    df: pd.DataFrame, feature: str, lookup_and_embed
) -> pd.DataFrame:

    emb = lookup_and_embed(tf.constant(df[feature].astype(str))).numpy()
    cols = [f"{feature}_{i}" for i in range(lookup_and_embed.output_shape[1])]

    return pd.DataFrame(emb, index=df.index, columns=cols)
