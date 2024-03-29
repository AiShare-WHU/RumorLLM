import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import numpy as np
import tensorflow as tf
import random


def setup_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


dataset_name = "poli"

seed = 42

setup_seed(seed)

dataset_name = "poli"
train_df = pd.read_csv("./data/" + dataset_name + "_train.csv")
validate_df = pd.read_csv("./data/" + dataset_name + "_validation.csv")
test_df = pd.read_csv("./data/" + dataset_name + "_test.csv")


X_train, X_test, y_train, y_test = (
    train_df["text"],
    test_df["text"],
    train_df["label"],
    test_df["label"],
)
X_valid, y_valid = validate_df["text"], validate_df["label"]
print(len(y_train), len(y_test))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)

max_len = max(max(len(seq) for seq in X_train_seq), max(len(seq) for seq in X_test_seq))
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding="post")
X_valid_padded = pad_sequences(X_valid_seq, maxlen=max_len, padding="post")


embedding_dim = 32
filter_sizes = [3, 4, 5]
num_filters = 128

model = Sequential()
model.add(
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=embedding_dim,
        input_length=max_len,
    )
)
for filter_size in filter_sizes:
    model.add(Conv1D(num_filters, filter_size, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(
    X_train_padded,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_valid_padded, y_valid),
)


y_pred_proba = model.predict(X_test_padded)
y_pred = [1 if proba >= 0.5 else 0 for proba in y_pred_proba]
print(dataset_name)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC: {roc_auc:.4f}")
