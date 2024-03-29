from cgi import test
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np

seed = 42

np.random.seed(seed)

dataset_name = "poli"
train_df = pd.read_csv("./data/" + dataset_name + "_train.csv")
validate_df = pd.read_csv("./data/" + dataset_name + "_validation.csv")
test_df = pd.read_csv("./data/" + dataset_name + "_test.csv")
train_df = pd.concat([train_df, validate_df], ignore_index=True)
# augmentation_df=pd.read_csv('./data/'+dataset_name+'_augmentation.csv')


X_train, X_test, y_train, y_test = (
    train_df["text"],
    test_df["text"],
    train_df["label"],
    test_df["label"],
)
print(len(y_train), len(y_test))


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# model = LinearSVC()
model = LogisticRegression()
# model=RandomForestClassifier(n_estimators=5)
model.fit(X_train_vectorized, y_train)


y_pred = model.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, y_pred)
print(dataset_name)

print(f"accuracy: {accuracy:.4f}")


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# y_score = model.decision_function(X_test_vectorized)
y_score = model.predict_proba(X_test_vectorized)[:, 1]
roc_auc = roc_auc_score(y_test, y_score)

print(f"AUC-ROC: {roc_auc:.4f}")
