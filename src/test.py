import pandas as pd
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import time
from torch.autograd import Variable
import numpy as np
import os
from sklearn import metrics
import random
from utils import preprocess_text

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def count_0_1(dataset):
    label_counts = {0: 0, 1: 0}
    for train_input_ids, train_attention_mask, train_type_ids, label in dataset:
        label_counts[label.item()] += 1
    print("Label 0 count:", label_counts[0])
    print("Label 1 count:", label_counts[1])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

dataset_name = "poli"
output_file = "./trained_models/"
epochs = 50
best_validate_acc = 0.000
best_test_acc = 0.000
best_loss = 100
batch_size = 8
best_validate_dir = ""
bert_model = "./deberta_model"
# bert_model='bert-base-cased'
# bert_model='./bert_model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 180
is_augmentation = False


class FakeNewsDataset(Dataset):
    def __init__(self, data):
        self.text = data["text"]  # tokenizer
        self.label = data["label"]  # 1 fake, 0 real
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        # self.tokenizer.save_pretrained('./bert_model')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer.encode_plus(
            preprocess_text(self.text[idx]),
            padding="max_length",  # Pad to max_length
            max_length=max_length,  # 156,180
            truncation=True,  # Truncate to max_length
            return_tensors="pt",
        )  # Return torch.Tensor objects

        token_ids = encoded_pair["input_ids"].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair["attention_mask"].squeeze(
            0
        )  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair["token_type_ids"].squeeze(
            0
        )  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        return token_ids, attn_masks, token_type_ids, self.label[idx]

    def count_labels(self):
        label_counts = {}
        for label in self.label:
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1
        return label_counts


all_df = pd.read_excel("./data/" + dataset_name + "_data.xlsx")
train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)
train_df, validate_df = train_test_split(train_df, test_size=0.125, random_state=42)
train_df.to_csv("./data/" + dataset_name + "_train.csv", index=False)
validate_df.to_csv("./data/" + dataset_name + "_validation.csv", index=False)
test_df.to_csv("./data/" + dataset_name + "_test.csv", index=False)

train_df = pd.read_csv("./data/" + dataset_name + "_train.csv")
val_df = pd.read_csv("./data/" + dataset_name + "_validation.csv")
test_df = pd.read_csv("./data/" + dataset_name + "_test.csv")

train_set = FakeNewsDataset(train_df)
valid_set = FakeNewsDataset(val_df)
test_set = FakeNewsDataset(test_df)
print(len(all_df), len(train_set), len(valid_set), len(test_set))
# all_df.to_csv('Buzz_data.csv')
augmentation_df = pd.read_csv("./data/" + dataset_name + "_augmentation.csv")
augmentation_dataset = FakeNewsDataset(augmentation_df)
print(len(augmentation_dataset))
print("max_length:" + str(max_length))
# train_ratio = 0.7
# val_ratio = 0.1
# test_ratio = 1-train_ratio-val_ratio
# dataset_percent=1


# num_samples = len(all_dataset)
# train_size = int(train_ratio * num_samples)
# val_size = int(val_ratio * num_samples)
# test_size = num_samples - train_size - val_size


# train_set, valid_set, test_set = random_split(all_dataset, [train_size, val_size, test_size])

# train_set=train_set+augmentation_dataset


# train_valid_set, test_set = random_split(all_dataset, [num_samples-test_size, test_size])
# train_valid_set=train_valid_set+augmentation_dataset
# t_v_len=len(train_valid_set)
# # count_0_1(train_valid_set)
# valid_len=int(t_v_len*val_ratio/(train_ratio+val_ratio))
# train_set, valid_set = random_split(train_valid_set, [t_v_len-valid_len, valid_len])

# count_0_1(train_set)
# count_0_1(valid_set)
print(test_set.count_labels())


if is_augmentation:
    train_loader = DataLoader(
        train_set + augmentation_dataset, batch_size=batch_size, shuffle=True
    )
else:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model, output_hidden_states=True, return_dict=True
        )
        self.bert_dim = 768
        self.hidden_dim = 64
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.linear1 = nn.Linear(self.bert_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 2)
        self.dropout = nn.Dropout(0.5)
        self.att_layer = nn.MultiheadAttention(embed_dim=self.bert_dim, num_heads=4)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # outputs.pooler_output: [bs, hidden_size]
        # sentence_embedding,_=self.att_layer(outputs.last_hidden_state,outputs.last_hidden_state,outputs.last_hidden_state)
        sentence_embedding = outputs.last_hidden_state

        sentence_embedding = torch.mean(sentence_embedding, dim=1).squeeze()
        # sentence_embedding,_=self.att_layer(sentence_embedding,sentence_embedding,sentence_embedding)
        logits = F.leaky_relu(self.linear1(self.dropout(sentence_embedding)))

        # logits = F.leaky_relu(self.linear1(self.dropout(outputs.pooler_output)))
        logits = self.linear2(self.dropout(logits))
        return logits


model = BertClassify().to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, list(model.parameters())),
    lr=1e-5,
    weight_decay=1e-4,
)

criterion = nn.CrossEntropyLoss()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


for epoch in range(epochs):
    start_time = time.time()
    cost_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    acc_vector = []
    valid_acc_vector = []
    test_acc_vector = []
    vali_cost_vector = []
    test_cost_vector = []

    for i, (
        train_input_ids,
        train_attention_mask,
        train_type_ids,
        train_labels,
    ) in enumerate(train_loader):
        train_input_ids, train_attention_mask, train_type_ids, train_labels = (
            to_var(train_input_ids),
            to_var(train_attention_mask),
            to_var(train_type_ids),
            to_var(train_labels),
        )

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        # train_image=image_embed(train_image)

        class_outputs = model(train_input_ids, train_attention_mask, train_type_ids)

        ## Fake or Real loss
        class_loss = criterion(class_outputs, train_labels)
        loss = class_loss
        loss.backward()
        optimizer.step()
        _, argmax = torch.max(class_outputs, 1)

        cross_entropy = True

        if True:
            accuracy = (train_labels == argmax.squeeze()).float().mean()
        else:
            _, labels = torch.max(train_labels, 1)
            accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

        class_cost_vector.append(class_loss.item())
        cost_vector.append(loss.item())
        acc_vector.append(accuracy.item())

    model.eval()
    validate_acc_vector_temp = []
    for i, (
        validate_input_ids,
        validate_attention_mask,
        validate_type_ids,
        validate_labels,
    ) in enumerate(validate_loader):
        (
            validate_input_ids,
            validate_attention_mask,
            validate_type_ids,
            validate_labels,
        ) = (
            to_var(validate_input_ids),
            to_var(validate_attention_mask),
            to_var(validate_type_ids),
            to_var(validate_labels),
        )

        validate_outputs = model(
            validate_input_ids, validate_attention_mask, validate_type_ids
        )
        if validate_outputs.ndim == 1:
            validate_outputs = validate_outputs.unsqueeze(0)
        validate_argmax = torch.argmax(validate_outputs, 1)
        vali_loss = criterion(validate_outputs, validate_labels)
        # domain_loss = criterion(domain_outputs, event_labels)
        # _, labels = torch.max(validate_labels, 1)
        validate_accuracy = (
            (validate_labels == validate_argmax.squeeze()).float().mean()
        )
        vali_cost_vector.append(vali_loss.item())
        # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
        validate_acc_vector_temp.append(validate_accuracy.item())
    validate_acc = np.mean(validate_acc_vector_temp)
    valid_acc_vector.append(validate_acc)
    model.train()
    print(
        "Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f."
        % (
            epoch + 1,
            epochs,
            np.mean(cost_vector),
            np.mean(class_cost_vector),
            np.mean(acc_vector),
            validate_acc,
        )
    )

    if validate_acc > best_validate_acc:
        best_validate_acc = validate_acc
        if not os.path.exists(output_file):
            os.mkdir(output_file)

        best_validate_dir = (
            output_file + str(time.time()) + "_epoch" + str(epoch + 1) + ".pkl"
        )

        torch.save(model.state_dict(), best_validate_dir)

    duration = time.time() - start_time


# Test the Model
print("testing model")
model = BertClassify()
model.load_state_dict(torch.load(best_validate_dir))
print(best_validate_dir)
#    print(torch.cuda.is_available())
if torch.cuda.is_available():
    model.cuda()
model.eval()
test_score = []
test_pred = []
test_true = []
for i, (test_input_ids, test_attention_mask, test_type_ids, test_labels) in enumerate(
    test_loader
):
    test_input_ids, test_attention_mask, test_type_ids, test_labels = (
        to_var(test_input_ids),
        to_var(test_attention_mask),
        to_var(test_type_ids),
        to_var(test_labels),
    )

    test_outputs = model(test_input_ids, test_attention_mask, test_type_ids)
    if test_outputs.ndim == 1:
        test_outputs = test_outputs.unsqueeze(0)
    test_argmax = torch.argmax(test_outputs, 1)
    if i == 0:
        test_score = to_np(test_outputs.squeeze())
        test_pred = to_np(test_argmax.squeeze())
        test_true = to_np(test_labels.squeeze())
    else:
        if test_outputs.size() == torch.Size([1, 2]):
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)
        else:
            test_score = np.concatenate(
                (test_score, to_np(test_outputs.squeeze())), axis=0
            )
            test_pred = np.concatenate(
                (test_pred, to_np(test_argmax.squeeze())), axis=0
            )
            test_true = np.concatenate(
                (test_true, to_np(test_labels.squeeze())), axis=0
            )

test_accuracy = metrics.accuracy_score(test_true, test_pred)
test_f1 = metrics.f1_score(test_true, test_pred)
test_precision = metrics.precision_score(test_true, test_pred)
test_recall = metrics.recall_score(test_true, test_pred)
test_score_convert = [x[1] for x in test_score]
test_aucroc = metrics.roc_auc_score(test_true, test_score_convert)
test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

# print(f'percent:{dataset_percent}')
print(dataset_name)
if is_augmentation:
    print("augmentation data:" + str(len(augmentation_dataset)))
else:
    print("No augmentation")
print(
    "Classification Acc: %.4f, precision: %.4f, recall: %.4f, F1: %.4f, AUC-ROC: %.4f"
    % (test_accuracy, test_precision, test_recall, test_f1, test_aucroc)
)
print("confusion_matrix:")
print(test_confusion_matrix)
