import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from torch import nn

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup


# data_xls = pd.read_excel('/home/qzh/Value_cn/5000.xlsx', index_col=0)
# data_xls.to_csv('/home/qzh/Value_cn/5000_5.csv', encoding='utf-8')

header = pd.read_csv("/home/qzh/Value_cn/5000_run.csv", nrows=1)

train_data = pd.read_csv("/home/qzh/Value_cn/5000_run.csv").head(3500)
val_data = pd.read_csv("/home/qzh/Value_cn/5000_run.csv", skiprows=3501).head(1000)
val_data.columns = header.columns
test_data = pd.read_csv("/home/qzh/Value_cn/5000_run.csv", skiprows=4501).head(500)
# test_data = pd.read_csv("/home/qzh/Value_cn/5_dimension.csv", nrows=1)
test_data.columns = header.columns
print(test_data)

# print(test_data)
# print(train_data)
# print(val_data)
# print(test_data)
# # 读取数据
# train_data = pd.read_csv("/home/qzh/Value_cn/test.csv")
# val_data = pd.read_csv("/home/qzh/Value_cn/test.csv")
# test_data = pd.read_csv("/home/qzh/Value_cn/test.csv")
# 假设BERT的预训练模型名为'bert-base-uncased'
MODEL_NAME = '/data/qzh/val_cn/bert-base-chinese'
# MODEL_NAME = '/data/qzh/val_cn/bert-base-chinese'
NUM_LABELS = 12  # bias, gender, religion, race, lgbtq, political
labels = [label for label in train_data.keys() if label not in ['text']]
del(labels[0])

print(labels)


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, labels,max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'][index]
        labels=self.labels
        
        labels_batch = {k: self.data[k] for k in self.data.keys() if k in labels}
        # print(labels_batch)
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros(len(labels))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[idx] = labels_batch[label][index]

        labels = labels_matrix.tolist()
        labels = torch.tensor(labels, dtype=torch.float32).squeeze()
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        # print(text,labels)

        return input_ids, attention_mask, labels


# 初始化tokenizer和模型
print(MODEL_NAME)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# print(model)
# 划分数据集
train_dataset = CustomDataset(train_data, tokenizer,labels)
# print(train_dataset[0])
val_dataset = CustomDataset(val_data, tokenizer,labels)
test_dataset = CustomDataset(test_data, tokenizer,labels)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 将模型移到设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)


# 定义损失函数
def loss_fn(outputs, targets):
    class_weight = torch.tensor([8.0,7.0,13.0,39.0,6.0,47.0,16.0,7.0,82.0,15.0,26.0,8.0], dtype=torch.float32).to(device)
    return torch.nn.BCEWithLogitsLoss(weight=class_weight)(outputs, targets)
 


# 训练模型
NUM_EPOCHS = 15
best_val_score = 0.0
patience = 5  # Number of epochs to wait for improvement
num_epochs_no_improvement = 0
LEARNING_RATE = 1e-5
# 初始化optimizer和scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                            num_training_steps=total_steps)


# 定义训练器
def train(epoch):
    model.train()
    for _, (input_ids, attention_mask, labels) in tqdm(enumerate(train_loader, 0)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask).logits
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        # 更新学习率 (Learning rate warm-up)
        scheduler.step()


# 定义验证函数
def validation(data_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, (input_ids, attention_mask, labels) in tqdm(enumerate(data_loader, 0)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = labels.to(device)
            # print(input_ids, attention_mask)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

def evaluate_metrics(targets, outputs):
    targets = np.array(targets)
    outputs = np.array(outputs)
    outputs = (outputs > 0.5).astype(int)

    overall_accuracy = accuracy_score(targets, outputs)
    overall_f1 = f1_score(targets, outputs, average='micro')
    overall_precision = precision_score(targets, outputs, average='micro')
    overall_recall = recall_score(targets, outputs, average='micro')

    class_accuracies = {}
    class_f1s={}
    class_precisions={}
    class_recalls={}
    for i, label in enumerate(labels):
        class_targets = targets[:, i]
        class_outputs = outputs[:, i]

        class_accuracy = accuracy_score(class_targets, class_outputs)
        class_accuracies[label] = class_accuracy

        class_f1 = f1_score(class_targets, class_outputs)
        class_f1s[label] = class_f1

        class_precision = precision_score(class_targets, class_outputs)
        class_precisions[label] = class_precision

        class_recall = recall_score(class_targets, class_outputs)
        class_recalls[label] = class_recall


    return overall_accuracy, overall_f1, overall_precision, overall_recall, class_accuracies,class_f1s,class_precisions,class_recalls

def test(text):
    model.eval()

    fin_outputs = []
    with torch.no_grad():

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        # print(input_ids,attention_mask)
        outputs = model(input_ids, attention_mask=attention_mask).logits
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    
    return fin_outputs[0]


for epoch in range(NUM_EPOCHS):
    train(epoch)

    # 验证模型并计算评价指标
    val_outputs, val_targets = validation(val_loader)
    val_accuracy, val_f1, val_precision, val_recall, val_class_accuracies,val_class_f1s,val_class_precisions,val_class_recalls = evaluate_metrics(val_targets, val_outputs)
    print(
        f"Validation Accuracy: {val_accuracy:.4f}, F1-score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    for label, accuracy in val_class_accuracies.items():
        print(f"Val {label} Accuracy: {accuracy:.4f}")
    # 保存在验证集上效果最好的模型
    if val_f1 > best_val_score:
        best_val_score = val_f1
        torch.save(model.state_dict(), "/data/qzh/val_cn/bert-base-chinese/_best_model_noAug_5000.pt")
        num_epochs_no_improvement = 0
    else:
        num_epochs_no_improvement += 1

    # 判断是否提前停止训练
    if num_epochs_no_improvement >= patience:
        print("Early stopping triggered. Training stopped.")
        break

# 加载效果最好的模型
model.load_state_dict(torch.load("/data/qzh/val_cn/bert-base-chinese/_best_model_noAug_5000.pt"))
model.to(device)

# print(test("我爱中国"))
# 在测试集上进行评估
test_outputs, test_targets = validation(test_loader)
test_accuracy, test_f1, test_precision, test_recall, test_class_accuracies ,test_class_f1s,test_class_precisions,test_class_recalls= evaluate_metrics(test_targets, test_outputs)

# print(
#     f"Test Overall Accuracy: {test_accuracy:.4f}, F1-score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

for label, accuracy in test_class_accuracies.items():
    print(f"Test {label} Accuracy: {accuracy:.4f}")

for label, accuracy in test_class_f1s.items():
    print(f"Test {label} f1: {accuracy:.4f}")

for label, accuracy in test_class_precisions.items():
    print(f"Test {label} precisions: {accuracy:.4f}")

for label, accuracy in test_class_recalls.items():
    print(f"Test {label} recalls: {accuracy:.4f}")

