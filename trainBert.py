import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
print(torch.cuda.device_count())
import pandas as pd
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import re

# data_xls = pd.read_excel('/home/qzh/Value_cn/friendliness.xlsx')
# data_xls.to_csv('/home/qzh/Value_cn/friendliness.csv', encoding='utf-8')

data = pd.read_csv("/home/qzh/Value_cn/5000_run.csv", delimiter=',', encoding='utf-8',
                   header=0)
# data = pd.read_csv("/home/qzh/Value_cn/quest150new.csv", delimiter=',', encoding='utf-8',
#                    header=0)             
# print(data.head())
data = data.sample(frac=1).reset_index(drop=True)
train_data = data.iloc[:int(len(data) * 0.7), :]
val_data = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9), :]
test_data = data.iloc[int(len(data) * 0.9):, :]
model_type='hflchinese-roberta-wwm-ext-large'

# Patriotism_scr	Dedication_scr	Integrity_scr	Friendliness_scr	Prosperity_scr	Democracy_scr	Civility_scr	Harmony_scr	Freedom_scr	Equality_scr	Justice_scr	Rule of Law_scr

model_type='bert-base-chinese'
MODEL_NAME = '/data/qzh/val_cn/'+model_type
tokenizer= BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)


class EarlyStopping:
    def __init__(self,value,patience=5, min_delta=0, path='/data/qzh/val_cn/check_point/'+model_type+'/'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.value=value
        self.best_score = None
        self.stop = False
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path+self.value+'.pt')
        self.best_model = deepcopy(model.state_dict())


class ArticleFakeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len,val_idx,text_idx):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.val_idx=val_idx
        self.text_idx=text_idx
        # TypeError: '>' not supported between instances of 'str' and 'float'
        # 所以要先转换成float
        # ValueError: could not convert string to float: 'ERROR'
        # 所以要先过滤掉不符合条件的数据ERROR
        # self.data = self.data[(self.data.iloc[:, 6] != 'ERROR')]
        # self.data.iloc[:, 1] = self.data.iloc[:, 1].astype(float)
        # self.data = self.data[(self.data.iloc[:, 1] > 0.33) | (self.data.iloc[:, 1] < -0.33)]
        # self.data = self.data[(self.data.iloc[:, 1] > 0.3) | (self.data.iloc[:, 1] < -0.3)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, self.text_idx]
        # print("属地化",self.text_idx,text)
        target = self.data.iloc[index,self.val_idx]
        # print("属地化",self.val_idx,target)
        # print(target)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }


text_idx=0

# 定义损失函数
criterion = CrossEntropyLoss()

# # 选择优化器
optimizer = Adam(model.parameters(), lr=1e-5)

def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, token_type_ids=token_type_ids, attention_mask=mask, labels=targets)
        # print(outputs.logits, targets)
        loss = criterion(outputs.logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):

            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            
            outputs = model(ids, token_type_ids=token_type_ids, attention_mask=mask, labels=targets)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.logits.argmax(-1).cpu().numpy().tolist())
    return outputs, targets, fin_outputs, fin_targets


def training(value):
    # 初始化早期停止对象
    print("保存至：",'/data/qzh/val_cn/check_point/'+model_type+'/'+value)
    early_stopping = EarlyStopping(value,patience=5, path='/data/qzh/val_cn/check_point/'+model_type+'/')
    EPOCHS = 20
    best_loss = np.inf
    for epoch in range(EPOCHS):
        train(epoch)
        outputs, targets, fin_outputs, fin_targets = validate(epoch)
        # 统计fin_outputs中0和1的个数并输出
        # print("fin_outputs中0和1的个数：", np.bincount(fin_outputs))
        # 计算损失函数
        loss = criterion(outputs.logits, targets)

        if loss < best_loss:
            best_loss = loss

        early_stopping(loss, model)

        # 计算一些评估指标
        accuracy = accuracy_score(fin_targets, fin_outputs)
        f1 = f1_score(fin_targets, fin_outputs, average='weighted')
        precision = precision_score(fin_targets, fin_outputs, average='weighted')
        
        recall = recall_score(fin_targets, fin_outputs, average='weighted')
        # print(f"Accuracy Score = {accuracy}")
        # print(f"F1 Score = {f1}")
        # print(f"Precision Score = {precision}")
        # print(f"Recall Score = {recall}")

        if early_stopping.stop:
            # print("Early stopping")
            break

# for val_idx in [2,3,5,6,7,8,9,10]:
#     # val_idx=2
#     val_list=["Patriotism","Dedication","Integrity","Friendliness","Prosperity","Democracy","Civility","Harmony","Freedom","Equality","Justice","Rule of Law"]
#     value=val_list[val_idx]

#     train_set = ArticleFakeDataset(train_data, tokenizer, 128,val_idx+2,text_idx+1)
#     training_loader = DataLoader(train_set, batch_size=64)

#     validation_set = ArticleFakeDataset(val_data, tokenizer, 128,val_idx+2,text_idx+1)
#     validation_loader = DataLoader(validation_set, batch_size=64)
#     # print(data)
#     test_set = ArticleFakeDataset(test_data, tokenizer, 128,val_idx+2,text_idx+1)
#     # print(test_set)
#     test_loader = DataLoader(test_set, batch_size=64)

#     training(value)

def evaluate(data_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_pro=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader, 0)):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            # print("看这里",targets)
            outputs = model(ids, token_type_ids=token_type_ids, attention_mask=mask, labels=targets)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).tolist())
            fin_pro.extend(torch.softmax(outputs.logits, dim=1).tolist())
            # fin_pro=np.array(fin_pro)

            
    return outputs, targets, fin_outputs, fin_targets,np.array(fin_pro)


def test(test_loader,value):
# Rule of Law_scr,Rule of Law_scr,Rule of Law_scr,Rule of Law_scr
# 加载性能最好的模型权重
    early_stopping = EarlyStopping(value,patience=5, path='/data/qzh/val_cn/check_point/'+model_type+'/')
    print("加载模型：",'/data/qzh/val_cn/check_point/'+model_type+'/'+value+'.pt')
    model.load_state_dict(torch.load('/data/qzh/val_cn/check_point/'+model_type+'/'+value+'.pt'))
    # print(test("我是是","Rule of Law"))
    # 创建测试集数据加载器
    #使用训练集评估模型
    # outputs, targets, fin_outputs, fin_targets = evaluate(training_loader)
    # # 统计fin_outputs中0和1的个数并输出
    # print("fin_outputs中0和1的个数：", np.bincount(fin_outputs))

    #使用验证集评估模型
    # outputs, targets, fin_outputs, fin_targets = evaluate(validation_loader)
    # # 统计fin_outputs中0和1的个数并输出
    # print("fin_outputs中0和1的个数：", np.bincount(fin_outputs))

    # # 使用测试集评估模型
    outputs, targets, fin_outputs, fin_targets,fin_pro = evaluate(test_loader)
    # print(np.mean(fin_pro[:, 1]))
    # print(fin_outputs)
    # 计算各项评估指标
    # print(fin_targets, fin_outputs)
    accuracy = accuracy_score(fin_targets, fin_outputs)
    f1 = f1_score(fin_targets, fin_outputs)
    precision = precision_score(fin_targets, fin_outputs)
    recall = recall_score(fin_targets, fin_outputs)

    # 统计fin_outputs中0和1的个数并输出
    print(value,"fin_outputs中0和1的个数：", np.bincount(fin_outputs))
    
    print("fin_targets", np.bincount(fin_targets))
    # print(f"Test Accuracy Score = {accuracy}")
    # print(f"Test F1 Score = {f1}")
    # print(f"Test Precision Score = {precision}") 
    # print(f"Test Recall Score = {recall}")
    print(f"{value}&{accuracy:.3f} & {precision:.3f}&{recall:.3f}& {f1:.3f}")
    # print(accuracy+f1,precision,recall)
    # print("hflchinese-roberta-wwm-ext-large:",val_list[val_idx])
    return str(np.mean(fin_pro[:, 1]))

text_list=["chatglm3-6b","Baichuan4","glm-4","claude-3-opus-20240229","gpt4","llama2"]

val_list=["Patriotism",	"Dedication","Integrity","Friendliness","Prosperity","Democracy","Civility","Harmony","Freedom","Equality","Justice","Rule of Law"]
# val_list=["Democracy","Freedom"] 
for val_idx in [10]:
    # val_idx=0
    text_idx=0

    value=val_list[val_idx]

    train_set = ArticleFakeDataset(train_data, tokenizer, 128,val_idx+2,1)
    training_loader = DataLoader(train_set, batch_size=64)

    validation_set = ArticleFakeDataset(val_data, tokenizer, 128,val_idx+2,1)
    validation_loader = DataLoader(validation_set, batch_size=64)
    # print(data)
    test_set = ArticleFakeDataset(test_data, tokenizer, 128,val_idx+2,1)
    # print(test_set)
    test_loader = DataLoader(test_set, batch_size=64)

    # training(value)
    test(test_loader,value)


# for text_idx in range(len(text_list)):
#     output=text_list[text_idx]

#     for val_idx in range(len(val_list)):
#         # train_set = ArticleFakeDataset(train_data, tokenizer, 128,val_idx,text_idx+1)
#         # training_loader = DataLoader(train_set, batch_size=64)

#         # validation_set = ArticleFakeDataset(val_data, tokenizer, 128,val_idx,text_idx+1)
#         # validation_loader = DataLoader(validation_set, batch_size=64)
#         # print(data)
#         test_set = ArticleFakeDataset(data, tokenizer, 128,val_idx+2,1)
#         # print(test_set)
#         test_loader = DataLoader(test_set, batch_size=64)
#         # test(test_loader,val_list[val_idx])


#         output+=" & "+test(test_loader,val_list[val_idx])
#         # print(output)
#     # break
#     print(output)

