import random
import numpy as np
import torch
import json

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
import tqdm

from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all

set_seed(42)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data(path):
    texts = []
    labels = []

    with open(path) as fd:
        for line in fd:
            info = json.loads(line)
            texts.append(info["text"])
            label = 1 if info["spam"] == "Y" else 0
            labels.append(label)

    return texts, labels

def train_model(model, train_loader, val_loader, device, save_path, epochs=3, learnin_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learnin_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    best_accuracy = 0
    training_stats = []

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            labels = batch['labels'].to(device)

            # forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)

            train_correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f'保存最佳模型，验证准确率: {val_accuracy:.2f}%')

        # 记录训练统计
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })

    return training_stats

def test_model(model, test_loader, device):
    """测试模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Test Phase"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=['高质', '低质'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    print(f'\n测试结果:')
    print(f'准确率: {accuracy:.4f}')
    print(f'\n分类报告:')
    print(class_report)
    print(f'\n混淆矩阵:')
    print(conf_matrix)
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

def predict_spam(model, tokenizer, text, device, max_length=512):
    """预测batch文本的label"""
    model.eval()
    
    encoding = tokenizer(
        text,
        padding=True, 
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        _, prediction = torch.max(logits, 1)
    
    pred = prediction.detach().cpu().numpy().flatten()# 0代表高质量，1代表低质量
    confidence = probabilities[:, 0].detach().cpu().numpy().flatten() # 高质量的置信度
    return pred, confidence

if __name__ == "__main__":
    # 设置 GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 模型参数设置
    NUM_LABELS = 2
    BATCH_SIZE = 64
    MAX_LENGTH = 512
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    PRETRAINED_PATH = 'bert-base-chinese'
    CKPT_PATH = "saved_model/best_bert_classifier.pth" 
    DATA_PATH = "data/output_quality_full.json"

    # 加载中文BERT tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_PATH,
        num_labels=NUM_LABELS,  # 二分类
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(device)

    # 准备数据
    texts, labels = create_data(DATA_PATH)

    # 分割数据集
    train_texts, other_texts, train_labels, other_labels = train_test_split(
        texts, labels, test_size=0.3
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        other_texts, other_labels, test_size=0.5
    )

    print(f'训练集大小: {len(train_texts)}')
    print(f'验证集大小: {len(val_texts)}')
    print(f'测试集大小: {len(test_texts)}')

    # 创建数据集和数据加载器
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练模型
    print("开始训练模型...")
    training_stats = train_model(model, train_loader, val_loader, device, CKPT_PATH, EPOCHS, LEARNING_RATE)
    json.dump(training_stats, open("./data/bert_training_stats.json", "w"))

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load(CKPT_PATH))

    # 测试模型
    test_results = test_model(model, test_loader, device)

    # 待过滤数据设置
    FILTER_BATCH_SIZE = 512
    FILTER_THRESHOLD = 0.98
    FILTER_PATH = "./data/baike_qa_train.json" 

    KEEP_OUTPUT_PATH = "./data/baike_qa_output.json"
    DROPED_OUTPUT_PATH = "./data/baike_qa_filtered.json"

    # 过滤新数据 
    print("\n开始过滤新数据...")
    test_sentences = []
    fd = open(FILTER_PATH)
    batch = []
    for line in fd:
        info = json.loads(line)
        if len(batch) < FILTER_BATCH_SIZE:
            info["title"] = info["title"].replace("\n", "")
            info["answer"] = info["answer"].replace("\n", "")
            batch.append(info["title"] + "\n" + info["answer"])
        else:
            test_sentences.append(batch)
            batch = []
    if batch:
        test_sentences.append(batch)

    fw = open(KEEP_OUTPUT_PATH, "w")
    fd = open(DROPED_OUTPUT_PATH, "w")
    total_count, keep_count = 0, 0
    for sentence in tqdm(test_sentences, total=len(test_sentences)):
        spam, confidence = predict_spam(model, tokenizer, sentence, device)
        total_count += 1
        for s, c, t in zip(spam, confidence, sentence): 
            title, answer = t.split("\n")
            json_str = {"title": title, "answer": answer}
            if s == 0 and c > FILTER_THRESHOLD:
                fw.write(json.dumps(json_str, ensure_ascii=False) + "\n")
                keep_count += 1
            else:
                fd.write(json.dumps(json_str, ensure_ascii=False) + "\n")
    print(f"total doc num: {total_count}, keep doc num: {keep_count}")