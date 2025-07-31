import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. 数据预处理
def load_data(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence, current_labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
            else:
                char, tag = line.split()  # 假设数据格式为"字符 标签"
                current_sentence.append(char)
                current_labels.append(tag)
    return sentences, labels

# 2. 构建词汇表和标签表
def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

def build_tag_dict(labels):
    tag_dict = {}
    for sentence_labels in labels:
        for tag in sentence_labels:
            if tag not in tag_dict:
                tag_dict[tag] = len(tag_dict)
    return tag_dict


# 3. 数据集类
class WordSegDataset(Dataset):
    def __init__(self, sentences, labels, vocab, tag_dict, window_size=3):
        self.data = []
        for sentence, sentence_labels in zip(sentences, labels):
            for i in range(len(sentence)):
                # 获取上下文窗口
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                context = sentence[start:end]
                # 填充或截断
                if len(context) < 2 * window_size + 1:
                    context = ['<PAD>'] * (2 * window_size + 1 - len(context)) + context
                # 转换为ID
                context_ids = [vocab.get(c, vocab['<UNK>']) for c in context]
                tag_id = tag_dict[sentence_labels[i]]
                self.data.append((context_ids, tag_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


# 4. 模型定义
class PCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_size, num_classes):
        super(PCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, filter_size, padding=filter_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, 1)
        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, num_filters)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # 5. 训练和评估
    def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    def evaluate_model(model, test_loader, device):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # 6. 主函数
        def main():
            # 加载数据
            train_sentences, train_labels = load_data('train.txt')
            test_sentences, test_labels = load_data('test.txt')

            # 构建词汇表和标签表
            vocab = build_vocab(train_sentences)
            tag_dict = build_tag_dict(train_labels)

            # 创建数据集和数据加载器
            train_dataset = WordSegDataset(train_sentences, train_labels, vocab, tag_dict)
            test_dataset = WordSegDataset(test_sentences, test_labels, vocab, tag_dict)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # 初始化模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = PCNN(
                vocab_size=len(vocab),
                embedding_dim=100,
                num_filters=64,
                filter_size=3,
                num_classes=len(tag_dict)
            ).to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练和评估
            train_model(model, train_loader, criterion, optimizer, device, epochs=10)
            evaluate_model(model, test_loader, device)

        if __name__ == '__main__':
            main()