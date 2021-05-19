# -*- coding: UTF-8 -*-​ 
import torch
import copy
import os
import torch.nn as nn

from data_processor import DataProcessor
from stm_model import STMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(512)

class Config:
    def __init__(self):
        self.data_path = {
            "train": "./dataset/dialogue/ubuntu_train_subtask_1.json",
            # "train": "./dataset/dialogue/ubuntu_dev_subtask_1.json", # for debug
            "dev": "./dataset/dialogue/ubuntu_dev_subtask_1.json"
        }
        self.vocab_path = "./dataset/dialogue/vocab.txt"
        self.model_save_path = "./stm_model_param.pkl"
        self.update_vocab = True

        self.vocab_size = 50000
        self.embed_dim = 300
        self.hidden_size = 150
        self.rnn_layer_num = 3
        self.directions = 2
        
        self.max_turn_num = 10
        self.max_seq_len = 50
        self.candidates_set_size = 2 #Rn@k: n=2，10，100, k=1

        self.batch_size = 12
        self.epochs = 1000
        self.dropout = 0.2
        self.lr = 0.0002
        self.num_classes = self.candidates_set_size

        self.device = device

def eval(model, loss_func, dev_loader):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for contexts, candidates, labels in dev_loader:
        contexts = contexts.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        preds = model(contexts, candidates)
        loss = loss_func(preds, labels)

        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)

        corrects += torch.sum(preds==labels).item()
        loss_val += loss.item() * contexts.size(0)
    dev_loss = loss_val / len(dev_loader.dataset)
    dev_acc = corrects / len(dev_loader.dataset)
    print("Dev Loss: {}, Dev Acc: {}".format(dev_loss, dev_acc))
    return dev_acc

def train(model, train_loader, dev_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for contexts, candidates, labels in train_loader:
            contexts = contexts.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            preds = model(contexts, candidates)
            loss = loss_func(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds==labels).item()
            loss_val += loss.item() * contexts.size(0)
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)

        if epoch % 20 == 0:
            print("----------epoch/epochs: {}/{}----------".format(epoch, epochs))
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            val_acc = eval(model, loss_func, dev_loader)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    return model

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.data_path)
    train_examples = processor.get_train_examples(config.candidates_set_size)
    dev_examples = processor.get_dev_examples(config.candidates_set_size)

    train_dataset_tokens = processor.get_dataset_tokens(train_examples)
    dev_dataset_tokens = processor.get_dataset_tokens(dev_examples)
    
    if not os.path.exists(config.vocab_path) or config.update_vocab:
        processor.create_vocab(train_dataset_tokens, config.vocab_path)
    
    train_dataset_indices, vocab_size = processor.get_dataset_indices(train_dataset_tokens,
                                                                     config.vocab_path,
                                                                     config.vocab_size)
    dev_dataset_indices, _ = processor.get_dataset_indices(dev_dataset_tokens,
                                                            config.vocab_path,
                                                            config.vocab_size)
    config.vocab_size = vocab_size # 实际词表大小

    train_tensor = processor.create_tensor_dataset(train_dataset_indices, config.max_turn_num, config.max_seq_len)
    dev_tensor = processor.create_tensor_dataset(dev_dataset_indices, config.max_turn_num, config.max_seq_len)
    
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=config.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_tensor, batch_size=config.batch_size, shuffle=False)

    model = STMModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_func = nn.BCELoss()
    model = train(model, train_loader, dev_loader, optimizer, loss_func, config.epochs)

    torch.save(model.state_dict(), config.model_save_path)