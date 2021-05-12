# -*- coding: UTF-8 -*-​ 
import torch
import copy

from data_processor import DataProcessor
from stm_model import STMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(512)

class Config:
    def __init__(self):
        self.data_path = {
            "train": "/home/fuyong/workspace/dataset/dialogue/ubuntu_train_subtask_1.json",
            "dev": "/home/fuyong/workspace/dataset/dialogue/ubuntu_dev_subtask_1.json"
        }
        self.vocab_path = "/home/fuyong/workspace/dataset/dialogue/vocab.txt"
        self.model_save_path = "./stm_model_param.pkl"

        self.vocab_size = 50000
        self.embed_dim = 300
        self.hidden_size = 150
        self.rnn_layer_num = 3

        self.max_turn_num = 9
        self.max_seq_len = 50
        self.candidate_set_size = 100

        self.batch_size = 12
        self.epochs = 10000
        self.dropout = 0.2
        self.lr = 0.001

def eval(model, dev_loader):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for contexts, candidates, labels in dev_loader:
        contexts = contexts.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        loss, preds = model(contexts, candidates, labels)
        corrects += torch.sum(preds==labels).item()
        loss_val += loss.item() * contexts.size(0)
    dev_loss = loss_val / len(dev_loader.dataset)
    dev_acc = corrects / len(dev_loader.dataset)
    print("Dev Loss: {}, Dev Acc: {}".format(dev_loss, dev_acc))
    return dev_acc

def train(model, train_loader, dev_loader, optimizer, epochs):
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

            loss, preds = model(contexts, candidates, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects += torch.sum(preds==labels).item()
            loss_val += loss.item() * contexts.size(0)
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)

        if epoch % 200 == 0:
            print("----------epoch/epochs: {}/{}----------".format(epoch, epochs))
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            val_acc = eval(model, dev_loader)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    return model

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config.data_path)
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()

    train_dataset_tokens = processor.get_dataset_tokens(train_examples)
    dev_dataset_tokens = processor.get_dataset_tokens(dev_examples)
    
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
    model = train(model, train_loader, dev_loader, optimizer, config.epochs)

    torch.save(model.state_dict(), config.model_save_path)