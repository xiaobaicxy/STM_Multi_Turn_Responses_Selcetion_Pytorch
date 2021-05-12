# -*- coding: UTF-8 -*-​ 
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(512)

# representation module: dialog(context)/candidate encoder
class BiGRUModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, batch_first=True, directions=2, dropout=0.2):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=in_dim,
                        hidden_size=out_dim,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        bidirectional=(directions==2),
                        dropout=dropout)
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len, in_dim]
        # batch_size = batch_size*turns_num/batch_size*candidate_set_size
        # in_dim = embed_dim/hidden_size
        # lengths: [batch_size]
        seq_len = x.size(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        x_sort = x.index_select(0, idx_sort)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths_sort = lengths_sort.to("cpu") # pack_padded_sequence的lengths参数必须是cpu上的参数

        #有的batch中，对话的轮数可能都小于max_turn_num，前面用全padding将其进行了填充，需要将这部分的有效长度置为1
        for i in range(len(lengths_sort)):
            if lengths_sort[i] == 0:
                lengths_sort[i] = 1

        x_pack = nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort, batch_first=True)
        out_pack, h_n =self.gru(x_pack)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)

        out_unsort = out.index_select(0, idx_unsort)

        # 某个batch中最长文本的有效长度都可能小于预先设置的max_seq_len，为了后续计算，需要将其填充
        if out_unsort.size(1) < seq_len:
            pad_tensor = Variable(torch.zeros(out_unsort.size(0), seq_len - out_unsort.size(1), out_unsort.size(2))).to(out_unsort)
            out_unsort = torch.cat([out_unsort, pad_tensor], 1)

        return out_unsort # [batch_size, seq_len, hidden_size*directions]

# matching block
class Conv3DModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3DModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=12, kernel_size=(3,3,3), bias=False),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,3,3))
                                )
        
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(3,3,3), bias=False),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))
                                )

        self.classifier = nn.Sequential(nn.Linear(384, 100),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(100, 1)
                                )
    
    def forward(self, x):
        # x: [batch_size*candidate_set_size, rnn_layer_num, turn_num, context_seq_len, candidate_seq_len]
        out = self.conv1(x)
        out = self.conv2(out)
        # print("before fc layer shape: ", out.shape)
        flatten = out.view(out.size(0), -1)
        out = self.classifier(flatten)
        return out
    
class STMModel(nn.Module):
    def __init__(self, config, pretrain_embedding=None):
        super(STMModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size
        self.max_turn_num = config.max_turn_num
        self.max_seq_len = config.max_seq_len
        self.candidate_set_size = config.candidate_set_size
        self.rnn_layer_num = config.rnn_layer_num

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        if pretrain_embedding is not None:
            self.embedding.weight.data.copy_(pretrain_embedding)
            self.embedding.weight.requires_grad = True
        
        self.context_encoder = nn.ModuleList(
                                        [BiGRUModel(in_dim=config.embed_dim if layer_id==0 else config.hidden_size,
                                                    out_dim=config.hidden_size,
                                                    dropout=config.dropout) 
                                        for layer_id in range(config.rnn_layer_num)]
                                        )
        
        self.candidate_encoder = nn.ModuleList(
                                        [BiGRUModel(in_dim=config.embed_dim if layer_id==0 else config.hidden_size,
                                                    out_dim=config.hidden_size,
                                                    dropout=config.dropout)
                                        for layer_id in range(config.rnn_layer_num)]
                                        )
        
        self.matching_block = Conv3DModel(config.rnn_layer_num, 36)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, contexts_indices, candidates_indices, labels_idx):
        # contexts_indices: [batch_size, turn_num, seq_length]
        # candidates_indices: [batch_size, candidate_set_size, seq_length]
        # label_idx: [batch_size]

        contexts_seq_len = (contexts_indices != 0).sum(dim=-1).long() # [batch_size, turn_num]
        contexts_turns_num = (contexts_seq_len != 0).sum(dim=-1).long() # [batch_size]
        candidates_seq_len = (candidates_indices != 0).sum(dim=-1).long() # [batch_size, candidate_set_size]
        candidates_set_size = (candidates_seq_len != 0).sum(dim=-1).long() # [batch_size]

        contexts_embed = self.embedding(contexts_indices) # [batch_size, turn_num, seq_length, embed_dim]
        contexts_embed = self.dropout(contexts_embed)

        candidates_embed = self.embedding(candidates_indices) # [batch_size, candidate_set_size, seq_length, embed_dim]
        candidates_embed = self.dropout(candidates_embed)

        contexts_all_inputs_len = contexts_seq_len.view(-1) # [batch_size*turn_num]
        candidates_all_inputs_len = candidates_seq_len.view(-1) # [batch_size*candidate_set_size]

        all_layers_contexts_hiddens = [contexts_embed]
        all_layers_candidates_hiddens = [candidates_embed]
        for layer_id in range(self.rnn_layer_num):
            if layer_id == 0:
                contexts_inputs = all_layers_contexts_hiddens[-1].view(-1, self.max_seq_len, self.embed_dim) # [batch_size*turn_num, seq_len, embed_dim]
                candidates_inputs = all_layers_candidates_hiddens[-1].view(-1, self.max_seq_len, self.embed_dim) # [batch_size*candidate_set_size, seq_len, embed_dim]
            else:
                contexts_inputs = all_layers_contexts_hiddens[-1].view(-1, self.max_seq_len, self.hidden_size) # [batch_size*turn_num, seq_len, hidden_size]
                candidates_inputs = all_layers_candidates_hiddens[-1].view(-1, self.max_seq_len, self.hidden_size) # [batch_size*candidate_set_size, seq_len, hidden_size]

            contexts_hiddens = self.context_encoder[layer_id](contexts_inputs, contexts_all_inputs_len)
            candidates_hiddens = self.candidate_encoder[layer_id](candidates_inputs, candidates_all_inputs_len)

            all_layers_contexts_hiddens.append(contexts_hiddens.view(-1, self.max_turn_num, self.max_seq_len, self.hidden_size*2))
            all_layers_candidates_hiddens.append(candidates_hiddens.view(-1, self.candidate_set_size, self.max_seq_len, self.hidden_size*2))

        # 去掉embedding层
        all_layers_contexts_hiddens = all_layers_contexts_hiddens[1:] 
        all_layers_candidates_hiddens = all_layers_candidates_hiddens[1:] 

        # [rnn_layer_num, batch_size,  turn_num, seq_len, hidden_size*2]
        all_layers_contexts_hiddens = torch.stack(all_layers_contexts_hiddens, dim=0)
        # [rnn_layer_num, batch_size, candidate_set_size, seq_len, hidden_size*2]
        all_layers_candidates_hiddens = torch.stack(all_layers_candidates_hiddens, dim=0) 
        
        # [rnn_layer_num, batch_size, turn_num, candidate_set_size, context_seq_len, candidate_seq_len]
        M = torch.einsum('nbtph, nbcqh->nbtcpq', (all_layers_contexts_hiddens, all_layers_candidates_hiddens)) / math.sqrt(self.hidden_size)
        # [batch_size*candidate_set_size, rnn_layer_num, turn_num, context_seq_len, candidate_seq_len]
        M = M.contiguous().view(-1, self.rnn_layer_num, self.max_turn_num, self.max_seq_len, self.max_seq_len) 
        
        logits = self.matching_block(M)
        logits = logits.view(-1, self.candidate_set_size)
        loss = self.loss_func(logits, labels_idx)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds

if __name__ == "__main__":
    pass