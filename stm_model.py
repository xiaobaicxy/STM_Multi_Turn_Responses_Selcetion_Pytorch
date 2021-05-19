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
        # x: [batch_size*turn_num, seq_len, embed_dim]
        # lengths: [batch_size*turn_num]
        # x: [batch_size*candidates_set_size, seq_len, embed_dim]
        # lengths: [batch_size*candidates_set_size]
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

        # out的文本长度为该batch中最长文本的有效长度
        # 某个batch中最长文本的有效长度都可能小于预先设置的max_seq_len，为了后续计算，需要将其填充
        if out_unsort.size(1) < seq_len:
            pad_tensor = Variable(torch.zeros(out_unsort.size(0), seq_len - out_unsort.size(1), out_unsort.size(2))).to(out_unsort)
            out_unsort = torch.cat([out_unsort, pad_tensor], 1)

        return out_unsort

# matching block
class Conv3DModel(nn.Module):
    def __init__(self, in_channels, candidates_size, num_classes):
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

        self.classifier = nn.Sequential(nn.Linear(candidates_size * 24 * 2 * 4 * 4, 100), # 输入维度与输入的turn_num和词向量维度以及cnn实现方式有关
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(100, num_classes)
                                )
    
    def forward(self, x, batch_size):
        # x: [batch_size*candidates_set_size, rnn_layer_num, turn_num, context_seq_len, candidate_seq_len]
        out = self.conv1(x)
        out = self.conv2(out)
        # print("before fc layer shape: ", out.shape)
        flatten = out.view(batch_size, -1)
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
        self.candidates_set_size = config.candidates_set_size
        self.rnn_layer_num = config.rnn_layer_num
        self.num_classes = config.num_classes
        self.directions = config.directions
        self.dropout = config.dropout

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        if pretrain_embedding is not None:
            self.embedding.weight.data.copy_(pretrain_embedding)
            self.embedding.weight.requires_grad = True
        
        self.context_encoder = nn.ModuleList(
                                        [BiGRUModel(in_dim=config.embed_dim,
                                                    out_dim=config.hidden_size,
                                                    dropout=config.dropout) 
                                        for layer_id in range(config.rnn_layer_num)]
                                        )
        
        self.candidate_encoder = nn.ModuleList(
                                        [BiGRUModel(in_dim=config.embed_dim,
                                                    out_dim=config.hidden_size,
                                                    dropout=config.dropout)
                                        for layer_id in range(config.rnn_layer_num)]
                                        )
        
        self.matching_block = Conv3DModel(config.rnn_layer_num+1, config.candidates_set_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, contexts_indices, candidates_indices):
        # contexts_indices: [batch_size, turn_num, seq_length]
        # candidates_indices: [batch_size, candidates_set_size, seq_length]

        batch_size = contexts_indices.size(0)
        contexts_seq_len = (contexts_indices != 0).sum(dim=-1).long() # [batch_size, turn_num]
        contexts_turns_num = (contexts_seq_len != 0).sum(dim=-1).long() # [batch_size]
        candidates_seq_len = (candidates_indices != 0).sum(dim=-1).long() # [batch_size, candidates_set_size]
        candidates_set_size = (candidates_seq_len != 0).sum(dim=-1).long() # [batch_size]

        contexts_embed = self.embedding(contexts_indices) # [batch_size, turn_num, seq_length, embed_dim]
        contexts_embed = self.dropout(contexts_embed)

        candidates_embed = self.embedding(candidates_indices) # [batch_size, candidates_set_size, seq_length, embed_dim]
        candidates_embed = self.dropout(candidates_embed)

        contexts_all_inputs_len = contexts_seq_len.view(-1) # [batch_size*turn_num]
        candidates_all_inputs_len = candidates_seq_len.view(-1) # [batch_size*candidates_set_size]

        assert self.embed_dim == self.hidden_size * self.directions
        
        all_layers_contexts_hiddens = [contexts_embed.view(-1, self.max_seq_len, self.embed_dim)]
        all_layers_candidates_hiddens = [candidates_embed.view(-1, self.max_seq_len, self.embed_dim)]
        for i in range(self.rnn_layer_num):
            contexts_hiddens = self.context_encoder[i](all_layers_contexts_hiddens[-1], contexts_all_inputs_len) # [batch_size*turn_num, seq_length, embed_dim]
            candidates_hiddens = self.candidate_encoder[i](all_layers_candidates_hiddens[-1], candidates_all_inputs_len) # [batch_size*candidates_set_size, seq_length, embed_dim]
            all_layers_contexts_hiddens.append(contexts_hiddens)
            all_layers_candidates_hiddens.append(candidates_hiddens)

        for i in range(self.rnn_layer_num + 1):
            all_layers_contexts_hiddens[i] = all_layers_contexts_hiddens[i].view(-1, self.max_turn_num, self.max_seq_len, self.embed_dim) # [batch_size, turn_num, seq_length, embed_dim]
            all_layers_candidates_hiddens[i] = all_layers_candidates_hiddens[i].view(-1, self.candidates_set_size, self.max_seq_len, self.embed_dim) # [batch_size, candidates_set_size, seq_length, embed_dim]
        
        all_layers_contexts_hiddens = torch.stack(all_layers_contexts_hiddens, dim=0) # [rnn_layer_num+1, batch_size,  turn_num, seq_len, hidden_size*directions]
        all_layers_candidates_hiddens = torch.stack(all_layers_candidates_hiddens, dim=0) # [rnn_layer_num+1, batch_size, candidates_set_size, seq_len, hidden_size*2]
        
        # [rnn_layer_num+1, batch_size, turn_num, candidates_set_size, context_seq_len, candidate_seq_len]
        M = torch.einsum('nbtph, nbcqh->nbtcpq', (all_layers_contexts_hiddens, all_layers_candidates_hiddens)) / math.sqrt(self.embed_dim)
        M = M.permute(1, 3, 0, 2, 4, 5).contiguous() # [batch_size, candidates_set_size, rnn_layer_num+1, turn_num, context_seq_len, candidate_seq_len]
        # [batch_size*candidates_set_size, rnn_layer_num, turn_num, context_seq_len, candidate_seq_len]
        M = M.contiguous().view(-1, self.rnn_layer_num+1, self.max_turn_num, self.max_seq_len, self.max_seq_len) 
        
        logits = self.matching_block(M, batch_size) # [batch_size, num_classes]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

if __name__ == "__main__":
    pass