from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        scaling_factor = torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attention = attention / scaling_factor
        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, V)
        return output


class MultiCrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)
        self.norm = sqrt(all_head_size)
        self.attention = CalculateAttention()

    def forward(self, x, y):
        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, self.num_heads, self.h_size)
        k_s = self.linear_k(y).view(batch_size, self.num_heads, self.h_size)
        v_s = self.linear_v(y).view(batch_size, self.num_heads, self.h_size)
        output = self.attention(q_s, k_s, v_s)
        output = output.transpose(1, 2).contiguous().view(batch_size, self.num_heads * self.h_size)
        output = self.linear_output(output)
        return output


class PredModel(nn.Module):
    def __init__(self, cfg):
        super(PredModel, self).__init__()
        self.drug_conv_1 = nn.Linear(100+708+4096, 1024)
        self.drug_conv_2 = nn.Linear(1024, 256)
        self.prot_conv_1 = nn.Linear(400+1493+4096, 1024)
        self.prot_conv_2 = nn.Linear(1024, 256)
        self.cra_1 = MultiCrossAttention(
            hidden_size=cfg.hidden_size,
            all_head_size=cfg.all_head_size,
            head_num=cfg.head_num
        )
        self.cra_2 = MultiCrossAttention(
            hidden_size=cfg.hidden_size,
            all_head_size=cfg.all_head_size,
            head_num=cfg.head_num
        )
        self.cls_1 = nn.Linear(
            in_features=100+708+400+1493,
            out_features=cfg.hidden_size
        )
        self.cls_2 = nn.Linear(
            in_features=cfg.hidden_size,
            out_features=cfg.label_num
        )
        self.relu = nn.ReLU()
        

    def forward(self, features):
        drug_features, prot_features = features
        dt_features = torch.cat((drug_features, prot_features), dim=-1)
        drug_features = self.cra_1(prot_features, drug_features)
        prot_features = self.cra_2(drug_features, prot_features)
        pred = self.relu(self.cls_1(dt_features))
        pred = self.cls_2(pred)
        return dt_features, pred
