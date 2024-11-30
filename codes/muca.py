import os
import os.path as osp
import time
import argparse
import torch
import random
import re
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
from metrics import eval_metrics
import warnings
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

atoms = ['N', 'CA', 'C', 'O', 'R', 'CB']
n_atoms = len(atoms)
atom_idx = {atom:atoms.index(atom) for atom in atoms}

def BLOSUM62(fastas, **kw):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # X
    }
    encodings = []
    for sequence in fastas:
        code = []
        for aa in sequence:
            code = blosum62[aa]
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr

def BINA(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    encodings = []
    for sequence in fastas:
        for aa in sequence:
            if aa not in AA:
                aa = 'X'
            if aa == 'X':
                code = [0 for _ in range(len(AA))]
                encodings.append(code)
                continue
            code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr


class CMPMDataset(object):
    def __init__(self, sample_file, tokenizer, pdb_dir=None, feature=None, pssm_dir=None, structure_features=None):
        self.seq_list = []
        self.label_list = []
        self.feature_list = []
        self.pdb_entry_list = []
        self.input_ids_list = []
        self.attention_mask_list = []
        self.pssm_dir = pssm_dir  
        self.tokenizer = tokenizer
        self.feature = feature
        self.structure_features = structure_features
        self.win_size = None
        self.protein_info = []

        print(f"Loading data from {sample_file}")
        seqlist = [record for record in SeqIO.parse(sample_file, "fasta")]
        print(f"Number of sequences in FASTA file: {len(seqlist)}")

        for record in tqdm(seqlist):
            seq = str(record.seq)
            desc = record.id.split('|')
            name, label = desc[0], int(desc[1])
            if len(desc) == 5:
                pos, length = int(desc[3]), int(desc[4])
            else:
                pos, length = 0, 0
            
            self.protein_info.append((name, pos))
            
            encoded = self.tokenizer.encode_plus(
                ' '.join(seq),
                add_special_tokens=True,
                padding='max_length',
                return_token_type_ids=False,
                pad_to_max_length=True,
                truncation=True,
                max_length=len(seq),
                return_tensors='pt'
            )
            
            self.input_ids_list.append(encoded['input_ids'].flatten())
            self.attention_mask_list.append(encoded['attention_mask'].flatten())
            
            pssm = self._read_pssm(os.path.join(pssm_dir, f"{name.split('_')[0]}.txt"), pos)
            self.feature_list.append(pssm)
            self.label_list.append(int(label))
            self.seq_list.append(seq)
            self.pdb_entry_list.append(name)
            
            if self.win_size is None:
                self.win_size = len(seq)

        assert len(self.seq_list) == len(self.label_list) == len(self.pdb_entry_list) == len(self.feature_list), \
            "Error: Lengths of seq_list, label_list, pdb_entry_list, and feature_list do not match!"

    def __getitem__(self, index):
        if index >= len(self.seq_list):
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.seq_list)}")
        
        seq = self.seq_list[index]
        name, position = self.protein_info[index]
        protein_name = name.split('_')[0]
        
        input_ids = self.input_ids_list[index]
        attention_mask = self.attention_mask_list[index]
        
        sequence_feature = self._get_encoding(seq, name, self.feature)
        
        structure_feature = self.structure_features.get((protein_name, position), None)
        if structure_feature is None:
            structure_feature = np.zeros(110)
        
        if sequence_feature.ndim == 1:
            sequence_feature = sequence_feature.reshape(-1, 1)
        elif sequence_feature.ndim > 2:
            sequence_feature = sequence_feature.reshape(sequence_feature.shape[0], -1)
        
        structure_feature = np.tile(structure_feature, (sequence_feature.shape[0], 1))

        combined_feature = np.concatenate([sequence_feature, structure_feature], axis=1)

        return input_ids, attention_mask, torch.tensor(combined_feature, dtype=torch.float), torch.tensor(self.label_list[index])

    def __len__(self):
        return len(self.pdb_entry_list)

    def _get_encoding(self, seq, sample_id, feature=[BLOSUM62, BINA, 'PSSM']):
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
        max_len = len(sample)
        all_fea = []
        for encoder in feature:
            if encoder == 'PSSM':
                pssm_file = os.path.join(self.pssm_dir, f"{sample_id.split('_')[0]}.txt")
                pssm = self._read_pssm(pssm_file, 0)
                all_fea.append(pssm)
            else:
                fea = encoder([sample])
                assert fea.shape[0] == max_len
                all_fea.append(fea)
        combined_features = np.hstack(all_fea)
        if combined_features.ndim == 1:
            combined_features = combined_features.reshape(1, -1)
        return combined_features

    def _read_pssm(self, pssm_file, center_pos):
        pssm = []
        with open(pssm_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Last') or line.startswith('position-specific') or line.startswith('PSSM') or line.startswith(' '):
                    continue
                parts = line.split()
                if not parts[0].isdigit():
                    continue
                if len(parts) < 24:
                    continue
                try:
                    scores = list(map(float, parts[2:22]))
                    pssm.append(scores)
                except ValueError:
                    print(f"无法转换为浮点数的行: {line}")
                    continue
        pssm = np.array(pssm) 
        window_size = 51
        half_window = window_size // 2
        start = center_pos - half_window - 1 
        end = center_pos + half_window  

        if start < 0:
            pad_width = abs(start)
            pssm = np.pad(pssm, ((pad_width, 0), (0, 0)), 'constant')
            start = 0
        if end > pssm.shape[0]:
            pad_width = end - pssm.shape[0]
            pssm = np.pad(pssm, ((0, pad_width), (0, 0)), 'constant')
            end = pssm.shape[0]
        
        fragment_pssm = pssm[start:end, :] 
        
        if fragment_pssm.shape[0] < window_size:
            pad_length = window_size - fragment_pssm.shape[0]
            fragment_pssm = np.pad(fragment_pssm, ((0, pad_length), (0, 0)), 'constant')
        
        fragment_pssm = (fragment_pssm - np.mean(fragment_pssm, axis=0)) / (np.std(fragment_pssm, axis=0) + 1e-6)
        
        return fragment_pssm

    def __getitem__(self, index):
        if index >= len(self.seq_list):
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.seq_list)}")
        seq = self.seq_list[index]
        name, position = self.protein_info[index]
        
        # 移除 "_trunk" 及其后面的内容
        protein_name = name.split('_')[0]
        
        # print(f"\nDebug: Processing protein {name} (matched as {protein_name}) at position {position}")
        
        input_ids = self.input_ids_list[index]
        attention_mask = self.attention_mask_list[index]
        
        sequence_feature = self._get_encoding(seq, self.feature)
        
        # 尝试获取结构特征，使用处理后的蛋白质名称和原始位置
        structure_feature = self.structure_features.get((protein_name, position), None)
        if structure_feature is None:
            # print(f"Warning: No structure feature found for {protein_name} at position {position}")
            structure_feature = np.zeros(110)
        
        # 确保 sequence_feature 是 2D 的，形状为 [seq_len, features]
        if sequence_feature.ndim == 1:
            sequence_feature = sequence_feature.reshape(-1, 1)
        elif sequence_feature.ndim > 2:
            sequence_feature = sequence_feature.reshape(sequence_feature.shape[0], -1)
        
        # 将 structure_feature 扩展到与 sequence_feature 相同的序列长度
        structure_feature = np.tile(structure_feature, (sequence_feature.shape[0], 1))
        
        # print(f"Index: {index}")
        # print(f"Original Protein ID: {name}")
        # print(f"Matched Protein ID: {protein_name}")
        # print(f"Position: {position}")
        # print(f"Structure feature found: {'Yes' if structure_feature is not None else 'No'}")
        # print(f"Sequence feature shape: {sequence_feature.shape}")
        # print(f"Structure feature shape: {structure_feature.shape}")
        # print(f"First few values of structure feature: {structure_feature[0, :5]}...")
        
        # 合并序列特征和结构特征
        combined_feature = np.concatenate([sequence_feature, structure_feature], axis=1)
        # print(f"combined_feature: {combined_feature.shape}")
        # print(f"Debug: combined_feature shape = {combined_feature.shape}")
        
        return input_ids, attention_mask, torch.tensor(combined_feature, dtype=torch.float), torch.tensor(self.label_list[index])
    def __len__(self):
        return len(self.pdb_entry_list)

    def _get_encoding(self, seq, sample_id, feature=[BLOSUM62, BINA, 'PSSM'], base_save_path='features'):
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
        max_len = len(sample)
        all_fea = []
        for encoder in feature:
            if encoder == 'PSSM':
                pssm = self.feature_list[-1]  
                all_fea.append(pssm)
            else:
                fea = encoder([sample])
                assert fea.shape[0] == max_len
                all_fea.append(fea)
        combined_features = np.hstack(all_fea)
        # 确保返回的是二维数组
        if combined_features.ndim == 1:
            combined_features = combined_features.reshape(1, -1)
        return combined_features


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def detach(x):
    return x.cpu().detach().numpy().squeeze()

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, win_size, out_dim=64, kernel_size=3, strides=1, dropout=0.2):
        super(CNNEncoder, self).__init__()
        if kernel_size == 9:
            if win_size < (kernel_size - 1) * 2:
                kernel_size = 7
        self.kernel_size = kernel_size
        self.strides = strides
        self.emd = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        # out_channels - (kernel_size - 1) * 2 - 2
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.emd(x.long())
        x = torch.permute(x, (0,2,1))
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input_x, axis=1):
        input_size = input_x.size()
        trans_input = input_x.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input_x):
        x = torch.tanh(self.fc1(input_x))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		
        return attention

class PLMEncoder(nn.Module):
    def __init__(self, BERT_encoder, out_dim, PLM_dim=1280, dropout=0.2):
        super(PLMEncoder, self).__init__()
        self.bert = BERT_encoder # BertModel.from_pretrained("Rostlab/prot_bert")
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv1d(PLM_dim, out_dim, kernel_size=3, stride=1, padding='same')
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, input_ids, attention_mask):
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        self.bertout = pooled_output
        imput = pooled_output.permute(0, 2, 1) # shape: (Batch, 1024, length)
        conv1_output = F.relu(self.bn1(self.conv1(imput)))  # shape: (Batch, out_channel, length)
        output = self.dropout(conv1_output)
        prot_out = torch.mean(output, axis=2, keepdim=True) # shape: (Batch, out_channel, 1)
        prot_out = prot_out.permute(0, 2, 1)  # shape: (Batch, 1, out_channel)
        return prot_out

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, n_layers, dropout, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.emd_layer = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim*2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        if self.bidirectional:
            self.lstm2 = nn.LSTM(hidden_dim * 4, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.lstm2 = nn.LSTM(hidden_dim * 2, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        emd = self.emd_layer(x.long())
        self.raw_emd = emd
        output, (final_hidden_state, final_cell_state) = self.lstm1(emd.float()) # shape: (Batch, length, 128)
        output = self.dropout1(output)
        lstmout2, (_, _) = self.lstm2(output) # shape: (Batch, length, 64)
        bi_lstm_output = self.dropout2(lstmout2)
        bi_lstm_output = torch.mean(bi_lstm_output, axis=1, keepdim=True) # shape: (Batch, 1, 64)
        return bi_lstm_output

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, win_size, kernel_size=9, strides=1, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.hidden_channels = hidden_dim
        if win_size < (kernel_size - 1) * 2:
            kernel_size = 7
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = torch.nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, 51, 171]
        x = x.permute(0, 2, 1)  # [Batch, 171, 51]
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x.unsqueeze(1)

class MetaDecoder(nn.Module):
    def __init__(self, combined_dim, dropout=0.5):
        super(MetaDecoder, self).__init__()
        self.fc1 = nn.Linear(combined_dim, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: [batch_size, 1, combined_dim]
        x = x.squeeze(1)  # [batch_size, combined_dim]
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)  # [batch_size, 1]
        return x.squeeze(-1)  # [batch_size]

class CMPMNet(nn.Module):
    def __init__(self, BERT_encoder, vocab_size, encoder_list=['cnn','lstm','plm'], PLM_dim=1024, win_size=51, structure_dim=110, embedding_dim=32, fea_dim=1061, hidden_dim=64, out_dim=32, n_layers=1, dropout=0.2, bidirectional=True):
        super(CMPMNet, self).__init__()
        dim_list = []
        self.encoder_list = encoder_list
        if 'cnn' in self.encoder_list:
            self.cnn_encoder = CNNEncoder(vocab_size, embed_dim=embedding_dim, hidden_dim=hidden_dim, win_size=win_size, out_dim=out_dim, dropout=dropout)
            dim_list.append(out_dim)
            
        if 'plm' in self.encoder_list:
            self.plm_encoder = PLMEncoder(BERT_encoder=BERT_encoder, out_dim=128, PLM_dim=PLM_dim, dropout=dropout)
            dim_list.append(128)
        if 'lstm' in self.encoder_list:
            self.lstm_encoder = BiLSTMEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
            if bidirectional:
                dim_list.append(out_dim*2)
            else:
                dim_list.append(out_dim)
        if 'fea' in self.encoder_list:
            total_fea_dim = 171
            self.fea_encoder = FeatureEncoder(input_dim=total_fea_dim, hidden_dim=hidden_dim, out_dim=out_dim, win_size=win_size, kernel_size=9, strides=1, dropout=dropout)
            dim_list.append(out_dim)

        combined_dim = sum(dim_list)
        self.decoder = MetaDecoder(combined_dim)

    def forward(self, input_ids, attention_mask, feature=None, g_data=None):
        fuse_x = []
        self.model_emd = {}

        if 'cnn' in self.encoder_list:
            cnn_out = self.cnn_encoder(input_ids).unsqueeze(1)
            fuse_x.append(cnn_out)
            self.model_emd['cnn_out'] = detach(cnn_out)
            # print(f"CNN output shape: {cnn_out.shape}")
            # print(f"CNN output shape: {cnn_out.shape}")

        if 'plm' in self.encoder_list:
            prot_out = self.plm_encoder(input_ids, attention_mask)
            fuse_x.append(prot_out)
            self.model_emd['bert_out'] = detach(self.plm_encoder.bertout)
            self.model_emd['plm_out'] = detach(prot_out)
            # print(f"PLM output shape: {prot_out.shape}")
            # print(f"PLM output shape: {prot_out.shape}")

        if 'lstm' in self.encoder_list:
            bi_lstm_output = self.lstm_encoder(input_ids)
            fuse_x.append(bi_lstm_output)
            self.model_emd['raw'] = detach(self.lstm_encoder.raw_emd)
            self.model_emd['lstm_out'] = detach(bi_lstm_output)
            # print(f"LSTM output shape: {bi_lstm_output.shape}")
            # print(f"LSTM output shape: {bi_lstm_output.shape}")

        if 'fea' in self.encoder_list and feature is not None:
            # print(f"Feature shape before fea_encoder: {feature.shape}")
            fea_out = self.fea_encoder(feature)
            fuse_x.append(fea_out)
            self.model_emd['fea_in'] = detach(feature)
            self.model_emd['fea_out'] = detach(fea_out)
            # print(f"fea output shape: {fea_out.shape}")
            # print(f"Feature shape after fea_encoder: {fea_out.shape}")

        # print(f"Number of tensors in fuse_x: {len(fuse_x)}")
        # for i, tensor in enumerate(fuse_x):
        #     print(f"Shape of tensor {i} in fuse_x: {tensor.shape}")

        # Concatenate all features if there's more than one
        if len(fuse_x) > 1:
            fused_features = torch.cat(fuse_x, dim=2)
        else:
            fused_features = fuse_x[0]
        # print(f"Fused features shape: {fused_features.shape}")
        logit = self.decoder(fused_features)
        # print(f"Logit shape after decoder: {logit.shape}")
        return logit
    # def _extract_embedding(self):
    #     print(f'Extract embedding from', list(model_emd.keys()))
    #     return self.model_emd

def detach(x):
    return x.cpu().detach().numpy().squeeze()

def random_run(SEED=3047):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # print(f"Random seed initialization: {SEED}!")

def train_one_epoch(loader, model, device, optimizer, criterion):
    model.train()
    train_step_loss = []
    train_total_acc = 0
    step = 1
    train_total_loss = 0
    for ind,(input_ids, attention_mask, feature, label) in enumerate(loader):
        # print(f"Batch {ind}, Feature shape: {feature.shape}")
        # 获取当前批次的蛋白质信息
        # batch_proteins = [loader.dataset.protein_info[i] for i in range(ind*loader.batch_size, (ind+1)*loader.batch_size) if i < len(loader.dataset)]
        
        # for i, (name, position) in enumerate(batch_proteins):
        #     print(f"Batch {ind}, Item {i}: Protein {name} at position {position}")
        #     print(f"Feature shape: {feature[i].shape}")
        #     print(f"Feature content:\n{feature[i]}")
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device).float()
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        # print(f"Pred shape: {pred.shape}, Label shape: {label.shape}")
        loss = criterion(pred, label.float())
        acc = ((pred > 0.5).float() == label).float().mean()
        # logits = pred.view(-1)  # 确保 logits 是 [batch_size] 维度
        # print(f"logits shape: {logits.shape}, label shape: {label.shape}")  # 添加这行来调试
        # loss = criterion(logits, label.float())
        # acc = (logits.round() == label).float().mean()
        # print(f"Training ... Step:{step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_step_loss.append(loss.item())
        train_total_acc += acc
        step += 1
    avg_train_acc = train_total_acc / step
    avg_train_loss = train_total_loss / step
    return train_step_loss, avg_train_acc, avg_train_loss, step

def test_binary(model, loader, criterion, device):
    model.eval()
    criterion.to(device)
    test_probs = []
    test_targets = []
    valid_total_acc = 0
    valid_total_loss = 0
    valid_step = 1
    for ind,(input_ids, attention_mask, feature, label) in enumerate(loader):
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        # print(f"Valid step:{valid_step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        valid_total_loss += loss.item()
        valid_total_acc += acc.item()
        test_probs.extend(logits.cpu().detach().numpy())
        test_targets.extend(label.cpu().detach().numpy())
        valid_step += 1

    avg_valid_loss = valid_total_loss / valid_step
    avg_valid_acc = valid_total_acc / valid_step
    # print(f"Avg Valid Loss: {avg_valid_loss:.4f} | Avg Valid Acc: {avg_valid_acc:.4f}")
    test_probs = np.array(test_probs)
    test_probs = 1 / (1 + np.exp(-test_probs))
    test_targets = np.array(test_targets)
    # print("test_probs shape:", test_probs.shape)
    # print("test_probs sample:", test_probs[:5])
    # print("test_labels shape:", test_targets.shape)
    # print("test_labels sample:", test_targets[:5])
    # print("Unique values in test_labels:", np.unique(test_targets))
    return test_probs, test_targets, avg_valid_loss, avg_valid_acc
def arg_parse():
    # argument parser
    parser = argparse.ArgumentParser()
    # directory and file settings
    root_dir = '/mnt/data/hrren/CM/Datasets/nr40/win51_copy'
    parser.add_argument("--project_name", default='CM', type=str,
                            help="Project name for saving model checkpoints and best model. Default:`CMPM`.")
    parser.add_argument("--train", default=osp.join(root_dir, 'CM_train_ratio_all.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_train_ratio_all.fa'`.")
    parser.add_argument("--test", default=osp.join(root_dir, 'CM_text_ratio_all.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_test_ratio_1.fa'`.")
    parser.add_argument("--model", default='/mnt/data/hrren/CM/Models', type=str,
                        help="Directory for model storage and logits. Default:``.")
    parser.add_argument("--result", default='/mnt/data/hrren/CM/', type=str,
                        help="Result directory for model training and evaluation. Default:`result/CM`.")
    parser.add_argument("--PLM", default='/mnt/data/hrren/esm2_t33_650M_UR50D', type=str, 
                        help="PLM directory. Default:``.")
    parser.add_argument("--pdb_dir", default='/mnt/data/hrren/CM/PDB', type=str,
                        help="PLM directory. Default:`Datasets/Structure/PDB`.")
    parser.add_argument('--epoch', type=int, default=200, metavar='[Int]',
                        help='Number of training epochs. (default:200)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='[Float]',
                        help='Learning rate. (default:1e-4)')
    parser.add_argument('--batch', type=int, default=64, metavar='[Int]',
                        help='Batch size cutting threshold for Dataloader.(default:256)')
    parser.add_argument('--cpu', '-cpu', type=int, default=8, metavar='[Int]',
                        help='CPU processors for data loading.(default:8).')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--emd_dim', '-ed', type=int, default=32, metavar='[Int]',
                        help='Word embedding dimension.(default:32).')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=64, metavar='[Int]',
                        help='Hidden dimension.(default:64).')
    parser.add_argument('--out_dim', '-od', type=int, default=32, metavar='[Int]',
                        help='Out dimension for each track.(default:32).')
    parser.add_argument('--dropout', '-dp', type=float, default=0.50, metavar='[Float]',
                        help='Dropout rate.(default:0.5).')
    parser.add_argument('--encoder', type=str, default='cnn,fea,plm', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,lstm,fea,plm. (default:`cnn,lstm,fea,plm`)')
    parser.add_argument('--seed', type=int, default=43, metavar='[Int]',
                        help='Random seed. (default:2024)')
    parser.add_argument('--patience', type=int, default=10000, metavar='[Int]',                        
                        help='Early stopping patience. (default:50)')
    parser.add_argument("--pssm_dir", default='/mnt/data2024/hrren/pssm/pssm_result', type=str,
                        help="Directory containing PSSM files.")
    return parser.parse_args()


def load_structure_features(file_path):
    try:
        # 读取CSV文件，跳过前两列
        df = pd.read_csv(file_path, header=None, skiprows=1, on_bad_lines='warn')
        
        # print(f"Loaded dataframe shape: {df.shape}")
        # print(f"First few rows:\n{df.head()}")
        
        # 使用第1和第2列（原始文件中的第3和第4列）作为索引
        df.set_index([0, 1], inplace=True)
        
        # 删除索引列
        df = df.iloc[:, 2:]
        
        # 创建特征字典，处理蛋白质ID和位点
        feature_dict = {(index[0].split('_')[0], index[1]+1): row.values for index, row in df.iterrows()}
        
        # print(f"Loaded {len(feature_dict)} unique protein structure features")
        if feature_dict:
            sample_key = next(iter(feature_dict.keys()))
            sample_value = feature_dict[sample_key]
            # print(f"Sample key: {sample_key}")
            # print(f"Shape of sample feature: {sample_value.shape}")
            # print(f"First few values of sample feature: {sample_value[:5]}")
        else:
            print("No features loaded.")
        return feature_dict
    
    except Exception as e:
        print(f"Error loading structure features: {e}")
        return {}

if __name__=='__main__':
    save_model=True
    args = arg_parse()
    print(args)
    project = args.project_name
    SEED = args.seed
    random_run(SEED)
    embedding_dim = args.emd_dim
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    lr = args.learning_rate
    num_epochs = args.epoch
    batch_size = args.batch
    cpu = args.cpu
    gpu = args.gpu
    model_dir = osp.join(args.model, f'{project}')
    os.makedirs(model_dir, exist_ok=True)
    result_dir = osp.join(args.result, f'{project}')
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pretrained_model = args.PLM # '/home/hrren/project/pretrained_LM/prot_bert/'
    encoder_list = args.encoder.split(',') # ['cnn','lstm','fea','plm','gnn']
    n_layers = 1
    dropout = args.dropout
    patience = args.patience
    if 'bert' in pretrained_model:
        PLM_dim = 1024
    elif 'esm' in pretrained_model:
        PLM_dim = 1280
    elif 'esm2' in pretrained_model:
        PLM_dim = 1280
    
    manual_fea = [BLOSUM62, BINA, 'PSSM']
    fea_dim = 64
    structure_dim = 110

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)
    if 'plm' in encoder_list:
        BERT_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=True, output_attentions=False).to(device)
    else:
        BERT_encoder = None

    pdb_dir = args.pdb_dir
    pssm_dir = args.pssm_dir

    structure_features = load_structure_features('/mnt/data/hrren/CM/Structure_learn/structure_features.csv')
    # print("Structure features loaded.")
    # print(f"Number of proteins with structure features: {len(structure_features)}")
    # print(f"Shape of first structure feature: {next(iter(structure_features.values())).shape}")
    # print(f"Sample keys: {list(structure_features.keys())[:5]}")  # 打印前5个键
    # print(f"Sample values: {list(structure_features.values())[0][:5]}")  # 打印第一个特征的前5个值
    train_file = args.train
    train_ds = CMPMDataset(
        sample_file=train_file,
        tokenizer=tokenizer,
        pdb_dir=pdb_dir,
        feature=manual_fea,
        pssm_dir=pssm_dir,
        structure_features=structure_features
    )
    test_file = args.test
    test_ds = CMPMDataset(
        sample_file=test_file,
        tokenizer=tokenizer,
        pdb_dir=pdb_dir,
        feature=manual_fea,
        pssm_dir=pssm_dir,
        structure_features=structure_features
    )
    # print(f"test_ds attributes: {dir(test_ds)}")
    # print(f"test_ds length: {len(test_ds)}")
    window_size = test_ds.win_size

    # print(f"Training dataset: {len(train_ds)}   |Testing dataset: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu,
        prefetch_factor=2
    )
    valid_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu,
        prefetch_factor=2
    )

    model = CMPMNet(
        BERT_encoder=BERT_encoder,
        vocab_size=tokenizer.vocab_size,
        encoder_list=encoder_list,
        PLM_dim=PLM_dim,
        win_size=window_size,
        embedding_dim=embedding_dim,
        fea_dim=fea_dim,
        structure_dim=110,  # 更新为110维
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # model.apply(weight_init)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Model Trainable Parameter: "+ str(params/1024/1024) + 'Mb' + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss().to(device)
    result_list = []
    all_train_loss_list = []
    best_loss = 1000
    best_epoch = 0
    best_model_state = None
    best_auc = 0
    patience = 0
    max_patience = args.patience
    desc=['Project', 'Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
    best_auc = 0
    for epoch in range(num_epochs):
        # Training
        start = time.perf_counter()
        train_step_loss, train_acc, train_loss, step = train_one_epoch(train_loader, model, device, optimizer, criterion)
        all_train_loss_list.extend(train_step_loss)
        end = time.perf_counter()
        print(f"Epoch {epoch+1} | {(end - start):.4f}s | Train | Loss: {train_loss: .6f}| Train acc: {train_acc:.4f}")
        start = time.perf_counter()
        test_probs, test_labels, valid_loss, valid_acc = test_binary(model, valid_loader, criterion, device)
        end = time.perf_counter()
        print(f"Epoch {epoch+1} | {(end - start):.4f}s | Test | Test loss: {valid_loss:.6f}| Test acc: {valid_acc:.4f}")
        acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_labels)
        # pred_bi = np.abs(np.ceil(test_probs - th_))
        # cm = confusion_matrix(test_labels, pred_bi)
        # tn,fp,fn,tp = cm.ravel()
        result_info = [project, epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
        result_list.append(result_info)
        print_results(result_info, desc)
        if valid_loss > best_loss:
            patience += 1
        else:
            patience = 0  
            best_loss = valid_loss  
        if patience > max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        

        # 在训练循环中记录最佳模型状态
        if auc_ > best_auc:
            best_auc = auc_
            best_epoch = epoch
            best_result = result_info
            best_test_probs = test_probs
            best_test_labels = test_labels
            best_model_state = model.state_dict().copy()  # 保存模型状态的副本

        # 在训练循环结束后添加
        if save_model and best_model_state is not None:
            best_model_path = osp.join(model_dir, f'save2best_{project}_model_auc_{best_auc:.4f}.pt')
            torch.save(best_model_state, best_model_path)
            print(f"\nSaved best model to {best_model_path}")
            print(f"Best model achieved AUC: {best_auc:.4f} at epoch {best_epoch}")
    print('\nBest result:\n')
    print_results(best_result, desc)
    # Save training step loss
    loss_df = pd.DataFrame(all_train_loss_list)
    loss_df.columns = ['Loss']
    loss_df.to_csv(osp.join(result_dir, f'2best_auc_{best_auc:.4f}_train_step_loss.csv'), sep='\t')
    
    
    # Save predicted logits and labels
    logit_df = pd.DataFrame({
        'Label': best_test_labels,
        'Probability': best_test_probs
    })
    logit_df.to_csv(osp.join(result_dir, f"2best_auc_{best_auc:.4f}_predictions.csv"), sep='\t', index=False)


    epoch_df = pd.DataFrame([best_result])
    epoch_df.columns = desc
    epoch_df.to_csv(osp.join(result_dir, f'2best_auc_{best_auc:.4f}_metrics.csv'), sep='\t', index=False)

    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(best_test_labels, best_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(osp.join(result_dir, f'2best_auc_{best_auc:.4f}_roc_curve.png'))
    plt.close()

    # Plot and save PR curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(best_test_labels, best_test_probs)
    pr_auc = average_precision_score(best_test_labels, best_test_probs)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(osp.join(result_dir, f'2best_auc_{best_auc:.4f}_pr_curve.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_loss_list)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Steps')
    plt.grid(True)
    plt.savefig(osp.join(result_dir, f'2best_auc_{best_auc:.4f}_training_loss.png'))
    plt.close()

    print(f"\nSaved all results with best AUC = {best_auc:.4f} in {result_dir}")
    print(f"Files saved:")
    print(f"- best_auc_{best_auc:.4f}_train_step_loss.csv")
    print(f"- best_auc_{best_auc:.4f}_predictions.csv")
    print(f"- best_auc_{best_auc:.4f}_metrics.csv")
    print(f"- best_auc_{best_auc:.4f}_roc_curve.png")
    print(f"- best_auc_{best_auc:.4f}_pr_curve.png")
    print(f"- best_auc_{best_auc:.4f}_training_loss.png")
    print(f"ROC curve and training loss curve have been saved in {result_dir}")
