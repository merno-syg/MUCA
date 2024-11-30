import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import glob
import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP
import pandas as pd
import torch
from pathlib import Path
import torch.nn.functional as F
import pickle

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

# 设置全局变量
device_set = "cuda:0"
device = torch.device(device_set)
structure_data = '/mnt/data/hrren/CM/PDB'
Atom_target = "NZ"  # Atom
train_data_address = '/mnt/data/hrren/CM/Datasets/nr40/win51_copy/CM_train_ratio_1.fa'
test_data_address = '/mnt/data/hrren/CM/Datasets/nr40/win51_copy/CM_text_ratio_1.fa'
PTM_name = train_data_address.split('/')[-2]

def normalize_protein_id(protein_id):
    return protein_id.split('_')[0]

def pdb_split(line):
    atom_type = "CNOS$"
    aa_trans_DICT = {
        'ALA': 'A', 'CYS': 'C', 'CCS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', }
    aa_type = "ACDEFGHIKLMNPQRSTVWYX"
    
    Atom_order = int(line[6:11].strip()) - 1
    atom = line[11:16].strip()
    amino = line[16:21].strip()
    AA_order = int(line[22:28].strip()) - 1
    x = float(line[28:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    atom_single_name = line.strip()[-1]
    
    atom_single_name_vec = np.zeros(len(atom_type))
    atom_single_name_vec[atom_type.find(atom_single_name)] = 1
    AA_single_name_vec = np.zeros(len(aa_type))
    AA_single_name_vec[aa_type.find(aa_trans_DICT[amino])] = 1
    
    atom_feature_combine = np.concatenate((atom_single_name_vec.reshape(1, -1), AA_single_name_vec.reshape(1, -1)), axis=1)
    
    return atom, amino, AA_order, Atom_order, x, y, z, atom_feature_combine
def Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine, modify_site_pdb,modify_site_pdb_NZ_index,dseq=3,dr=10,dlong=5,k=10 ):
    #每个修饰位点原子特征的列表
    modify_site_pdb_ATOM_combine_node_edge_feature = []
    # 遍历每一个位点
    for AA in  modify_site_pdb:
        # 获取当前氨基酸的特定原子的索引
        atom_index=modify_site_pdb_NZ_index[AA]
        # 获取该原子的坐标
        point = coord_all_Atom_tensor[atom_index]

        # 计算当前原子和所有其它原子的距离
        expanded_point = point.unsqueeze(0).expand(coord_all_Atom_tensor.shape[0], -1)
        differences = coord_all_Atom_tensor - expanded_point
        distances_square = torch.sum(differences ** 2, dim=1)
        dist = (torch.sqrt(distances_square)).unsqueeze(0)
        # 初始化邻接矩阵和边特征矩阵
        nodes = dist.shape[1]
        adj = torch.zeros((1, nodes))#邻接矩阵
        E = torch.zeros(( nodes, 2*dseq-1 +1+1+1+1  ))#边特征矩阵
        # 找k阶近邻
        _, indices = torch.topk(dist, k=k + 1, largest=False)
        knn = indices[0][1:]#去除自己，取k个近邻
        
        # 为每个原子创建边特征
        for j in range(nodes):
            not_edge = True#标记是否为边
            # 计算当前原子的序列距离
            dij_seq = abs(atom_index - j)
            # 根据序列距离和空间距离设置边特征
            if dij_seq < dseq and dist[0][j] < dr/2 :
                E[j][0 - 1 + atom_index - j + dseq] = 1#设置边特征
                not_edge = False
            if dist[0][j] < dr and dij_seq >= dlong:
                E[j][0 - 1 + 2 * dseq] = 1#设置边特征
                not_edge = False
            if j in knn and dij_seq >= dlong:
                E[j][0 + 2 * dseq] = 1#设置边特征
                not_edge = False
            #如果无边链接，跳过
            if not_edge:
                continue
            adj[0][j] = 1
            E[j][0 + 1 + 2 * dseq] = dij_seq #保留序列距离
            E[j][0 + 2 + 2 * dseq] = dist[0][j]  #保留空间距离

        #聚合边特征
        EDGE_feature_sum = torch.matmul(adj, E[:,0:7])
        EDGE_feature_mean= torch.matmul(adj, E[:,7:])/((adj == 1).sum().item() )
        #聚合节点特征
        aggregate_EDGE_feature = torch.cat([EDGE_feature_sum, EDGE_feature_mean], dim=1)
        aggregate_node_feature=torch.matmul(adj, (torch.from_numpy(ALL_atom_feature_combine)).to(torch.float32)    )
        #组合节点和边特征
        ATOM_combine_node_edge_feature = torch.cat([aggregate_node_feature, aggregate_EDGE_feature], dim=1)
        modify_site_pdb_ATOM_combine_node_edge_feature.append(ATOM_combine_node_edge_feature)
    # 将所有修饰位点的特征组合成一个张量
    modify_site_pdb_ATOM_combine_node_edge_feature=torch.cat(modify_site_pdb_ATOM_combine_node_edge_feature, dim=0)
    return modify_site_pdb_ATOM_combine_node_edge_feature
def Amino_acid_granularity(coord_all_AA_tensor, ALL_AA_feature_combine, modify_site_pdb,dseq=3,dr=10,dlong=5,k=10 ):
    #储存每个修饰位点特征的列表
    modify_site_pdb_AA_combine_node_edge_feature = []
    
    #遍历每个需要分析的位点
    for AA in  modify_site_pdb:
        AA_index=AA
        #获取当前氨基酸的坐标
        point = coord_all_AA_tensor[AA_index]

        #计算当前氨基酸与所有其它氨基酸的距离
        expanded_point = point.unsqueeze(0).expand(coord_all_AA_tensor.shape[0], -1)
        differences = coord_all_AA_tensor - expanded_point
        distances_square = torch.sum(differences ** 2, dim=1)
        dist = (torch.sqrt(distances_square)).unsqueeze(0)
        #初始化邻接矩阵和边特征矩阵
        nodes = dist.shape[1]
        adj = torch.zeros((1, nodes))
        E = torch.zeros(( nodes, 2*dseq-1 +1+1+1+1  ))
        #找出k阶近邻
        _, indices = torch.topk(dist, k=k + 1, largest=False)
        knn = indices[0][1:]


        #为每个氨基酸创建边特征
        for j in range(nodes):
            not_edge = True
            dij_seq = abs(AA_index - j)
            #根据序列距离和空间距离设置边特征
            if dij_seq < dseq and dist[0][j] < dr/2 :
                E[j][0 - 1 + AA_index - j + dseq] = 1
                not_edge = False
            if dist[0][j] < dr and dij_seq >= dlong:
                E[j][0 - 1 + 2 * dseq] = 1
                not_edge = False
            if j in knn and dij_seq >= dlong:
                E[j][0 + 2 * dseq] = 1
                not_edge = False
            #如果没有边链接，跳过
            if not_edge:
                continue
            # 设置邻接矩阵和额外的边特征
            adj[0][j] = 1
            E[j][0 + 1 + 2 * dseq] = dij_seq
            E[j][0 + 2 + 2 * dseq] = dist[0][j]
        # non_zero_indices = torch.nonzero(E)
        # if non_zero_indices.numel() > 0:
        #     print("边特征矩阵中存在非零值，位置如下：")
        #     print(non_zero_indices)
        # else:
        #     print("边特征矩阵中的所有值都是零。")  
        # print(f"Amino Acid Index: {AA_index}")
        # print(f"Adjacency Matrix Shape: {adj.shape}, Content: {adj}")
        # print(f"Edge Features Shape: {E.shape}, Content: {E}")
        #聚合边特征
        EDGE_feature_sum = torch.matmul(adj, E[:,0:7])
        EDGE_feature_mean= torch.matmul(adj, E[:,7:])/((adj == 1).sum().item() )
        aggregate_EDGE_feature = torch.cat([EDGE_feature_sum, EDGE_feature_mean], dim=1)
        aggregate_node_feature_1=torch.matmul(adj,torch.from_numpy(ALL_AA_feature_combine[:,0:5]  ).to(torch.float32))
        #聚合节点特征
        aggregate_node_feature_1=aggregate_node_feature_1/((adj == 1).sum().item() )
        aggregate_node_feature_2=torch.matmul(adj,torch.from_numpy(ALL_AA_feature_combine [:,5:]  ).to(torch.float32))
        aggregate_node_feature = torch.cat([aggregate_node_feature_1, aggregate_node_feature_2], dim=1)
        #组合节点和边特征
        combine_node_edge_feature = torch.cat([aggregate_node_feature, aggregate_EDGE_feature], dim=1)
        modify_site_pdb_AA_combine_node_edge_feature.append(combine_node_edge_feature)
    #将所有修饰位点的特征组合成一个张量
    modify_site_pdb_AA_combine_node_edge_feature=torch.cat(modify_site_pdb_AA_combine_node_edge_feature, dim=0)
    return modify_site_pdb_AA_combine_node_edge_feature
def process_dssp_and_pdb(dssp, pdb_document, modify_site_pdb, Atom_target):
    aa_type = "ACDEFGHIKLMNPQRSTVWYX"
    SS_type = "HBEGITS-"
    dssp_feature = []
    
    for i in range(len(dssp)):
        SS_vec = np.zeros(8)
        SS = dssp.property_list[i][2]
        SS_vec[SS_type.find(SS)] = 1
        PHI = dssp.property_list[i][4]
        PSI = dssp.property_list[i][5]
        ASA = dssp.property_list[i][3]
        aa_name_onehot = np.zeros(21)
        aa_name = dssp.property_list[i][1]
        
        if aa_name in aa_type:
            aa_name_onehot[aa_type.find(aa_name)] = 1
        else:
            aa_name_onehot[20] = 1
            
        feature1 = np.concatenate((np.array([PHI, PSI, ASA]), SS_vec))
        feature2 = aa_name_onehot
        feature = np.concatenate((feature1, feature2))
        dssp_feature.append(feature)
        
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:, 0:2]
    ASA_SS = dssp_feature[:, 2:]
    radian = angle * (np.pi / 180)
    ALL_AA_dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)
    
    ALL_atom_feature_combine = np.array([]).reshape(0, 26)
    coord_all_Atom = []
    coord_all_AA = []
    modify_site_pdb_NZ_index = {}
    with open(pdb_document, 'r') as f:
        filt_atom = 'CA'
        for line in f:
            if not line.startswith('ATOM'):
                continue
                
            atom, amino, amino_order, atom_order, x, y, z, atom_feature_combine = pdb_split(line)
            ALL_atom_feature_combine = np.vstack([ALL_atom_feature_combine, atom_feature_combine])
            
            if atom == filt_atom:
                coord_all_AA.append([x, y, z])
                
            coord_all_Atom.append([x, y, z])
            
            if atom == Atom_target and amino_order in modify_site_pdb:
                modify_site_pdb_NZ_index[amino_order] = atom_order
                
    print(f"Processed modification sites: {modify_site_pdb_NZ_index}")
    coord_all_AA_tensor = torch.FloatTensor(coord_all_AA)
    coord_all_Atom_tensor = torch.FloatTensor(coord_all_Atom)
    
    # 根据修饰位点的存在情况处理特征
    if len(modify_site_pdb_NZ_index) == len(modify_site_pdb):
        modify_site_pdb_aggregate_atom_node_and_edge_feature = Atom_granularity(
            coord_all_Atom_tensor, ALL_atom_feature_combine,
            modify_site_pdb, modify_site_pdb_NZ_index)
        modify_site_pdb_aggregate_AA_node_and_edge_feature = Amino_acid_granularity(
            coord_all_AA_tensor, ALL_AA_dssp_feature, modify_site_pdb)
        total_complete_structure = torch.from_numpy(np.mean(ALL_AA_dssp_feature, axis=0, keepdims=True))
        total_complete_structure_used = total_complete_structure.repeat(
            modify_site_pdb_aggregate_AA_node_and_edge_feature.shape[0], 1)
        final_feature = torch.cat((
            total_complete_structure_used,
            modify_site_pdb_aggregate_AA_node_and_edge_feature,
            modify_site_pdb_aggregate_atom_node_and_edge_feature
        ), dim=1)
        used_modify_site_pdb = modify_site_pdb
    else:
        final_feature = torch.zeros((len(modify_site_pdb), 112))
        used_modify_site_pdb = modify_site_pdb
        
    return final_feature, used_modify_site_pdb
def read_txt(address):
    """读取文本文件，提取蛋白质ID、修饰位点、标签和序列"""
    sequences = []
    labels = []
    modify_site = []
    uniprot_ID = []
    
    with open(address, 'r') as file:
        for line in file:
            if line.startswith('>'):
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    uniprot_ID.append(parts[0][1:])
                    labels.append(int(parts[1]))
                    modify_site.append(parts[3])
            else:
                sequences.append(line.strip())
                
    return uniprot_ID, modify_site, labels, sequences

def get_structure_feature(train_address, test_address, structure_data_address, Atom_target):
    """从训练和测试数据中提取结构特征"""
    # 读取训练和测试数据的蛋白质ID和修饰位点
    uniprot_ID_train, modify_site_train, _, _ = read_txt(train_address)
    uniprot_ID_test, modify_site_test, _, _ = read_txt(test_address)
    
    # 初始化蛋白质字典
    protein_dict = {}
    
    # 处理训练集数据
    for protein_id, site in zip(uniprot_ID_train, modify_site_train):
        normalized_id = normalize_protein_id(protein_id)
        if normalized_id in protein_dict:
            protein_dict[normalized_id].append(site)
        else:
            protein_dict[normalized_id] = [site]
    
    # 处理测试集数据
    for protein_id, site in zip(uniprot_ID_test, modify_site_test):
        normalized_id = normalize_protein_id(protein_id)
        if normalized_id in protein_dict:
            protein_dict[normalized_id].append(site)
        else:
            protein_dict[normalized_id] = [site]
    
    # 去除重复的修饰位点
    for protein_id in protein_dict:
        protein_dict[protein_id] = list(set(protein_dict[protein_id]))
    
    # 转换修饰位点为整数
    protein_dict_site = {key: list(map(int, value)) for key, value in protein_dict.items()}
    
    train_and_test_structure_feature = []
    
    # 处理每个蛋白质的结构特征
    for key in tqdm(protein_dict_site.keys(), desc="Processing proteins"):
        possible_files = glob.glob(os.path.join(structure_data_address, normalize_protein_id(key) + '*.pdb'))
        if not possible_files:
            print(f"Warning: PDB file not found for {normalize_protein_id(key)}")
            continue
            
        file_path = possible_files[0]
        modify_site_pdb = [site - 1 for site in protein_dict_site[key]]
        
        try:
            p = PDBParser()
            structure = p.get_structure("1", file_path)
            model = structure[0]
            dssp = DSSP(model, file_path, dssp='/home/hrren/PTM-CMGMS-main/Codes/Multi-granularity-Structure/mkdssp/mkdssp')
            
            feature, modify_site_pdb_shunxu = process_dssp_and_pdb(dssp, file_path, modify_site_pdb, Atom_target)
            
            if feature.shape[0] > 0:
                normalized_key = normalize_protein_id(key)
                arrayB = np.array([normalized_key] * feature.shape[0]).reshape(-1, 1)
                arrayC = np.array(modify_site_pdb_shunxu).reshape(-1, 1)
                result = np.concatenate((arrayB, arrayC), axis=1)
                feature_ndarray = feature.numpy()
                df = pd.concat([pd.DataFrame(result), pd.DataFrame(feature_ndarray)], axis=1)
                train_and_test_structure_feature.append(df)
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            continue
    
    if train_and_test_structure_feature:
        train_and_test_structure_feature = pd.concat(train_and_test_structure_feature, ignore_index=True)
        
        # 保存结构特征
        feature_save_path = os.path.join(f'./{PTM_name}_result', 'structure_features.csv')
        os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
        train_and_test_structure_feature.to_csv(feature_save_path, index=False)
        print(f"Structure features saved to {feature_save_path}")
        
        return train_and_test_structure_feature
    else:
        print("Warning: No structure features were generated.")
        return pd.DataFrame()

def read_txt_with_features(address, train_and_test_structure_feature_input, output_file):
    """读取文本文件并结合结构特征"""
    sequences = []
    labels = []
    modify_site = []
    uniprot_ID = []
    feature = []
    unmatched_proteins = []

    # 读取序列文件
    with open(address, 'r') as file:
        for line in file:
            if line.startswith('>'):
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    uniprot_ID.append(parts[0][1:].split('_')[0])
                    labels.append(int(parts[1]))
                    modify_site.append(int(parts[3]))
            else:
                sequences.append(line.strip())
    
    # 处理每个蛋白质
    for i in tqdm(range(len(uniprot_ID)), desc="Processing sequences"):
        protein_id = uniprot_ID[i]
        
        # 查找匹配的特征
        matching_rows = train_and_test_structure_feature_input[
            (train_and_test_structure_feature_input.iloc[:, 0] == protein_id) & 
            (train_and_test_structure_feature_input.iloc[:, 1].astype(int) == modify_site[i] - 1)
        ]
        
        if not matching_rows.empty:
            feature.append(matching_rows.iloc[0, 2:].values.astype(float))
        else:
            print(f"No match found for protein {protein_id}")
            unmatched_proteins.append(protein_id)
            feature.append(np.zeros(train_and_test_structure_feature_input.shape[1] - 2, dtype=float))

    # 保存未匹配的蛋白质
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for protein in unmatched_proteins:
            f.write(f"{protein}\n")
    
    print(f"Unmatched proteins have been saved to {output_file}")
    
    return pd.DataFrame({
        'sequence': sequences,
        'label': labels,
        'uniprot_ID': uniprot_ID,
        'modify_site': modify_site,
        'feature': feature
    })
train_and_test_structure_feature = get_structure_feature(
    train_data_address,
    test_data_address,
    structure_data,
    Atom_target
)
class My_model(nn.Module):
    def __init__(self, input_dim=112):
        super().__init__()
        self.feature_reduction = nn.Linear(input_dim, 34)
        
        self.structure_feature_learn_complete_block = nn.Sequential(
            nn.Linear(34, 34//2),
            nn.LeakyReLU(),
            nn.Linear(34//2, 34))
            
        self.structure_feature_learn_AA_block = nn.Sequential(
            nn.Linear(34, 34//2),
            nn.LeakyReLU(),
            nn.Linear(34//2, 34))
            
        self.structure_feature_learn_atom_block = nn.Sequential(
            nn.Linear(34, 34//2),
            nn.LeakyReLU(),
            nn.Linear(34//2, 34))
        
        self.feature_learned = nn.Sequential(
            nn.Linear(102, 51),
            nn.LeakyReLU(),
            nn.Linear(51, 28))
        
        self.feature_final = nn.Sequential(
            nn.Linear(28, 2))
            
        self.combine_block_MLP = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid())

    def forward(self, structure_feature):
        if structure_feature.dim() == 3:
            structure_feature = structure_feature.view(-1, structure_feature.size(-1))
        
        x = self.feature_reduction(structure_feature)
        
        x_Complete = self.structure_feature_learn_complete_block(x)
        x_AA = self.structure_feature_learn_AA_block(x)
        x_ATOM = self.structure_feature_learn_atom_block(x)
        
        x1 = torch.cat((x_Complete, x_AA, x_ATOM), dim=1)
        
        x = self.feature_learned(x1)
        return x

    def trainModel(self, structure_feature):
        with torch.no_grad():
            output = self.forward(structure_feature)
        feature = self.feature_final(output)
        logit = self.combine_block_MLP(feature)
        return logit, feature, output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin_input):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin_input

    def forward(self, output1, output2, label):
        device = output1.device
        output2 = output2.to(device)
        label = label.to(device)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class MyDataset_test(Dataset):
    def __init__(self, data):
        self.feature = data
        self.length = len(data)
        
    def __getitem__(self, idx):
        # 获取特征数据
        feature = self.feature.iloc[idx]['feature']
        
        # 确保特征是numpy数组并且类型正确
        if isinstance(feature, list):
            feature = np.array(feature, dtype=np.float32)
        elif isinstance(feature, np.ndarray):
            if feature.dtype == np.object_:
                feature = feature.astype(np.float32)
            elif feature.dtype != np.float32:
                feature = feature.astype(np.float32)
                
        # 获取标签并确保是整数
        label = int(self.feature.iloc[idx]['label'])
        
        return (torch.tensor(feature, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return self.length

def prepare_datasets(train_structure_feature, test_structure_feature, batch_size):
    """
    准备数据加载器，包含数据类型检查和错误处理
    """
    print("检查训练数据特征类型...")
    train_features = train_structure_feature['feature'].values
    if any(isinstance(x, (list, np.ndarray)) and (isinstance(x, np.ndarray) and x.dtype == np.object_) for x in train_features):
        print("转换训练数据特征类型...")
        train_structure_feature['feature'] = train_structure_feature['feature'].apply(
            lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else 
            x.astype(np.float32) if isinstance(x, np.ndarray) else x
        )
    
    print("检查测试数据特征类型...")
    test_features = test_structure_feature['feature'].values
    if any(isinstance(x, (list, np.ndarray)) and (isinstance(x, np.ndarray) and x.dtype == np.object_) for x in test_features):
        print("转换测试数据特征类型...")
        test_structure_feature['feature'] = test_structure_feature['feature'].apply(
            lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else 
            x.astype(np.float32) if isinstance(x, np.ndarray) else x
        )
    
    print("创建数据加载器...")
    # 创建训练数据的打乱版本
    train_structure_feature_shuffle = train_structure_feature.sample(frac=1, random_state=1)
    
    # 创建数据加载器
    train_Comparative_learning_data = DataLoader(
        MyDataset_test(train_structure_feature_shuffle),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate
    )
    
    data_loaders = {
        "Train_data": DataLoader(
            MyDataset_test(train_structure_feature),
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False
        ),
        "Test_data": DataLoader(
            MyDataset_test(test_structure_feature),
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False
        )
    }
    
    return train_Comparative_learning_data, data_loaders



def save_model(model, save_dir, epoch, optimizer=None):
    """保存模型和训练状态"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_architecture': type(model).__name__
    }
    
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    
    # 验证所有必要的权重是否存在
    expected_keys = [
        'feature_reduction.weight', 'feature_reduction.bias',
        'structure_feature_learn_complete_block.0.weight', 'structure_feature_learn_complete_block.0.bias',
        'structure_feature_learn_complete_block.2.weight', 'structure_feature_learn_complete_block.2.bias',
        'structure_feature_learn_AA_block.0.weight', 'structure_feature_learn_AA_block.0.bias',
        'structure_feature_learn_AA_block.2.weight', 'structure_feature_learn_AA_block.2.bias',
        'structure_feature_learn_atom_block.0.weight', 'structure_feature_learn_atom_block.0.bias',
        'structure_feature_learn_atom_block.2.weight', 'structure_feature_learn_atom_block.2.bias',
        'feature_learned.0.weight', 'feature_learned.0.bias',
        'feature_learned.2.weight', 'feature_learned.2.bias',
        'feature_final.0.weight', 'feature_final.0.bias',
        'combine_block_MLP.0.weight', 'combine_block_MLP.0.bias'
    ]
    
    model_state = state['model_state_dict']
    missing_keys = [key for key in expected_keys if key not in model_state]
    
    if missing_keys:
        raise ValueError(f"Missing weights in model state: {missing_keys}")
    
    # 保存检查点
    model_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(state, model_path)
    print(f"Model saved to {model_path}")
    
    # 保存最新模型
    latest_path = os.path.join(save_dir, 'latest_model.pth')
    torch.save(state, latest_path)
    print(f"Latest model saved to {latest_path}")

def load_model(model, model_path, optimizer=None):
    """加载模型和训练状态"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state = torch.load(model_path)
    
    if state['model_architecture'] != type(model).__name__:
        raise ValueError(f"Model architecture mismatch: expected {type(model).__name__}, "
                       f"got {state['model_architecture']}")
    
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    return state['epoch']

def collate(batch):
    batch_a_structure_feature = []
    batch_b_structure_feature = []
    label_a_list = []
    label_b_list = []
    label_Comparative_learning = []
    batch_size = len(batch)

    for i in range(int(batch_size / 2)):
        structure_a, label_a = (batch[i][0]).unsqueeze(0), (batch[i][1]).unsqueeze(0)
        structure_b, label_b = (batch[i + int(batch_size / 2)][0]).unsqueeze(0), (batch[i + int(batch_size / 2)][1]).unsqueeze(0)
        
        label_a_list.append(label_a)
        label_b_list.append(label_b)
        batch_a_structure_feature.append(structure_a)
        batch_b_structure_feature.append(structure_b)
        
        label = (label_a ^ label_b)
        label_Comparative_learning.append(label)

    structure_1 = torch.cat(batch_a_structure_feature)
    structure_2 = torch.cat(batch_b_structure_feature)
    label = torch.cat(label_Comparative_learning)
    label1 = torch.cat(label_a_list)
    label2 = torch.cat(label_b_list)

    return structure_1, structure_2, label, label1, label2
# 初始化参数
print('Please wait, this process will take some time...')
print("Loading and processing data...")

# 创建保存目录
path = Path(f'./{PTM_name}_result/')
if not os.path.exists(path):
    os.makedirs(path)

# 设置随机种子
setup_seed(3407)

# 训练参数
batch_size = 256
lr = 0.0001
n_epoch = 300
w1 = 1/9
dim_feature = 112//4
margin_1 = 10

# 准备数据
train_structure_feature = read_txt_with_features(
    train_data_address, 
    train_and_test_structure_feature, 
    f'./{PTM_name}_result/unmatched_sites_train.txt'
)

test_structure_feature = read_txt_with_features(
    test_data_address, 
    train_and_test_structure_feature,
    f'./{PTM_name}_result/unmatched_sites_test.txt'
)

train_structure_feature_shuffle = train_structure_feature.sample(frac=1, random_state=1)

# 创建数据加载器
train_Comparative_learning_data = DataLoader(
    MyDataset_test(train_structure_feature_shuffle),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate
)

data_loaders = {
    "Train_data": DataLoader(
        MyDataset_test(train_structure_feature), 
        batch_size=batch_size,
        pin_memory=True, 
        shuffle=False
    ),
    "Test_data": DataLoader(
        MyDataset_test(test_structure_feature), 
        batch_size=batch_size,
        pin_memory=True, 
        shuffle=False
    )
}

# 初始化模型、损失函数和优化器
model = My_model().to(device)
criterion_Comparative = ContrastiveLoss(margin_1).to(device)
loss_function = nn.BCELoss(reduction='sum').to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 创建模型保存目录
save_dir = f'./{PTM_name}_result/model_checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 训练循环
best_loss = float('inf')
print("Starting training...")

for epoch in range(1, n_epoch + 1):
    model.train()
    epoch_loss = 0
    
    tbar = tqdm(enumerate(train_Comparative_learning_data), 
                total=len(train_Comparative_learning_data),
                desc=f"Epoch {epoch}/{n_epoch}")
    
    for idx, (structure_1, structure_2, label, label1, label2) in tbar:
        # 将数据移动到设备
        structure_1 = structure_1.to(device)
        structure_2 = structure_2.to(device)
        label = label.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        # 前向传播
        output1 = model(structure_1)
        output2 = model(structure_2)
        output3, _, _ = model.trainModel(structure_1)
        output4, _, _ = model.trainModel(structure_2)

        # 计算损失
        loss1 = criterion_Comparative(output1, output2, label)
        loss2 = loss_function(output3.squeeze(), label1.to(torch.float32))
        loss3 = loss_function(output4.squeeze(), label2.to(torch.float32))
        
        loss = w1 * loss1 + (1 - w1) * (loss2 + loss3)
        epoch_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        tbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 计算平均epoch损失
    avg_epoch_loss = epoch_loss / len(train_Comparative_learning_data)
    print(f"Epoch [{epoch}/{n_epoch}], Average Loss: {avg_epoch_loss:.4f}")
    
    # 每10个epoch保存一次检查点
    if epoch % 10 == 0:
        save_model(model, save_dir, epoch, optimizer)
    
    # 如果是最佳模型则保存
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        save_model(model, f'{save_dir}/best_model', epoch, optimizer)
        print(f"Saved new best model with loss: {best_loss:.4f}")

# 保存最终模型
print("Training completed. Saving final model...")
final_save_dir = f'./{PTM_name}_result/final_model'
save_model(model, final_save_dir, n_epoch, optimizer)

# 特征提取
print("Extracting features...")
def test(net: torch.nn.Module, test_loader, device, dim_feature):
    net.eval()
    return_feature_result = torch.empty((0, dim_feature), device=device)
    
    with torch.no_grad():
        for idx, (structure_feature, _) in tqdm(enumerate(test_loader), 
                                              total=len(test_loader),
                                              desc="Processing batches"):
            structure_feature = structure_feature.to(device)
            _, _, feature = net.trainModel(structure_feature)
            return_feature_result = torch.cat((return_feature_result, feature), dim=0)
            
    return return_feature_result

# 为训练集和测试集提取特征
print("Extracting features for training and test sets...")
for dataset_name in ['Train_data', 'Test_data']:
    print(f"Processing {dataset_name}...")
    features = test(model, data_loaders[dataset_name], device, dim_feature)
    features = features.cpu().numpy()
    
    # 保存特征
    save_path = f'./{PTM_name}_result/features/Comparative_learning_feature_dim{dim_feature}_{dataset_name.lower()}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {save_path}")

print('All processing completed!')