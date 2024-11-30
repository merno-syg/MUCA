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

device_set="cuda:0"
device = torch.device(device_set)
structure_data='/mnt/data/hrren/CM/PDB'
device_set="cuda:0"
Atom_target="NZ"#Atom
train_data_address='/mnt/data/hrren/CM/Datasets/nr40/win51_copy/CM_train_ratio_1.fa'
test_data_address='/mnt/data/hrren/CM/Datasets/nr40/win51_copy/CM_text_ratio_1.fa'
PTM_name=train_data_address.split('/')[-2]

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
    #从PDB行中提取信息
    Atom_order=int(line[6:11].strip())-1
    atom=line[11:16].strip()
    amino=line[16:21].strip()
    AA_order=int(line[22:28].strip())-1
    x=line[28:38].strip()
    y=line[38:46].strip()
    z=line[46:54].strip()
    atom_single_name=line.strip()[-1]
    #为原子和氨基酸类型创建one-hot编码
    atom_single_name_vec = np.zeros(len(atom_type))
    atom_single_name_vec[atom_type.find(atom_single_name)] = 1
    AA_single_name_vec = np.zeros(len(aa_type))
    AA_single_name_vec[   aa_type.find(   aa_trans_DICT[ amino ] )  ] = 1
    #concat 原子和氨基酸特征向量
    atom_feature_combine= np.concatenate(( atom_single_name_vec.reshape(1, -1)   , AA_single_name_vec.reshape(1, -1)),axis=1)

    return atom,amino,AA_order, Atom_order, float(x),float(y),float(z)  ,atom_feature_combine


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

def process_dssp_and_pdb(dssp,pdb_document,modify_site_pdb,Atom_target):
    # 20+1种氨基酸类型
    aa_type = "ACDEFGHIKLMNPQRSTVWYX"
    #二级结构类型（H: α-螺旋, E: β-折叠, G: 3-10 螺旋等）
    SS_type = "HBEGITS-"
    dssp_feature = []
    for i in range(len(dssp)):
        SS_vec = np.zeros(8) #初始化二级结构向量，长度为8
        SS=dssp.property_list[i][2] #获取当前的氨基酸的二级结构
        SS_vec[SS_type.find(SS)] = 1 #将对应的二级结构位置置为1
        PHI = dssp.property_list[i][4] # 获取 φ 角
        PSI = dssp.property_list[i][5] # 获取 ψ 角
        ASA = dssp.property_list[i][3] # 获取 ASA (表面积)
        aa_name_onehot=np.zeros(21) #初始化氨基酸名称的one-hot编码
        aa_name=dssp.property_list[i][1] #获取当前氨基酸的名称

        # 如果氨基酸名称在标准氨基酸类型种，one-hot编码种对应位置为1
        if aa_name in aa_type:
            aa_name_onehot[aa_type.find(aa_name)] = 1
        else:
            aa_name_onehot[20] = 1 #否则标记为未知氨基酸
        #组合特征，前面是φ, ψ, ASA 和二级结构向量，后面是氨基酸名称的独热编码
        feature1= np.concatenate(   (np.array([PHI, PSI, ASA]), SS_vec))
        feature2=aa_name_onehot
        feature=np.concatenate(   (feature1,feature2 ))
        #将当前氨基酸的特征添加到列表中
        dssp_feature.append(feature)
    #转化为numpy数组
    dssp_feature = np.array(dssp_feature)
    # 提取 φ、ψ 角
    angle = dssp_feature[:, 0:2]
    # 提取 ASA 和 SS 特征
    ASA_SS = dssp_feature[:, 2:]
     # 将角度从度转为弧度
    radian = angle * (np.pi / 180)
    # 计算所有氨基酸的 DSSP 特征组合，求取正弦和余弦
    ALL_AA_dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)
    # 初始化原子特征组合的空数组
    ALL_atom_feature_combine = np.array([]).reshape(0, 5 + 21)  # 5是原子特征的维度，21是氨基酸独热编码
 
    coord_all_Atom = [] # 存放所有原子的坐标
    coord_all_AA = [] # 存放所有氨基酸的坐标
    modify_site_pdb_NZ_index = {} # 存储修饰位点的原子索引

    with open(pdb_document, 'r') as f:
        filt_atom = 'CA'  # 设置过滤的原子类型为 C-alpha
        for line in f:
            kind = line[:6].strip()   # 获取行类型
            if kind not in ['ATOM']:  # 仅处理 "ATOM" 类型的行
                continue

            # 从 PDB 行中解析出原子、氨基酸、顺序及坐标信息
            atom, amino, amino_order, atom_order, x, y, z  ,atom_feature_combine= pdb_split(line)
            # 将当前原子的特征组合到 ALL_atom_feature_combine 中

            ALL_atom_feature_combine=np.vstack([ALL_atom_feature_combine, atom_feature_combine])

            if atom == filt_atom: # 如果当前原子是 C-alpha
                coord_all_AA.append([x, y, z])  # 将其坐标添加到 coord_all_AA

            coord_all_Atom.append([x, y, z])   # 将原子坐标添加到 coord_all_Atom

            # 如果当前原子是目标原子且其氨基酸序列在修饰位点中

            if  atom==Atom_target  and  (amino_order in modify_site_pdb):
                modify_site_pdb_NZ_index[amino_order]=atom_order   # 记录其原子索引
        print(f"Processed modification sites: {modify_site_pdb_NZ_index}")
    # 将氨基酸和原子的坐标转换为 FloatTensor
        coord_all_AA_tensor=torch.FloatTensor(coord_all_AA)
        coord_all_Atom_tensor = torch.FloatTensor(coord_all_Atom)

    # 检查修饰位点的数量是否匹配，并调用特征提取函数
    if len(modify_site_pdb_NZ_index) == len(modify_site_pdb):
        # 所有修饰位点都存在，特征提取
        modify_site_pdb_aggregate_atom_node_and_edge_feature = Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine,modify_site_pdb, modify_site_pdb_NZ_index, dseq=3, dr=10, dlong=5, k=10 )
        modify_site_pdb_aggregate_AA_node_and_edge_feature = Amino_acid_granularity(coord_all_AA_tensor,ALL_AA_dssp_feature ,modify_site_pdb, dseq=3,dr=10, dlong=5, k=10)
        
        # 计算整体特征并进行重复以匹配氨基酸数量
        total_complete_structure= torch.from_numpy(  np.mean(ALL_AA_dssp_feature, axis=0, keepdims=True) )
        total_complete_structure_used=total_complete_structure.repeat(  modify_site_pdb_aggregate_AA_node_and_edge_feature.shape[0]   , 1)
        # 合并最终特征
        final_feature=torch.cat((total_complete_structure_used, modify_site_pdb_aggregate_AA_node_and_edge_feature, modify_site_pdb_aggregate_atom_node_and_edge_feature), dim=1)
        used_modify_site_pdb=modify_site_pdb

    elif len(modify_site_pdb_NZ_index) != len(modify_site_pdb) and len(modify_site_pdb_NZ_index) !=0:
        # 部分修饰位点存在
        modify_site_pdb_exist=list(modify_site_pdb_NZ_index.keys())
        modify_site_pdb_not_exist = [num for num in modify_site_pdb if num not in modify_site_pdb_exist]

        # 提取存在的修饰位点的特征
        modify_site_pdb_aggregate_atom_node_and_edge_feature = Atom_granularity(coord_all_Atom_tensor, ALL_atom_feature_combine,modify_site_pdb_exist, modify_site_pdb_NZ_index, dseq=3, dr=10, dlong=5, k=10 )
        modify_site_pdb_aggregate_AA_node_and_edge_feature = Amino_acid_granularity(coord_all_AA_tensor,ALL_AA_dssp_feature ,modify_site_pdb_exist, dseq=3,dr=10, dlong=5, k=10)
        
        # 计算整体特征
        total_complete_structure= torch.from_numpy(  np.mean(ALL_AA_dssp_feature, axis=0, keepdims=True) )
        total_complete_structure_used=total_complete_structure.repeat(  modify_site_pdb_aggregate_AA_node_and_edge_feature.shape[0]   , 1)
        final_feature_modify_site_pdb_exist=torch.cat((total_complete_structure_used, modify_site_pdb_aggregate_AA_node_and_edge_feature, modify_site_pdb_aggregate_atom_node_and_edge_feature), dim=1)

        # 为不存在的修饰位点初始化全零特征
        final_modify_site_pdb_not_exist = torch.zeros((len(modify_site_pdb_not_exist), 112))
        
        # 合并存在与不存在的修饰位点特征
        final_feature=torch.cat((final_feature_modify_site_pdb_exist, final_modify_site_pdb_not_exist), dim=0)
        used_modify_site_pdb=modify_site_pdb_exist+modify_site_pdb_not_exist
    
    elif len(modify_site_pdb_NZ_index) == 0:
        # 如果没有修饰位点，返回全零特征
        final_feature = torch.zeros((len(modify_site_pdb), 112))
        used_modify_site_pdb = modify_site_pdb  # 存储所有修饰位点
    return final_feature,used_modify_site_pdb   # 返回最终特征和修饰位点

def read_txt(address):
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
    # 从训练和测试数据文件中读取蛋白质 ID 和修饰位点
    uniprot_ID_train, modify_site_train, _, _ = read_txt(train_address)
    uniprot_ID_test, modify_site_test, _, _ = read_txt(test_address)
    # 初始化一个字典用于存储蛋白质 ID 及其对应的修饰位点
    protein_dict = {}
    # 处理训练集的蛋白质 ID 和修饰位点
    for protein_id, site in zip(uniprot_ID_train, modify_site_train):
        normalized_id = normalize_protein_id(protein_id) # 标准化蛋白质 ID
        if normalized_id in protein_dict:
            protein_dict[normalized_id].append(site) # 如果 ID 存在，则添加修饰位点
        else:
            protein_dict[normalized_id] = [site]     # 否则，创建新条目
    
    # 处理测试集的蛋白质 ID 和修饰位点
    for protein_id, site in zip(uniprot_ID_test, modify_site_test):
        normalized_id = normalize_protein_id(protein_id) 
        if protein_id in protein_dict:
            protein_dict[normalized_id].append(site)  # 合并修饰位点
        else:
            protein_dict[normalized_id] = [site]      # 创建新条目
    # 删除重复的修饰位点
    for protein_id, site in protein_dict.items():
        protein_dict[protein_id] = list(set(site))
    # 将蛋白质字典中的修饰位点转换为整数列表
    protein_dict_site = {key: list(map(int, value)) for key, value in protein_dict.items()}

    train_and_test_structure_feature = []  # 存储训练和测试的结构特征
    
    # 遍历每个蛋白质 ID
    for key in tqdm(protein_dict_site.keys(), desc="Processing proteins"):
        possible_files = glob.glob(os.path.join(structure_data_address, normalize_protein_id(key) + '*.pdb')) 
        if not possible_files:
            print(f"Warning: PDB file not found for {normalize_protein_id(key)}")
            continue

        file_path = possible_files[0]  # 获取找到的第一个文件路径
        modify_site_pdb = [site - 1 for site in protein_dict_site[key]] # 将修饰位点转为基于 0 的索引

        pdb_document = file_path # 设置 PDB 文档路径
        p = PDBParser()
        structure = p.get_structure("1", pdb_document)# 解析结构
        model = structure[0]
        dssp = DSSP(model, pdb_document, dssp='/home/hrren/PTM-CMGMS-main/Codes/Multi-granularity-Structure/mkdssp/mkdssp')
        feature, modify_site_pdb_shunxu = process_dssp_and_pdb(dssp, pdb_document, modify_site_pdb, Atom_target)  # 提取特征
        # 如果特征不为空，将其与蛋白质 ID 和修饰位点组合
        if feature.shape[0] > 0:
            normalized_key = normalize_protein_id(key)  # 标准化蛋白质 ID
            arrayB = np.array(normalized_key).repeat(feature.shape[0], axis=0).reshape(feature.shape[0], 1) # 创建 ID 列
            arrayC = np.array(modify_site_pdb_shunxu).reshape(-1, 1)    # 创建修饰位点列
            result = np.concatenate((arrayB, arrayC), axis=1)   # 合并 ID 和修饰位点
            feature_ndarray = feature.numpy()   # 转为 NumPy 数组
            df = pd.concat([pd.DataFrame(result), pd.DataFrame(feature_ndarray)], axis=1)   # 创建 DataFrame
            train_and_test_structure_feature.append(df) # 将结果添加到列表中

    #  如果生成了结构特征，将它们组合成一个 DataFrame
    # if train_and_test_structure_feature:
    #     train_and_test_structure_feature = pd.concat(train_and_test_structure_feature, ignore_index=True)
    #     return train_and_test_structure_feature
    # else:
    #     print("Warning: No structure features were generated. Check your input data and file paths.")
    #     return pd.DataFrame()
    if train_and_test_structure_feature:
            train_and_test_structure_feature = pd.concat(train_and_test_structure_feature, ignore_index=True)
            
            # 保存结构特征为 CSV 文件
            feature_save_path = './structure_features.csv'
            train_and_test_structure_feature.to_csv(feature_save_path, index=False)
            print(f"Structure features saved to {feature_save_path}")
            
            return train_and_test_structure_feature
    else:
            print("Warning: No structure features were generated. Check your input data and file paths.")
            return pd.DataFrame()

def read_txt_with_features(address, train_and_test_structure_feature_input, output_file):
    sequences = []
    labels = []
    modify_site = []
    uniprot_ID = []
    feature = []
    unmatched_proteins = []

    with open(address, 'r') as file:
        for line in file:
            if line.startswith('>'):
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    uniprot_ID.append(parts[0][1:].split('_')[0])  # 只取蛋白质ID部分
                    labels.append(int(parts[1]))
                    modify_site.append(int(parts[3]))
            else:
                sequences.append(line.strip())
    
    for i in tqdm(range(len(uniprot_ID)), desc="Processing sequences"):
        protein_id = uniprot_ID[i]
        
        matching_rows = train_and_test_structure_feature_input[
            (train_and_test_structure_feature_input.iloc[:, 0] == uniprot_ID[i]) & 
            (train_and_test_structure_feature_input.iloc[:, 1].astype(int) == modify_site[i] + 1)
        ]
        
        if not matching_rows.empty:
            # 如果找到匹配的蛋白质，使用第一个匹配的行作为特征
            feature.append(matching_rows.iloc[0, 2:].values.astype(float))
        else:
            print(f"No match found for protein {protein_id}")
            unmatched_proteins.append(protein_id)
            # 使用零向量或平均特征作为默认值
            feature.append(np.zeros(train_and_test_structure_feature_input.shape[1] - 2, dtype=float))

    # 保存未匹配的蛋白质
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
print('Please wait, this process will take some time...')
print("Loading and processing data...")

train_and_test_structure_feature = get_structure_feature(train_data_address, test_data_address, structure_data, Atom_target)

train_unmatched_file = f'./{PTM_name}_result/unmatched_sites_train.txt'
test_unmatched_file = f'./{PTM_name}_result/unmatched_sites_text.txt'

train_structure_feature=read_txt_with_features(train_data_address, train_and_test_structure_feature, train_unmatched_file)
test_structure_feature=read_txt_with_features(test_data_address, train_and_test_structure_feature,test_unmatched_file)
train_structure_feature_shuffle=train_structure_feature.sample(frac=1, random_state=1)#随机打乱训练特征数据


def collate(batch):
    batch_a_structure_feature = []
    batch_b_structure_feature = []
    label_a_list = []
    label_b_list = []
    label_Comparative_learning = []
    batch_size = len(batch)

    for i in range(int(batch_size / 2)):

        structure_a, label_a=  (batch[i][0]).unsqueeze(0) ,  (batch[i][1]).unsqueeze(0)
        structure_b, label_b = (batch[i + int(batch_size / 2)][0]).unsqueeze(0) , (batch[i + int(batch_size / 2)][1]).unsqueeze(0) #转换成一个批次[C, H, W] [1, C, H, W]
        label_a_list.append(label_a)
        label_b_list.append(label_b)
        batch_a_structure_feature.append(structure_a)
        batch_b_structure_feature.append(structure_b)
        # label = (label_a != label_b).float()
        label = (label_a ^ label_b)
        label_Comparative_learning.append(label)

    structure_1=torch.cat( batch_a_structure_feature)
    structure_2=torch.cat(batch_b_structure_feature)
    label = torch.cat(label_Comparative_learning)
    label1 = torch.cat(label_a_list)
    label2 = torch.cat(label_b_list)

    return structure_1,structure_2, label, label1, label2  #前一半样本的结构特征，后一半样本的结构特征，比较学习得到的标签，各自来自前后半部分样本的标签


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
        with torch.no_grad():   #训练阶段关闭了梯度计算？
            output = self.forward(structure_feature)
        feature = self.feature_final(output)
        logit = self.combine_block_MLP(feature)
        return logit, feature, output

def test(net: torch.nn.Module, test_loader, loss_function, device, dim_feature):
    net.eval()
    return_feature_result = torch.empty((0, dim_feature), device=device)  # 创建一个空的张量用于存储返回的特征结果，初始化为形状 (0, dim_feature)
    with torch.no_grad():   # 禁止梯度计算，进行推理而不需要计算梯度
        for idx, (structure_feature, label) in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
            structure_feature, y = structure_feature.to(device), label.to(device)

            if structure_feature.dim() == 3:
                batch_size, seq_len, feature_dim = structure_feature.size() # 将三维输入展平为二维，便于模型处理
                structure_feature = structure_feature.view(batch_size * seq_len, feature_dim)
            
            # 通过模型进行推理，获取特征
            _, _, feature = net.trainModel(structure_feature)

            # 如果输入是三维的，则将提取的特征变换回三维
            if structure_feature.dim() == 3:
                feature = feature.view(batch_size, seq_len, -1)

            # 将提取的特征结果按行连接到返回结果中
            return_feature_result = torch.cat((return_feature_result, feature), dim=0)
    return return_feature_result

class MyDataset_test(Dataset):
    def __init__(self, data  ):
        self.feature=data
        self.length = data.shape[0]
    def __getitem__(self, idx):
        numpy_array = np.array(self.feature.iloc[idx, 4:116])  # 从数据集中获取特定范围的特征
        return (torch.tensor(self.feature.iloc[idx, 4:116], dtype=torch.float32),      # 将特征转换为张量
                torch.tensor(self.feature.iloc[idx, 1]))  # 获取标签并转换为张量
    def __len__(self):
        return self.length
    
# class MyDataset_test(Dataset):
#     def __init__(self, data):
#         self.data = data
#         self.length = data.shape[0]
    
#     def __getitem__(self, idx):
#         feature = np.array(self.data.iloc[idx]['feature'], dtype=float)
#         return (torch.tensor(feature, dtype=torch.float32),
#                 torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long))
    
#     def __len__(self):
#         return self.length  

class ContrastiveLoss(nn.Module):

    def __init__(self, margin_input):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin_input      # 存储边界值
    def forward(self, output1, output2, label):
        device = output1.device
        output2 = output2.to(device)
        label = label.to(device)
        euclidean_distance = F.pairwise_distance(output1, output2)   # 计算两个输出的欧几里得距离
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive  # 返回对比损失
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = My_model().to(device)
margin_1=10
criterion_Comparative = ContrastiveLoss(margin_1).to(device)
loss_function = nn.BCELoss(reduction='sum').to(device)



print('Please wait, this process will take about 1 minute...............................................')
path = Path(f'./{PTM_name}_result/')
# writer = SummaryWriter(path)
setup_seed(3407)
batch_size=256
lr=0.0001
n_epoch=300
w1=1/9
dim_feature=112//4
print("Checking data types and shapes:")
print(train_structure_feature['feature'].apply(lambda x: type(x)).value_counts())
print(train_structure_feature['feature'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else None).value_counts())

# 确保所有特征都是 numpy 数组
train_structure_feature['feature'] = train_structure_feature['feature'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
test_structure_feature['feature'] = test_structure_feature['feature'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
train_Comparative_learning_data = DataLoader(MyDataset_test(train_structure_feature_shuffle), 
                                             batch_size=batch_size, pin_memory=True, 
                                             shuffle=False, collate_fn=collate)

data_loaders = {}
data_loaders["Train_data"] = DataLoader(MyDataset_test(train_structure_feature), batch_size=batch_size,pin_memory=True, shuffle=False)
data_loaders["Test_data"] = DataLoader(MyDataset_test(test_structure_feature), batch_size=batch_size,pin_memory=True, shuffle=False)

model = My_model().to(device)

# 初始化对比损失函数，传入边界值
criterion_Comparative = ContrastiveLoss(margin_1)
loss_function = nn.BCELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
phase=0

for epoch in range(1, n_epoch + 1):
    tbar = tqdm(enumerate(train_Comparative_learning_data), disable=False, total=len(train_Comparative_learning_data))
    for idx, (structure_1, structure_2, label, label1, label2) in tbar:
        model.train()
        structure_1 = structure_1.to(device)
        structure_2 = structure_2.to(device)
        label = label.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        output1 = model(structure_1)
        output2 = model(structure_2)
        output3, _, _ = model.trainModel(structure_1)
        output4, _, _ = model.trainModel(structure_2)

        loss1 = criterion_Comparative(output1, output2, label)  # 计算对比损失
        loss2 = loss_function(output3.squeeze(), label1.to(torch.float32)) # 计算第一个特征的交叉熵损失
        loss3 = loss_function(output4.squeeze(), label2.to(torch.float32)) # 计算第二个特征的交叉熵损失
        
        # 合成总损失，结合对比损失与其他损失
        loss = w1 * loss1 + (1 - w1) * (loss2 + loss3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{n_epoch}], Loss: {loss.item():.4f}")



# 定义测试函数，用于提取特征
def test(net: torch.nn.Module, test_loader, loss_function, device, dim_feature):
    net.eval()
    return_feature_result = torch.empty((0, dim_feature), device=device)
    with torch.no_grad():
        for idx, (structure_feature, label) in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
            structure_feature, y = structure_feature.to(device), label.to(device)
            class_fenlei, representation, feature = net.trainModel(structure_feature)
            return_feature_result = torch.cat((return_feature_result, feature), dim=0)

    return return_feature_result   

  # 提取训练和测试数据的特征  
for _p in tqdm(['Test_data', 'Train_data'], desc="Extracting features"):
    data_loader = data_loaders[_p]
    Comparative_learning_feature = test(model, data_loader, loss_function, device, dim_feature) # 提取特征
    Comparative_learning_feature = Comparative_learning_feature.cpu().numpy()
    
    with open(f'./{path.name}/ture/Comparative_learning_feature_dim{dim_feature}_{_p.lower()}.pkl', 'wb') as f:
        pickle.dump(Comparative_learning_feature, f)

print("Feature extraction completed.")
for _p in ['Train_data', 'Test_data']:
    Comparative_learning_feature = test(model, data_loaders[_p], loss_function, device, dim_feature)
    Comparative_learning_feature = Comparative_learning_feature.cpu().numpy()
    file_suffix = 'train' if _p == 'Train_data' else 'test'
    with open(f'./{path.name}/ture/Comparative_learning_feature_dim{dim_feature}_{file_suffix}.pkl', 'wb') as f:
        pickle.dump(Comparative_learning_feature, f)

print('finished!!!!!!!!!!!!')