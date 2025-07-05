import os
import os.path as osp

import math
import pandas as pd
from zipfile import ZipFile

import torch
import datasets
from torch.utils.data import Dataset
from torch import Tensor

import torch_geometric
from torch_geometric.utils import degree

import copy
import random
import numpy as np

from typing import Any, Dict, List, Mapping

import pickle

def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))


class CustomDataCollator:
    def __init__(self, mgm_mask_ratio=0.5, align_mask_ratio=0.5, cross_hidden_size=3584):
        self.mgm_mask_ratio = mgm_mask_ratio
        self.align_mask_ratio = align_mask_ratio
        self.cross_hidden_size = cross_hidden_size

    def __call__(self, inputs) -> Dict[str, Any]:

        graphs = [i[0] for i in inputs]
        vgraphs = [i[1] for i in inputs]

        batch_size = len(graphs)
        batch = {}

        # Handling MGM data
        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max((g['input_edges'].shape[-1]) for g in graphs)

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["node_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        # SSL Labels
        batch["input_indegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_outdegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        mgm_mask_ratio = self.mgm_mask_ratio

        for idx, g in enumerate(graphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            masked_node_num = int(node_num * mgm_mask_ratio)

            edge_index = copy.deepcopy(g['input_edges'])

            indegree = degree(edge_index[0], num_nodes=node_num)
            outdegree = degree(edge_index[1], num_nodes=node_num)

            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_nodes"][idx, :node_num] = input_nodes + 1
            batch["node_mask"][idx][torch.randperm(node_num)[:masked_node_num]] = 1
            batch["padding_mask"][idx, :node_num] = 1

            batch["input_indegree"][idx, :node_num] = indegree
            batch["input_outdegree"][idx, :node_num] = outdegree
            batch["labels"][idx, :node_num] = input_nodes

        # Handling VGAlign Data
        max_vg_node_num = max(g['input_nodes'].shape[0] for g in vgraphs)
        max_vg_edge_num = max((g['input_edges'].shape[-1]) for g in vgraphs)   

        batch["input_vg_nodes"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)
        batch["input_vg_edges"] = torch.zeros(batch_size, 2, max_vg_edge_num, dtype=torch.long)
        batch["input_vg_verilogs"] = torch.zeros(batch_size, self.cross_hidden_size, dtype=torch.float)
        batch["vg_node_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)
        batch["vg_padding_mask"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)
        batch["vg_labels"] = torch.zeros(batch_size, max_vg_node_num, dtype=torch.long)

        align_mask_ratio = self.align_mask_ratio

        for idx, g in enumerate(vgraphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            masked_node_num = int(node_num * align_mask_ratio)

            edge_index = copy.deepcopy(g['input_edges'])
            edge_num = edge_index.shape[1]
            batch["input_vg_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_vg_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_vg_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_vg_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_vg_nodes"][idx, :node_num] = input_nodes + 1
            batch["vg_node_mask"][idx][torch.randperm(node_num)[:masked_node_num]] = 1
            batch["vg_padding_mask"][idx, :node_num] = 1

            batch["input_vg_verilogs"][idx] = g['input_verilogs']
            batch["vg_labels"][idx, :node_num] = input_nodes

        return batch
    

class CustomDataset(Dataset):
    def __init__(self, dataset: List[Dataset]):
        super(CustomDataset, self).__init__()
        self.ds_mgm = dataset[0]
        self.ds_vgalign = dataset[1]
        self.len_mgm = len(self.ds_mgm)
        self.len_vgalign = len(self.ds_vgalign)
        self.total_len = max(self.len_mgm, self.len_vgalign)
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        """
        Problems:
        If training for >1 epoch in unified mode, the same generative & embedding samples will 
        always be in the same batch as the same index is used for both datasets.
        Solution:
        Don't train for >1 epoch by duplicating the dataset you want to repeat in the folder.
        Upon loading, each dataset is shuffled so indices will be different.
        """
        mgm, vgalign = None, None

        if item >= self.len_mgm:
            random_item = random.randint(0, self.len_mgm - 1)
            mgm = self.ds_mgm[random_item]
        else:
            mgm = self.ds_mgm[item]

        if item >= self.len_vgalign:
            random_item = random.randint(0, self.len_vgalign - 1)
            vgalign = self.ds_vgalign[random_item]
        else:
            vgalign = self.ds_vgalign[item]

        return mgm, vgalign
    

class GraphDataCollator:
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max((g['input_edges'].shape[-1]) for g in graphs)
        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["node_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        # SSL Labels
        batch["input_indegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_outdegree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        # print(self.mask_ratio)
        mask_ratio = self.mask_ratio

        for idx, g in enumerate(graphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            masked_node_num = int(node_num * mask_ratio)

            edge_index = copy.deepcopy(g['input_edges'])

            indegree = degree(edge_index[0], num_nodes=node_num)
            outdegree = degree(edge_index[1], num_nodes=node_num)

            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_nodes"][idx, :node_num] = input_nodes + 1
            batch["node_mask"][idx][torch.randperm(node_num)[:masked_node_num]] = 1
            batch["padding_mask"][idx, :node_num] = 1

            batch["input_indegree"][idx, :node_num] = indegree
            batch["input_outdegree"][idx, :node_num] = outdegree
            batch["labels"][idx, :node_num] = input_nodes
        
        return batch


class GraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str):
        super(GraphDataset, self).__init__()
        fileDF = pd.read_csv(data_path)
        self.files = fileDF['fileName'].tolist()
        self.root_path = root_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")
        # print(g)
        mask = (g.edge_type == 1).squeeze()
        not_edge_index = g.edge_index[:, mask]
        buff_edge_index = g.edge_index[:, ~mask]

        num_nodes = g.node_type.shape[0]
        new_nodes_count = mask.sum().item()

        u, v = not_edge_index
        new_nodes = num_nodes + torch.arange(new_nodes_count)
        u_new = torch.stack([u, new_nodes])
        v_new = torch.stack([new_nodes, v])

        new_node_type = torch.tensor([3]*new_nodes_count, dtype=torch.long)
        new_edge_index = torch.cat([u_new, v_new], dim=1)

        node_type = torch.cat((g.node_type, new_node_type), dim=0)
        edge_index = torch.cat((buff_edge_index, new_edge_index), dim=1)

        return_dict = dict(input_nodes=node_type, input_edges=edge_index)
        return return_dict


class VerilogGraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, pyg_path: str, verilog_path: str):
        super(VerilogGraphDataset, self).__init__()
        self.pyg_path = pyg_path
        self.pyg_list = os.listdir(pyg_path)
        self.verilog_list = np.load(verilog_path, allow_pickle=True)[0]
        self.designs = [d.split(".")[0] for d in self.pyg_list]

    def __len__(self):
        return len(self.designs)

    def __getitem__(self, i):
        design = self.designs[i]
        pyg_file = osp.join(self.pyg_path, f"{design}.pt")
        g = torch.load(pyg_file, map_location="cpu")
        node_type = torch.tensor(g.node_type, dtype=torch.long)
        edge_type = torch.tensor(g.edge_type, dtype=torch.long)
        
        mask = (edge_type == 1).squeeze()
        not_edge_index = g.edge_index[:, mask]
        buff_edge_index = g.edge_index[:, ~mask]

        num_nodes = node_type.shape[0]
        new_nodes_count = mask.sum().item()

        u, v = not_edge_index
        new_nodes = num_nodes + torch.arange(new_nodes_count)
        u_new = torch.stack([u, new_nodes])
        v_new = torch.stack([new_nodes, v])

        new_node_type = torch.tensor([3]*new_nodes_count, dtype=torch.long)
        new_edge_index = torch.cat([u_new, v_new], dim=1)

        node_type = torch.cat((node_type, new_node_type), dim=0)
        edge_index = torch.cat((buff_edge_index, new_edge_index), dim=1)

        input_verilog = torch.tensor(self.verilog_list[design], dtype=float).squeeze(0)

        return_dict = dict(input_nodes=node_type, input_edges=edge_index, input_verilogs=input_verilog)
        return return_dict


class GraphQoRDataCollator:
    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max((g['input_edges'].shape[-1]) for g in graphs)
        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_synvecs"] = torch.zeros(batch_size, 20, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, dtype=torch.float)

        for idx, g in enumerate(graphs):
            input_nodes = g["input_nodes"]
            node_num = input_nodes.shape[0]
            edge_index = copy.deepcopy(g['input_edges'])

            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 0, edge_num:] = edge_index[0][0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]
            batch["input_edges"][idx, 1, edge_num:] = edge_index[1][0]

            batch["input_nodes"][idx, :node_num] = input_nodes + 1
            batch["input_synvecs"][idx] = g["input_synvecs"]
            batch["padding_mask"][idx, :node_num] = 1
            batch["labels"][idx] = g["labels"]
            # batch["labels"][idx] = g["labels"] / node_num
        
        return batch


class GraphQoRDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str, pkl_path: str, target: str = None):
        super(GraphQoRDataset, self).__init__()
        fileDF = pd.read_csv(data_path)
        self.files = fileDF['fileName'].tolist()

        with open(pkl_path,'rb') as f:
            self.targetStatsDict = pickle.load(f)
        if target is not None:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets(target)
        else:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets()
        self.target = target

        self.root_path = root_path

    def computeMeanAndVarianceOfTargets(self, targetVar='nodes'):
        meanAndVarTargetDict = {}
        for des in self.targetStatsDict.keys():
            numNodes, _, _, areaVar, delayVar = self.targetStatsDict[des]
            if targetVar == 'delay':
                meanTarget, varTarget = getMeanAndVariance(delayVar)
            elif targetVar == 'area':
                meanTarget, varTarget = getMeanAndVariance(areaVar)
            else:
                meanTarget, varTarget = getMeanAndVariance(numNodes)
            meanAndVarTargetDict[des] = [meanTarget,varTarget]
        return meanAndVarTargetDict
    
    def addNormalizedTargets(self, data):
        sid = data.synID[0]
        desName = data.desName[0]
        if self.target == 'delay':    
            targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        elif self.target == 'area':
            targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget],dtype=torch.float32)
        else:
            targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
            normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            # normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        return label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")

        mask = (g.edge_type == 1).squeeze()
        not_edge_index = g.edge_index[:, mask]
        buff_edge_index = g.edge_index[:, ~mask]

        num_nodes = g.node_type.shape[0]
        new_nodes_count = mask.sum().item()

        u, v = not_edge_index
        new_nodes = num_nodes + torch.arange(new_nodes_count)
        u_new = torch.stack([u, new_nodes])
        v_new = torch.stack([new_nodes, v])

        new_node_type = torch.tensor([3]*new_nodes_count, dtype=torch.long)
        new_edge_index = torch.cat([u_new, v_new], dim=1)

        node_type = torch.cat((g.node_type, new_node_type), dim=0)
        edge_index = torch.cat((buff_edge_index, new_edge_index), dim=1)

        synvec = g['synVec']
        label = self.addNormalizedTargets(g)

        return_dict = dict(input_nodes=node_type, input_edges=edge_index, input_synvecs=synvec, labels=label)
        return return_dict
    

class GraphQoRDeepGateDataCollator:
    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, 128, dtype=torch.float)
        batch["input_synvecs"] = torch.zeros(batch_size, 20, dtype=torch.long)
        batch["input_node_num"] = torch.zeros(batch_size, dtype=torch.long)
        batch["labels"] = torch.zeros(batch_size, dtype=torch.float)

        for idx, g in enumerate(graphs):
            node_num = g["input_node_num"]
            batch["input_nodes"][idx] = g["input_nodes"]
            batch["input_synvecs"][idx] = g["input_synvecs"]
            batch["input_node_num"] = node_num
            # batch["labels"][idx] = g["labels"]
            batch["labels"][idx] = g["labels"] / node_num * 100
        
        return batch


class GraphQoRDeepGateDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str, deepgate_path: str, pkl_path: str, target: str = None):
        super(GraphQoRDeepGateDataset, self).__init__()
        fileDF = pd.read_csv(data_path)
        self.files = []

        design_delete = ["or1200", "or1200_fpu", "aes_ncsu", 's35932', 's38417', 's38584']
        for f in fileDF['fileName'].tolist():
            if f.split("_syn")[0] not in design_delete:
                self.files.append(f)

        with open(pkl_path,'rb') as f:
            self.targetStatsDict = pickle.load(f)
        if target is not None:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets(target)
        else:
            self.meanVarDataDict = self.computeMeanAndVarianceOfTargets()
        self.target = target

        self.nodes = np.load(deepgate_path, allow_pickle=True)[0]

        self.root_path = root_path

    def computeMeanAndVarianceOfTargets(self, targetVar='nodes'):
        meanAndVarTargetDict = {}
        for des in self.targetStatsDict.keys():
            numNodes, _, _, areaVar, delayVar = self.targetStatsDict[des]
            if targetVar == 'delay':
                meanTarget, varTarget = getMeanAndVariance(delayVar)
            elif targetVar == 'area':
                meanTarget, varTarget = getMeanAndVariance(areaVar)
            else:
                meanTarget, varTarget = getMeanAndVariance(numNodes)
            meanAndVarTargetDict[des] = [meanTarget,varTarget]
        return meanAndVarTargetDict
    
    def addNormalizedTargets(self, data):
        sid = data.synID[0]
        desName = data.desName[0]
        if self.target == 'delay':    
            targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        elif self.target == 'area':
            targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget],dtype=torch.float32)
        else:
            targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
            # normTarget = (self.targetStatsDict[desName][targetIdentifier][sid] - self.meanVarDataDict[desName][0]) / self.meanVarDataDict[desName][1]
            normTarget = self.targetStatsDict[desName][targetIdentifier][sid]
            label = torch.tensor([normTarget], dtype=torch.float32)
        return label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filePathArchive = osp.join(self.root_path, self.files[i])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                g = torch.load(myfile, map_location="cpu")
        design = self.files[i].split("_syn")[0]

        node_num = g['node_type'].shape[0]

        nodes = torch.mean(torch.mean(torch.tensor(self.nodes[design], dtype=torch.float), dim=0), dim=0)
        synvec = g['synVec']
        label = self.addNormalizedTargets(g)

        return_dict = dict(input_nodes=nodes, input_synvecs=synvec, input_node_num=node_num, labels=label)
        return return_dict