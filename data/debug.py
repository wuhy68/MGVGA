import torch
import pandas as pd
import numpy as np
import os
import zipfile
import copy

from tqdm import tqdm

import torch
import pandas as pd
from torch.utils.data import Dataset

import torch    
import numpy as np
from typing import Any, Dict, List, Mapping

import torch.nn as nn
from torch_geometric.nn import SAGEConv 
from transformers.models.graphormer.configuration_graphormer import GraphormerConfig


class GraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, root_path: str, data_path: str):
        super(GraphDataset, self).__init__()
        files = pd.read_csv(data_path)['file'].tolist()
        self.data = []
        for f in files:
            self.data.append(torch.load(f"{root_path}/{f}", map_location="cpu"))
        self.data = self.data[:1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        g = self.data[i]
        node_type=g['node_type']
        edge_index=g['edge_index']
        label=[g['nodes']]
        return_dict = dict(input_nodes=node_type, input_edges=edge_index, labels=label)
        return return_dict
    

class GraphDataCollator:
    def __call__(self, graphs: List[dict]) -> Dict[str, Any]:

        max_node_num = max(g['input_nodes'].shape[0] for g in graphs)
        max_edge_num = max(2 * (g['input_edges'].shape[-1] + g['input_nodes'].shape[0] + 1) for g in graphs)
        batch_size = len(graphs)
        batch = {}

        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_edges"] = torch.zeros(batch_size, 2, max_edge_num, dtype=torch.long)
        batch["padding_mask"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)

        for idx, g in enumerate(graphs):
            print(g['input_edges'])
            node_num = g["input_nodes"].shape[0]
            edge_index = copy.deepcopy(g['input_edges'])
            edge_index[0] += 1
            edge_index[1] += 1
            f = [0] * (node_num + 1)
            s = [i for i in (range(node_num + 1))]
            edge_sup = torch.tensor([f, s],  dtype=torch.long)
            edge_index = torch.cat([edge_index, edge_sup], 1)
            edge_index = torch.cat([edge_index[[1, 0]], edge_index], 1)
            edge_num = edge_index.shape[1]
            batch["input_edges"][idx, 0, :edge_num] = edge_index[0]
            batch["input_edges"][idx, 1, :edge_num] = edge_index[1]

            batch["input_nodes"][idx, :node_num] = g["input_nodes"] + 1
            batch["padding_mask"][idx, :node_num] = 1
        
        batch["labels"] = torch.from_numpy(np.concatenate([g["labels"] for g in graphs]))

        return batch


class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads

        self.node_token_embedding = nn.Embedding(3 + 1, config.hidden_size, padding_idx=0)
        self.graph_token_embedding = nn.Embedding(1, config.hidden_size)
        self.edge_embedding = SAGEConv(in_channels=config.hidden_size, out_channels=config.hidden_size)

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
    ) -> torch.Tensor:
        n_graph = input_nodes.shape[0]

        node_feature = self.node_token_embedding(input_nodes) # [n_graph, n_node, n_hidden]
        graph_token_feature = self.graph_token_embedding.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        graph_edge_feature = []
        
        for i in range(n_graph):
            graph_edge_feature.append(self.edge_embedding(graph_node_feature[i], input_edges[i]).unsqueeze(0))
        graph_edge_feature = torch.cat(graph_edge_feature, dim=0)

        return graph_node_feature, graph_edge_feature

def main():
    train_dataset = GraphDataset(root_path = "/datasets/netlist_dataset/OPENABC1", 
                                data_path = "/data/openabc1/train.csv")
    config = GraphormerConfig()

    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn=GraphDataCollator(), batch_size=1, shuffle=False)
    model = GraphormerGraphNodeFeature(config)
    model = model.cuda()
    for i in range(2):
        for step, data in enumerate(dataloader):
            print(data["input_nodes"], data["input_edges"])
            print(data["input_nodes"].shape, data["input_edges"].shape)
            print(len(data["input_nodes"][0]))
            print(max(data["input_edges"][0][0]).numpy())
            try:
                graph_node_embedding, graph_edge_embedding = model(data["input_nodes"].cuda(), data["input_edges"].cuda())
                # print(graph_node_embedding.shape)
                # print(graph_edge_embedding.shape)
            except:
                print(data["input_nodes"], data["input_edges"])
                print(data["input_nodes"].shape, data["input_edges"].shape)
                print(len(data["input_nodes"][0]))
                print(max(data["input_edges"][0][0]).numpy())
                break
    # for step, data in enumerate(dataloader):
    #     if len(data["input_nodes"][0]) == max(data["input_edges"][0][0]).numpy():
    #         print(data["input_nodes"])
    #         print(data["input_edges"])
    #         break

if __name__ == "__main__":
    main()