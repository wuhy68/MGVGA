import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss, HuberLoss

from torch_geometric.nn import DeepGCNLayer, GENConv, GCNConv, SAGEConv

from transformers.activations import ACT2FN
from transformers.modeling_utils import ModelOutput, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)

from transformers.configuration_utils import PretrainedConfig
from aigmae.configuration_vgmae import AIGMAEConfig  
        
@dataclass
class AIGMAEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None
    vg_node_loss: Optional[Tuple[torch.FloatTensor]] = None
    indegree_loss: Optional[torch.FloatTensor] = None
    indegree_loss_t: Optional[torch.FloatTensor] = None
    outdegree_loss: Optional[torch.FloatTensor] = None
    outdegree_loss_t: Optional[torch.FloatTensor] = None

@dataclass
class AIGMAEQoROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    

class AIGMAEFeature(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        self.node_token_emb = nn.Embedding(config.num_classes + 1, config.hidden_size, padding_idx=0)

    def forward(
        self,
        input_nodes: torch.Tensor,
    ) -> torch.Tensor:
        graph_node_feature = self.node_token_emb(input_nodes) # [n_graph, n_node, n_hidden]
        
        return graph_node_feature


class AIGMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class AIGCrossAttention(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super(AIGCrossAttention, self).__init__()
        assert config.hidden_size % config.cross_num_heads == 0

        self.d_model = config.hidden_size
        self.num_heads = config.cross_num_heads
        self.head_dim = config.hidden_size // config.cross_num_heads

        # Linear layers for Q, K, V projections
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.cross_hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.cross_hidden_size, config.hidden_size)

        # Output linear layer
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self, 
        g_emb: torch.Tensor, 
        v_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size = g_emb.size(0)
        v_emb = v_emb.unsqueeze(1)
        
        # Perform linear projections
        q = self.linear_q(g_emb)  # (B, Lq, d_model)
        k = self.linear_k(v_emb)    # (B, 1, d_model)
        v = self.linear_v(v_emb)  # (B, 1, d_model)
        
        # Reshape Q, K, V for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lq, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lk, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lk, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply the query padding mask
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, Lq, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Final linear projection
        attn_output = self.linear_out(attn_output)
        
        return attn_output


class AIGMAEBlock(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        conv = GENConv(config.hidden_size, config.hidden_size, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
        norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.layer = DeepGCNLayer(conv, norm, block='res+', dropout=0.0)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g = input_nodes.size(0)
        graphs_rep = []
        for i in range(g):
            o = self.layer(input_nodes[i], input_edges[i])
            graphs_rep.append(o.unsqueeze(0))
        graphs_rep = torch.cat(graphs_rep, dim=0)

        if padding_mask is not None:
            graphs_rep = graphs_rep * padding_mask.unsqueeze(-1)

        return graphs_rep
    

class AIGAlignBlock(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.attn = AIGCrossAttention(config)
        self.mlp = AIGMLP(config.hidden_size)
        self.attn_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_verilogs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = input_nodes
        input_nodes = self.attn_norm(input_nodes)
        graphs_rep = self.attn(input_nodes, input_verilogs, padding_mask)
        graphs_rep = graphs_rep + residual

        residual = graphs_rep
        graphs_rep = self.mlp_norm(graphs_rep)
        graphs_rep = self.mlp(graphs_rep)
        graphs_rep = graphs_rep + residual

        return graphs_rep


class AIGMAEEncoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.config = config

        self.layers = nn.ModuleList([])
        self.layers.extend([AIGMAEBlock(config) for _ in range(config.num_encoder_layers - 1)])
        self.act = nn.ReLU()
        self.conv = AIGMAEBlock(config)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            input_nodes = layer(
                input_nodes=input_nodes,
                input_edges=input_edges,
                padding_mask=padding_mask,
            )
            input_nodes = self.act(input_nodes)

        graph_rep = self.conv(input_nodes, input_edges, padding_mask)

        return graph_rep

    
class AIGMAEDecoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.conv = AIGMAEBlock(config)

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        graph_rep = self.conv(input_nodes, input_edges, padding_mask)

        return graph_rep
    

class AIGAlignDecoder(nn.Module):
    def __init__(self, config: AIGMAEConfig):
        super().__init__()
        
        self.config = config

        self.layers = nn.ModuleList([])
        self.layers.extend([AIGAlignBlock(config) for _ in range(config.num_cross_decoder_layers)])

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_verilogs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            input_nodes = layer(
                input_nodes=input_nodes,
                input_verilogs=input_verilogs,
                padding_mask=padding_mask,
            )

        return input_nodes


class AIGMAENodePredictionHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        logits = self.classifier(input_nodes)
        return logits


class AIGMAEDegreePredictionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mlp = AIGMLP(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        logits = self.classifier(self.mlp(input_nodes))
        return logits


class AIGMAEPreTrainedModel(PreTrainedModel):
    config_class = AIGMAEConfig

    def normal_(self, data: torch.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, 
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AIGMAEEmbeddingModel(AIGMAEPreTrainedModel):

    def __init__(self, config: AIGMAEConfig):
        super().__init__(config)

        self.config = config

        self.graph_encoder = AIGMAEEncoder(config)

        self.post_init()
       
    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        graph_emb = self.graph_encoder(
            input_nodes, input_edges, padding_mask
        )

        if padding_mask is not None:
            graph_emb = (graph_emb * padding_mask.unsqueeze(-1)).sum(1) / padding_mask.sum(1).unsqueeze(-1)
            graph_emb = graph_emb.float()
        else:
            graph_emb = torch.mean(graph_emb, dim=1)

        if not return_dict:
            return tuple(x for x in [graph_emb] if x is not None)
        return AIGMAEOutput(hidden_states=graph_emb)


class AIGMAEModel(AIGMAEPreTrainedModel):

    def __init__(self, config: AIGMAEConfig):
        super().__init__(config)

        self.config = config

        self.token_emb = AIGMAEFeature(config)

        self.graph_encoder = AIGMAEEncoder(config)
        self.graph_decoder = AIGMAEDecoder(config)
        self.vg_decoder = AIGAlignDecoder(config)

        self.mask_token = nn.Embedding(1, config.hidden_size)
        self.vg_mask_token = nn.Embedding(1, config.hidden_size)

        self.node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.vg_node_prediction_head = AIGMAENodePredictionHead(config.hidden_size, config.num_classes)
        self.indegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)
        self.outdegree_prediction_head = AIGMAEDegreePredictionHead(config.hidden_size)

        self.post_init()

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_edges: torch.Tensor,
        node_mask: torch.Tensor,
        input_vg_nodes: torch.Tensor,
        input_vg_edges: torch.Tensor,
        input_vg_verilogs: torch.Tensor,
        vg_node_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        vg_padding_mask: Optional[torch.Tensor] = None,
        input_indegree: Optional[torch.Tensor] = None,
        input_outdegree: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vg_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MGM Tasks
        input_nodes = self.token_emb(input_nodes)
        encoder_graph_rep = self.graph_encoder(
            input_nodes, input_edges, padding_mask
        )

        masked_graph_rep = encoder_graph_rep
        node_mask = node_mask.bool() 
        indices = torch.where(node_mask)[0], torch.where(node_mask)[1]
        masked_graph_rep[indices] = self.mask_token.weight.expand_as(masked_graph_rep[indices])

        decoder_graph_rep = self.graph_decoder(
            masked_graph_rep, input_edges, padding_mask
        )

        node_logits = self.node_prediction_head(decoder_graph_rep)
        indegree_logits = self.indegree_prediction_head(decoder_graph_rep)
        outdegree_logits = self.outdegree_prediction_head(decoder_graph_rep)

        # VGAlign Task
        input_vg_nodes = self.token_emb(input_vg_nodes)
        masked_vg_graph_rep = input_vg_nodes
        vg_node_mask = vg_node_mask.bool()
        vg_indices = torch.where(vg_node_mask)[0], torch.where(vg_node_mask)[1]
        masked_vg_graph_rep[vg_indices] = self.vg_mask_token.weight.expand_as(masked_vg_graph_rep[vg_indices])
        encoder_vg_graph_rep = self.graph_encoder(
            masked_vg_graph_rep, input_vg_edges, vg_padding_mask
        )

        decoder_vg_graph_rep = self.vg_decoder(
            encoder_vg_graph_rep, input_vg_verilogs, vg_padding_mask
        )

        vg_node_logits = self.vg_node_prediction_head(decoder_vg_graph_rep)

        loss = None
        node_loss = None
        vg_node_loss = None
        degree_loss = None
        
        if labels is not None:
            # NodeType Prediction
            node_target = labels * node_mask + (node_mask == 0) * -100
            node_loss = F.cross_entropy(node_logits.view(-1, self.config.num_classes), node_target.view(-1), reduction='mean')
            vg_node_target = vg_labels * vg_node_mask + (vg_node_mask == 0) * -100
            vg_node_loss = F.cross_entropy(vg_node_logits.view(-1, self.config.num_classes), vg_node_target.view(-1), reduction='mean')

            # Degree & Level Prediction
            degree_mask = (padding_mask == 1)
            indegree_loss = F.l1_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float())
            outdegree_loss = F.l1_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float())
            indegree_loss_t = F.huber_loss(indegree_logits[degree_mask].squeeze(), input_indegree[degree_mask].squeeze().float(), delta=1.0)
            outdegree_loss_t = F.huber_loss(outdegree_logits[degree_mask].squeeze(), input_outdegree[degree_mask].squeeze().float(), delta=1.0)
            
            degree_loss = indegree_loss_t + outdegree_loss_t
            degree_loss = degree_loss * 0.1

            loss = node_loss + vg_node_loss + degree_loss

        if not return_dict:
            return tuple(x for x in [loss, node_loss, vg_node_loss, indegree_loss, indegree_loss_t, outdegree_loss, outdegree_loss_t] if x is not None)
        return AIGMAEOutput(loss=loss, node_loss=node_loss, vg_node_loss=vg_node_loss,
                            indegree_loss=indegree_loss, indegree_loss_t=indegree_loss_t, 
                            outdegree_loss=outdegree_loss, outdegree_loss_t=outdegree_loss_t)