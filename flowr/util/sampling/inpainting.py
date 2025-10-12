import torch
from copy import deepcopy
from torch_geometric.data import Data, Batch
from typing import List, Union, Optional
import numpy as np

def create_copy_and_fill(data: Data, lig_mask: torch.Tensor) -> Data:
    
    n_total = len(lig_mask)
    n_fixed = data.pos.size(0)
    assert n_fixed <= n_total
    pos = data.pos
    x = data.x
    charges = data.charges
    pos_new = torch.zeros((n_total, pos.size(1)),
                          device=pos.device, 
                          dtype=pos.dtype)
    x_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
    charges_new = torch.zeros((n_total,), device=x.device, dtype=x.dtype)
    pos_new[:n_fixed,:] = pos
    x_new[:n_fixed] = x
    charges_new[:n_fixed] = charges
    data_copy = deepcopy(data)
    tmp = Data(x=x_new, pos=pos_new, charges=charges_new, lig_inpaint_mask=lig_mask)
    data_copy.update(tmp)
    
    return data_copy

def get_edge_mask_inpainting(edge_index: torch.Tensor, edge_attr: torch.Tensor, fixed_nodes_indices: torch.Tensor):
    
    if str(fixed_nodes_indices.dtype) == torch.bool:
        fixed_nodes_indices = torch.where(fixed_nodes_indices)[0]
        
    edge_0 = torch.where(
                    edge_index[0][:, None] == fixed_nodes_indices[None, :]
                )[0]
    
    edge_1 = torch.where(
        edge_index[1][:, None] == fixed_nodes_indices[None, :]
    )[0]
    
    edge_index_between_fixed_nodes = edge_0[
        torch.where(edge_0[:, None] == edge_1[None, :])[0]
    ]
    
    edge_mask_between_fixed_nodes = torch.zeros_like(
        edge_attr, dtype=torch.bool, device=edge_index.device
    )
    
    edge_mask_between_fixed_nodes[edge_index_between_fixed_nodes] = True
    
    return edge_index_between_fixed_nodes, edge_mask_between_fixed_nodes

def prepare_inpainting_ligand_batch(data: Data,
                                    vary_n_nodes: bool, 
                                    nodes_bias: Optional[int], 
                                    num_graphs: int,
                                    device, 
                                    keep_ids):
    
    n_fixed = data.pos.size(0)
    if vary_n_nodes:
        nodes_bias_ = torch.randint(low=1,
                                    high=nodes_bias,
                                    size=(num_graphs,), 
                                    device=device,
                                    dtype=torch.long)
    else:
        nodes_bias_ = torch.ones((num_graphs,), device=device, dtype=torch.long).fill_(nodes_bias)
        
    lig_mask_added = [torch.tensor([False] * n.item()).to(device) for n in nodes_bias_]
    
    if keep_ids is None:
        lig_mask = torch.ones((n_fixed,), dtype=torch.bool, device=device)
    else:
        lig_mask = torch.zeros((n_fixed,), dtype=torch.bool, device=device)
        if isinstance(keep_ids, list):
            keep_ids = torch.tensor(keep_ids, device=device, dtype=torch.long)
        elif isinstance(keep_ids, np.ndarray):
            keep_ids = torch.from_numpy(keep_ids).to(device).long()
        lig_mask[keep_ids] = True
        
    lig_mask_batch = [torch.concat((lig_mask, added), dim=0) for added in lig_mask_added]
    n_atoms_total = n_fixed + nodes_bias_
    datalist = [create_copy_and_fill(data, lig_mask=l) for l in  lig_mask_batch]
    return Batch.from_data_list(datalist)