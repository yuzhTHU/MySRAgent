# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, residual=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.residual = residual
        
        self.update_node = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_edge = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        if residual:
            self.shortcut_node = nn.Linear(node_dim, out_dim) \
                                if node_dim != out_dim else nn.Identity()
            self.shortcut_edge = nn.Linear(edge_dim, out_dim) \
                                if edge_dim != out_dim else nn.Identity()

    # ⚠️ 移除了 A，并将 G 改名为 edge_list 以防混淆
    def forward(self, v, e, edge_list, num_nodes):
        """
        - v: [N, V, node_dim]      (N: SampleNum, V: TotalNodes)
        - e: [N, E, edge_dim]      (N: SampleNum, E: TotalEdges)
        - edge_list: [2, E]        (第0行: source, 第1行: target)
        - num_nodes: [N]           (每个样本的节点数量)
        - return: [N, V, out_dim], [N, E, out_dim]
        """
        N, V, _ = v.shape
        _, E, _ = e.shape
        
        # 1. 获取源节点和目标节点的索引: [E]
        sour_idx = edge_list[0]
        term_idx = edge_list[1]
        
        # 2. 提取特征: [N, E, node_dim]
        v_sour = v[:, sour_idx, :] 
        v_term = v[:, term_idx, :] 
        x = torch.cat([v_sour, v_term, e], dim=-1)

        # 3. 计算节点消息
        y = self.update_node(x) # [N, E, out_dim]
        
        # 4. 消息聚合 (Scatter Add)
        update_v = torch.zeros(N, V, self.out_dim, device=y.device)
        # 调整 target index 的形状以匹配 src (y)
        # term_idx.view(1, -1, 1) -> [1, E, 1] -> 展开为 [N, E, out_dim]
        scatter_idx = term_idx.view(1, -1, 1).expand(N, E, self.out_dim)
        update_v.scatter_add_(dim=1, index=scatter_idx, src=y)
        
        # 🌟 5. 核心修改：动态计算稀疏度数 (Degree)
        # torch.bincount 可以统计 term_idx 中每个节点作为目标节点出现了多少次
        degree = torch.bincount(term_idx, minlength=V).clamp(min=1).float() # [V,]
        
        # 将 degree 调整为 [1, V, 1] 以支持广播除法
        update_v = update_v / degree.view(1, V, 1) # [N, V, out_dim]
        
        # 6. 计算边更新
        update_e = self.update_edge(x) # [N, E, out_dim]

        if self.residual:
            update_v = update_v + self.shortcut_node(v)
            update_e = update_e + self.shortcut_edge(e)

        return F.relu(update_v), F.relu(update_e)


class GNN(nn.Module):
    def __init__(self, d_emb, n_layers, node_dim, edge_dim, dropout):
        super(GNN, self).__init__()
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        self.GNN_layers = nn.ModuleList([GNNLayer(node_dim, edge_dim, d_emb)] + \
                                        [GNNLayer(d_emb, d_emb, d_emb) for _ in range(n_layers - 1)])
        self.dropout = nn.Dropout(p=dropout)

    def fwd(self, v, e, edge_list, num_nodes):
        for net in self.GNN_layers:
            v, e = net(v, e, edge_list, num_nodes)
            v = self.dropout(v)
            e = self.dropout(e)
        return v, e

    def forward(self, v, e, edge_list, num_nodes, chunk_size=None):
        if chunk_size is None or chunk_size >= v.size(0):
            return self.fwd(v, e, edge_list, num_nodes)
        
        v_out_list = []
        e_out_list = []
        for v_chunk, e_chunk in zip(
            torch.split(v, chunk_size, dim=0), 
            torch.split(e, chunk_size, dim=0)
        ):
            v_chunk_out, e_chunk_out = self.fwd(v_chunk, e_chunk, edge_list, num_nodes)
            v_out_list.append(v_chunk_out)
            e_out_list.append(e_chunk_out)
        v_final = torch.cat(v_out_list, dim=0)
        e_final = torch.cat(e_out_list, dim=0)
        
        return v_final, e_final
