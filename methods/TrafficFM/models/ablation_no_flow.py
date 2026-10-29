import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 基础组件 (保持不变)
# ==========================================

def make_supports(adj_mx, device):
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    mats = adj_mx if isinstance(adj_mx, (list, tuple)) else [adj_mx]
    supports = []
    for M in mats:
        if (sp is not None) and sp.issparse(M):
            arr = M.toarray()
        else:
            if hasattr(M, 'A'):
                arr = M.A
            else:
                arr = np.asarray(M)
        supports.append(torch.as_tensor(arr, dtype=torch.float32, device=device))
    return supports

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        x = torch.einsum('bcnt,nm->bcmt', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, use_softmax=True):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.order = order
        self.support_len = support_len
        self.use_softmax = use_softmax
        self.alpha = nn.Parameter(torch.zeros(support_len, order))
        self.beta = nn.Parameter(torch.tensor(1.0))
        c_in_mlp = (order * support_len + 1) * c_in
        self.mlp = linear(c_in_mlp, c_out)
        self.dropout = dropout

    def forward(self, x, support):
        outs = []
        outs.append(self.beta * x)
        for s, A in enumerate(support):
            if self.use_softmax:
                w = F.softmax(self.alpha[s], dim=0)
            else:
                w = torch.sigmoid(self.alpha[s])
                w = w / (w.sum() + 1e-12)
            x_power = self.nconv(x, A)
            for k in range(self.order):
                outs.append(w[k] * x_power)
                x_power = self.nconv(x_power, A)
        h = torch.cat(outs, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# ==========================================
# ==========================================

class MetaModulation(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.scale_proj = nn.Linear(emb_dim, channels)
        self.shift_proj = nn.Linear(emb_dim, channels)
        self.channels = channels

    def forward(self, x, E_joint):
        # x: [B, C, N, T]
        # E_joint: [B, N, emb_dim]
        scale = self.scale_proj(E_joint).permute(0, 2, 1).unsqueeze(-1) # [B, C, N, 1]
        shift = self.shift_proj(E_joint).permute(0, 2, 1).unsqueeze(-1) # [B, C, N, 1]
        out = x * (1 + scale) + shift
        return out

# ==========================================
# 跨注意力融合模块
# ==========================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads=8, dropout=0.0):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.query_conv = nn.Conv2d(query_dim, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(key_dim, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(value_dim, hidden_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(hidden_dim, key_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        B, _, N, T = key_value.size()
        Q = self.query_conv(query)
        K = self.key_conv(key_value)
        V = self.value_conv(key_value)
        Q = Q.view(B, self.num_heads, self.head_dim, N, T).permute(0, 1, 3, 4, 2)
        K = K.view(B, self.num_heads, self.head_dim, N, T).permute(0, 1, 3, 4, 2)
        V = V.view(B, self.num_heads, self.head_dim, N, T).permute(0, 1, 3, 4, 2)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, N, T)
        out = self.output_conv(attn_out)
        return out + key_value

# ==========================================
# 消融：简单的 MLP 输出头 (替代 Flow Matching)
# ==========================================

class MLPDecoder(nn.Module):
    """
    简单的 MLP 解码器，直接从条件特征预测输出序列
    输入: [B, cond_dim, N, T_in]
    输出: [B, out_dim, N, 1]
    """
    def __init__(self, cond_dim, out_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim

        # 构建多层 MLP
        layers = []
        current_dim = cond_dim

        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(current_dim, hidden_dim, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # 最后一层直接输出到目标维度
        layers.append(nn.Conv2d(current_dim, out_dim, kernel_size=1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, cond):
        """
        Args:
            cond: [B, cond_dim, N, T_in] 条件特征
        Returns:
            output: [B, out_dim, N, 1] 预测输出
        """
        # 对时间维度进行平均池化，得到 [B, cond_dim, N, 1]
        cond_pooled = torch.mean(cond, dim=3, keepdim=True)

        # 通过 MLP 得到 [B, out_dim, N, 1]
        output = self.mlp(cond_pooled)

        return output

def row_normalize(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row_sum = A.sum(dim=1, keepdim=True)
    A = A / (row_sum + eps)
    return A

def edge_dropout_rowwise(A: torch.Tensor, drop_prob: float, training: bool, eps: float = 1e-12) -> torch.Tensor:
    if (not training) or drop_prob <= 0.0: return A
    keep_prob = 1.0 - drop_prob
    mask = (torch.rand_like(A) < keep_prob).to(A.dtype)
    A_drop = A * mask
    row_sum = A_drop.sum(dim=1, keepdim=True)
    empty = (row_sum <= eps)
    if empty.any():
        A_drop = torch.where(empty, A, A_drop)
        row_sum = A_drop.sum(dim=1, keepdim=True)
    A_norm = A_drop / (row_sum + eps)
    return A_norm

# ==========================================
# 主模型 gwnet (无 Flow Matching 消融版本)
# ==========================================

class gwnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=144,  # 144 steps 预测
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=8,
        layers=3,
        adj_reg_l1=0.0,
        adj_reg_l2=0.0,
        adj_temperature=1.0,
        edge_drop=0.0,
        gn_groups=8,
        tod_emb_dim=8,
        dow_emb_dim=8,
        spatial_emb_dim=16,
        in_tod_idx=1,
        in_dow_idx=2,
        # ST Embedding Dimension
        st_emb_dim=32,
        # MLP Decoder 参数
        mlp_hidden_dim=512,
        mlp_num_layers=3,
        mlp_dropout=0.1
    ):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        self.in_tod_idx = in_tod_idx
        self.in_dow_idx = in_dow_idx

        self.adj_reg_l1 = adj_reg_l1
        self.adj_reg_l2 = adj_reg_l2
        self.adj_temperature = adj_temperature
        self.edge_drop = edge_drop

        self.tod_emb_dim = tod_emb_dim
        self.dow_emb_dim = dow_emb_dim
        self.spatial_emb_dim = spatial_emb_dim

        self.tod_emb_dict = nn.Embedding(288, tod_emb_dim)
        self.dow_emb_dict = nn.Embedding(7, dow_emb_dim)
        self.spatial_emb = nn.Parameter(torch.randn(num_nodes, spatial_emb_dim), requires_grad=True)

        self.joint_emb_dim = tod_emb_dim + dow_emb_dim + spatial_emb_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_meta_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.residual_meta_mod = MetaModulation(residual_channels, self.joint_emb_dim)
        self.skip_meta_mod = MetaModulation(skip_channels, self.joint_emb_dim)

        self.static_support_names = []
        if supports is not None:
            for idx, A in enumerate(supports):
                A = A.float()
                A_norm = row_normalize(A.clamp_min(0.0))
                name = f'static_support_{idx}'
                self.register_buffer(name, A_norm)
                self.static_support_names.append(name)

        self.nodevec1 = None
        self.nodevec2 = None
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            else:
                aptinit_t = torch.as_tensor(aptinit, dtype=torch.float32)
                U, S, Vh = torch.linalg.svd(aptinit_t, full_matrices=False)
                U10 = U[:, :10]
                S10 = torch.sqrt(S[:10])
                V10 = Vh[:10, :]
                initemb1 = U10 @ torch.diag(S10)
                initemb2 = torch.diag(S10) @ V10
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)

        # ST-Projection
        self.st_proj = nn.Conv2d(skip_channels, st_emb_dim, kernel_size=(1, 1))

        self.cross_attn = CrossAttentionFusion(
            query_dim=residual_channels,
            key_dim=st_emb_dim,
            value_dim=st_emb_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=0.0
        )

        receptive_field = 1
        self.supports_len = len(self.static_support_names) + (1 if (self.gcn_bool and self.addaptadj) else 0)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.residual_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1))
                )
                self.skip_meta_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.GroupNorm(num_groups=min(gn_groups, residual_channels), num_channels=residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.receptive_field = receptive_field

        # 消融：用简单的 MLP 替代 Flow Matching
        self.decoder = MLPDecoder(
            cond_dim=st_emb_dim,
            out_dim=out_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        )

    def _get_static_supports(self):
        return [getattr(self, name) for name in self.static_support_names]

    def _build_adaptive_adj(self):
        reg_loss = 0.0
        if (self.nodevec1 is None) or (self.nodevec2 is None):
            return None, reg_loss
        logits = F.relu(self.nodevec1 @ self.nodevec2)
        if self.adj_reg_l1 > 0.0:
            reg_loss = reg_loss + self.adj_reg_l1 * logits.abs().sum()
        if self.adj_reg_l2 > 0.0:
            reg_loss = reg_loss + self.adj_reg_l2 * (logits.pow(2).sum())
        tau = max(self.adj_temperature, 1e-6)
        adp = F.softmax(logits / tau, dim=1)
        adp = edge_dropout_rowwise(adp, self.edge_drop, self.training)
        adp_norm = row_normalize(adp)
        return adp_norm, reg_loss

    def _get_joint_embedding(self, input):
        # input: [B, C, N, T]
        B, C, N, T = input.size()
        tod_index = input[:, self.in_tod_idx, 0, T-1].long()
        dow_index = input[:, self.in_dow_idx, 0, T-1].long()

        E_tod = self.tod_emb_dict(tod_index)
        E_dow = self.dow_emb_dict(dow_index)
        E_t = torch.cat([E_tod, E_dow], dim=-1)

        E_s = self.spatial_emb
        E_t_b = E_t.unsqueeze(1).repeat(1, N, 1)
        E_s_b = E_s.unsqueeze(0).repeat(B, 1, 1)
        E_joint = torch.cat([E_t_b, E_s_b], dim=-1) # [B, N, dim]
        return E_joint

    def forward(self, input, y=None, **kwargs):
        """
        Args:
            input: [B, C, N, T_in] 输入序列
            y: [B, T_out, N, 1] 真实标签 (训练时使用)
        Returns:
            如果 y is None (推理): 返回 [B, T_out, N, 1] 预测结果
            如果 y is not None (训练): 返回标量损失
        """
        # 1. 提取嵌入
        E_joint = self._get_joint_embedding(input)

        # 2. 特征分离：仅使用第 0 通道 (速度)
        x_speed = input[:, 0:1, :, :]

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x_speed, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = x_speed

        # 适配输入维度
        if self.start_conv.in_channels != x.size(1):
            x = input if self.start_conv.in_channels == input.size(1) else x

        x = self.start_conv(x)
        skip = 0

        static_supports = self._get_static_supports()
        adp, adj_reg = None, 0.0
        if self.gcn_bool and self.addaptadj:
            adp, adj_reg = self._build_adaptive_adj()

        supports_to_use = list(static_supports)
        if self.gcn_bool and self.addaptadj and (adp is not None):
            supports_to_use = supports_to_use + [adp]

        layer_cnt = self.blocks * self.layers
        for i in range(layer_cnt):
            residual = x

            # WaveNet 核心计算
            filt = self.filter_convs[i](residual)
            filt = torch.tanh(filt)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filt * gate

            s = self.skip_meta_convs[i](x)
            s = self.skip_meta_mod(s, E_joint)

            if isinstance(skip, int):
                skip = s
            else:
                min_T = min(skip.size(3), s.size(3))
                skip = skip[:, :, :, -min_T:]
                s = s[:, :, :, -min_T:]
                skip = skip + s

            if self.gcn_bool and len(supports_to_use) > 0:
                x = self.gconv[i](x, supports_to_use)
            else:
                x = self.residual_convs[i](x)

            x = self.residual_meta_mod(x, E_joint)

            if residual.dim() == 4 and x.dim() == 4:
                min_T2 = min(residual.size(3), x.size(3))
                residual = residual[:, :, :, -min_T2:]
                x = x[:, :, :, -min_T2:]
                x = x + residual

            x = self.bn[i](x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 3. ST-Projection
        st_context = self.st_proj(skip)

        # 4. Cross Attention
        x_transformed = F.relu(x)
        condition = self.cross_attn(query=x_transformed, key_value=st_context)
        condition = F.relu(condition)

        # 5. MLP 解码器输出
        output = self.decoder(condition)  # [B, out_dim, N, 1]

        if y is None:
            # 推理模式：返回预测结果
            return output
        else:
            # 训练模式：计算损失
            # y: [B, T_out, N, 1], output: [B, T_out, N, 1]
            mse_loss = F.mse_loss(output, y)
            mae_loss = F.l1_loss(output, y)

            # 复合损失：MSE + 0.5 * MAE
            total_loss = mse_loss + 0.5 * mae_loss + adj_reg

            return total_loss
