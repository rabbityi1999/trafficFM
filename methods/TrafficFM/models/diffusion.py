import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==========================================
# 基础组件 (保留原样)
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
        scale = self.scale_proj(E_joint).permute(0, 2, 1).unsqueeze(-1) # [B, C, N, 1]
        shift = self.shift_proj(E_joint).permute(0, 2, 1).unsqueeze(-1) # [B, C, N, 1]
        out = x * (1 + scale) + shift
        return out

# ==========================================
# Cross Attention Fusion
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
# UNet Components
# ==========================================

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(self.max_freq), steps=half, device=device))
        # t can be either float (for continuous) or int (for discrete steps).
        # Ensure t is broadcastable
        if t.dim() == 1:
            t = t.unsqueeze(1)
        args = t * freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.size(1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(1)))
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, cond_ch=None, time_emb_dim=None, kernel_size=(3, 1), groups=8):
        super().__init__()
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2 
        if kernel_size[1] == 1: pad_w = 0 
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=(pad_h, pad_w))
        self.gn1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=(pad_h, pad_w))
        self.gn2 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.time_proj = None
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, in_ch)
        self.cond_proj = None
        if cond_ch is not None:
            self.cond_proj = nn.Conv2d(cond_ch, in_ch, kernel_size=1)

    def forward(self, x, t_emb=None, cond_emb=None):
        h = self.conv1(x)
        h = self.gn1(h)
        if t_emb is not None and self.time_proj is not None:
            t_feat = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t_feat
        if cond_emb is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond_emb)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.relu(h, inplace=True)
        return x + h

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(ch, ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
    def forward(self, x, target_size_N):
        x = self.tconv(x)
        if x.size(2) > target_size_N: x = x[:, :, :target_size_N, :]
        elif x.size(2) < target_size_N: x = F.pad(x, (0, 0, 0, target_size_N - x.size(2)))
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, in_ch, cond_ch, time_emb_dim=128, gn_groups=8):
        super().__init__()
        self.cond_reduce = nn.Conv2d(cond_ch, in_ch, kernel_size=1)
        self.time_emb = ContinuousTimeEmbedding(time_emb_dim)
        self.rb1 = ResidualBlock(in_ch, cond_ch=in_ch, time_emb_dim=time_emb_dim, groups=gn_groups)
        self.down1 = Downsample(in_ch)
        self.rb2 = ResidualBlock(in_ch, cond_ch=in_ch, time_emb_dim=time_emb_dim, groups=gn_groups)
        self.down2 = Downsample(in_ch)
        self.rb_mid = ResidualBlock(in_ch, cond_ch=in_ch, time_emb_dim=time_emb_dim, groups=gn_groups)
        self.up2 = Upsample(in_ch)
        self.rb3 = ResidualBlock(in_ch, cond_ch=in_ch, time_emb_dim=time_emb_dim, groups=gn_groups)
        self.up1 = Upsample(in_ch)
        self.rb4 = ResidualBlock(in_ch, cond_ch=in_ch, time_emb_dim=time_emb_dim, groups=gn_groups)
        self.out = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def _align_cond(self, cond_emb, x_like):
        return F.adaptive_avg_pool2d(cond_emb, output_size=(x_like.size(2), x_like.size(3)))

    def forward(self, x, cond, t):
        if cond is None: cond_base = torch.zeros_like(x)
        else: cond_base = self.cond_reduce(self._align_cond(cond, x))
        t_emb = self.time_emb(t)
        
        h1 = self.rb1(x, t_emb=t_emb, cond_emb=self._align_cond(cond_base, x))
        d1 = self.down1(h1)
        h2 = self.rb2(d1, t_emb=t_emb, cond_emb=self._align_cond(cond_base, d1))
        d2 = self.down2(h2)
        h_mid = self.rb_mid(d2, t_emb=t_emb, cond_emb=self._align_cond(cond_base, d2))
        u2 = self.up2(h_mid, target_size_N=h2.size(2)) + h2
        h3 = self.rb3(u2, t_emb=t_emb, cond_emb=self._align_cond(cond_base, u2))
        u1 = self.up1(h3, target_size_N=h1.size(2)) + h1
        h4 = self.rb4(u1, t_emb=t_emb, cond_emb=self._align_cond(cond_base, u1))
        return self.out(h4)

# ==========================================
# 关键替换：Diffusion Component
# 替换原有的 FlowMatchingUNet
# ==========================================

def extract(a, t, x_shape):
    """Helper to extract coefficients at specified timesteps t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)  
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DiffusionUNet(nn.Module):
    def __init__(self, input_dim, cond_dim, out_dim, gn_groups=8, patch_size=12, hidden_dim=256, l1_lambda=0.5, total_steps=1000):
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.l1_lambda = l1_lambda 
        self.total_steps = total_steps
        assert out_dim % patch_size == 0, f"out_dim {out_dim} must be divisible by patch_size {patch_size}"
        self.num_patches = out_dim // patch_size
        
        # Token Embedding (Patching)
        self.token_embed = nn.Conv2d(patch_size, hidden_dim, kernel_size=1)
        self.token_decode = nn.Conv2d(hidden_dim, patch_size, kernel_size=1)
        self.cond_proj = nn.Conv2d(cond_dim, hidden_dim, kernel_size=1)
        
        # Backbone
        self.unet = ConditionalUNet(in_ch=hidden_dim, cond_ch=hidden_dim, time_emb_dim=128, gn_groups=gn_groups)

        # Diffusion Hyperparameters (Linear Schedule)
        scale = 1000 / total_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float32)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers to move with device
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _process_input(self, y):
        # [B, T, N, 1] or [B, T, N, C]
        B, T, N, _ = y.shape
        y = y.view(B, self.num_patches, self.patch_size, N)
        y = y.permute(0, 2, 3, 1) # [B, patch_size, N, num_patches]
        emb = self.token_embed(y)
        return emb

    def _process_output(self, emb):
        out = self.token_decode(emb)
        B, _, N, num_patches = out.shape
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(B, self.out_dim, N, 1)
        return out

    def loss(self, y_true, cond, drop_prob=0.1):
        """
        Standard DDPM Training Loss: Predict noise epsilon
        """
        device = y_true.device
        B = y_true.shape[0]
        
        # 1. Prepare Data (x_0)
        x_start = self._process_input(y_true)
        
        # 2. Sample Time t and Noise epsilon
        t = torch.randint(0, self.total_steps, (B,), device=device).long()
        noise = torch.randn_like(x_start)
        
        # 3. Forward Diffusion q(x_t | x_0)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        # 4. Condition Processing & Dropout
        cond = self.cond_proj(cond)
        if drop_prob > 0.0 and self.training:
            mask = (torch.rand(B, device=device) < drop_prob).float().view(B, 1, 1, 1)
            cond = cond * (1.0 - mask)
            
        # 5. Predict Noise
        noise_pred = self.unet(x_t, cond, t.float()) # Pass t as float for ContinuousEmbedding

        # 6. Loss Calculation
        mse_loss = F.mse_loss(noise_pred, noise)
        l1_loss = F.l1_loss(noise_pred, noise)
        composite_loss = mse_loss + self.l1_lambda * l1_loss

        return composite_loss

    def sample(self, cond, guidance_scale=0.0, fm_steps=20, solver='euler', track_grad=False):
        """
        Sampling using DDIM for efficiency, mapped from the original 'fm_steps'.
        Ignoring 'solver' param as we use DDIM/DDPM logic.
        """
        device = cond.device
        B, _, N, _ = cond.shape
        cond = self.cond_proj(cond)
        
        # Start from Gaussian Noise
        x = torch.randn(B, self.hidden_dim, N, self.num_patches, device=device, dtype=cond.dtype)
        
        # Determine Timesteps (DDIM)
        # Map fm_steps (e.g., 20) to diffusion timesteps subsequence
        step_ratio = self.total_steps // fm_steps
        timesteps = (torch.arange(fm_steps - 1, -1, -1) * step_ratio).long().to(device) # [950, 900, ..., 0]
        
        ctx = torch.enable_grad if track_grad else torch.no_grad
        with ctx():
            for i, t_step in enumerate(timesteps):
                # Batch of current timestep
                t_tensor = torch.full((B,), t_step, device=device, dtype=torch.float32)
                t_long = torch.full((B,), t_step, device=device, dtype=torch.long)
                
                # Classifier-Free Guidance
                if guidance_scale > 0.0:
                    # Duplicate input for [uncond, cond]
                    x_in = torch.cat([x] * 2)
                    t_in = torch.cat([t_tensor] * 2)
                    cond_uncond = torch.zeros_like(cond)
                    cond_in = torch.cat([cond_uncond, cond])
                    
                    noise_pred = self.unet(x_in, cond_in, t_in)
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    e_t = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    e_t = self.unet(x, cond, t_tensor)
                
                # DDIM Update
                # x_{t-1} = sqrt(alpha_{t-1}) * (x_t - sqrt(1-alpha_t)*e_t) / sqrt(alpha_t) + sqrt(1-alpha_{t-1}) * e_t
                # (Assuming sigma=0 for deterministic DDIM)
                
                alpha_bar_t = extract(self.alphas_cumprod, t_long, x.shape)
                alpha_bar_prev = extract(self.alphas_cumprod, torch.max(t_long - step_ratio, torch.tensor(0, device=device)), x.shape)
                
                # Predict x_0
                pred_x0 = (x - extract(self.sqrt_one_minus_alphas_cumprod, t_long, x.shape) * e_t) / extract(self.sqrt_alphas_cumprod, t_long, x.shape)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * e_t
                
                # Update x
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            y_pred = self._process_output(x)
        return y_pred

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
        in_dim=1,
        out_dim=24,
        residual_channels=16,#32
        dilation_channels=16,#32
        skip_channels=128,#256
        end_channels=512,
        kernel_size=2,
        blocks=8,
        layers=3,
        adj_reg_l1=0.0,
        adj_reg_l2=0.0,
        adj_temperature=1.0,
        edge_drop=0.0,
        gn_groups=8,
        patch_size=12,
        tod_emb_dim=8,
        dow_emb_dim=8,
        spatial_emb_dim=16,
        in_tod_idx=1,
        in_dow_idx=2,
        # 新增参数: Spatiotemporal Embedding Dimension (用于解码器条件)
        st_emb_dim=32,
        # 复合损失函数参数: L1 loss权重系数
        l1_lambda=0.5,
        # Cross-Attention hidden dimension
        cross_attn_hidden_dim=512
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
        
        # ----------------------------------------------------
        # ----------------------------------------------------
        self.tod_emb_dim = tod_emb_dim
        self.dow_emb_dim = dow_emb_dim
        self.spatial_emb_dim = spatial_emb_dim
        
        self.tod_emb_dict = nn.Embedding(288, tod_emb_dim)
        self.dow_emb_dict = nn.Embedding(7, dow_emb_dim) 
        self.spatial_emb = nn.Parameter(torch.randn(num_nodes, spatial_emb_dim), requires_grad=True)

        # Meta-Query 维度
        self.joint_emb_dim = tod_emb_dim + dow_emb_dim + spatial_emb_dim
        
        # ----------------------------------------------------
        
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
            hidden_dim=cross_attn_hidden_dim,
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

        # Diffusion Integrated (Replacing Flow Matching)
        self.flow = DiffusionUNet(
            input_dim=skip_channels,
            cond_dim=st_emb_dim, # 使用 ST-Embedding 作为条件
            out_dim=out_dim,
            gn_groups=gn_groups,
            patch_size=patch_size,
            l1_lambda=l1_lambda,
            total_steps=1000 # 标准 Diffusion 步数
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

    def forward(self, input, y=None, guidance_scale=0.0, drop_prob=0.1, fm_steps=20, solver='euler'):
        # 1. 提取嵌入
        E_joint = self._get_joint_embedding(input) 
        
        # 2. 特征分离
        x_speed = input[:, 0:1, :, :] 
        
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x_speed, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = x_speed
        
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
            
            # WaveNet 核心计算 (Filter + Gate)
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

        # Cross Attention
        x_transformed = F.relu(x)
        condition = self.cross_attn(query=x_transformed, key_value=st_context)
        condition = F.relu(condition) 

        # --- Diffusion (DDPM/DDIM) ---
        if y is None:
            track_grad = self.training
            # Map parameters to diffusion context
            # fm_steps -> DDIM steps
            output = self.flow.sample(
                condition,
                guidance_scale=guidance_scale,
                fm_steps=fm_steps, # Used to determine DDIM stride
                solver=solver,     # Ignored, using DDIM
                track_grad=track_grad
            )
            return output
        else:
            base_loss = self.flow.loss(y_true=y, cond=condition, drop_prob=drop_prob)
            total_loss = base_loss + adj_reg
            return total_loss