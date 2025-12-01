import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from basicsr.archs.arch_util import to_2tuple , trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
from basicsr.archs.pscan import pscan  


def index_reverse(index):
    """ index: (B, L) - paddle Tensor (int64) """
    # create index_r same shape as index, such that index_r[i, index[i,:]] = arange(L)
    B, L = index.shape
    index_r = paddle.zeros_like(index)
    ind = paddle.arange(0, L, dtype=index.dtype)  # on CPU by default
    B_idx = paddle.arange(0, B, dtype='int64').unsqueeze(1).tile([1, L])  # (B, L)
    scatter_idx = paddle.stack([B_idx, index], axis=-1)  # (B, L, 2)
    scatter_idx = scatter_idx.reshape([-1, 2]).astype('int64')  # (B*L, 2)
    updates = paddle.arange(0, L, dtype=index.dtype).tile([B])  # (B*L,)
    shape = [B, L]
    index_r = paddle.scatter_nd(scatter_idx, updates, shape)
    return index_r


def semantic_neighbor(x, index):

    if index.dtype != 'int64':
        index = index.astype('int64')

    B = x.shape[0]
    L = x.shape[1]
    C = x.shape[2] if x.ndim > 2 else 1

    # flatten batch维度，方便 gather
    x_flat = paddle.reshape(x, [B * L, C])  # [B*L, C]
    offset = paddle.arange(B, dtype='int64') * L  # [B]
    offset = paddle.reshape(offset, [B, 1])       # [B, 1]
    index_flat = index + offset                     # [B, L]
    index_flat = paddle.reshape(index_flat, [-1])  # [B*L]

    gathered_flat = paddle.gather(x_flat, index_flat)  # [B*L, C]
    gathered = paddle.reshape(gathered_flat, [B, L, C])
    return gathered


# ---------- modules ----------

class dwconv(nn.Layer):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2D(hidden_features, hidden_features, kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        # x: (B, N, C) where N = H*W
        B, N, C = x.shape
        H, W = x_size
        x = paddle.transpose(x, perm=[0, 2, 1])  # (B, C, N)
        x = paddle.reshape(x, [B, self.hidden_features, H, W])
        x = self.depthwise_conv(x)
        x = paddle.reshape(x, [B, self.hidden_features, -1])
        x = paddle.transpose(x, perm=[0, 2, 1])
        return x


class ConvFFN(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Gate(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # conv2d with groups=dim
        self.conv = nn.Conv2D(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)

    def forward(self, x, H, W):
        x1, x2 = paddle.chunk(x, 2, axis=-1)
        B, N, C = x.shape
        # norm on x2, then conv expects (B, C//2, H, W)
        x2_n = self.norm(x2)
        x2_t = paddle.transpose(x2_n, perm=[0, 2, 1])
        x2_r = paddle.reshape(x2_t, [B, C // 2, H, W])
        x2_conv = self.conv(x2_r)
        x2_flat = paddle.reshape(x2_conv, [B, C // 2, -1])
        x2_out = paddle.transpose(x2_flat, perm=[0, 2, 1])
        return x1 * x2_out


class GatedMLP(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.sg(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    # x: (b, h, w, c)
    b, h, w, c = x.shape
    x = paddle.reshape(x, [b, h // window_size, window_size, w // window_size, window_size, c])
    x = paddle.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = paddle.reshape(x, [-1, window_size, window_size, c])
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = paddle.reshape(windows, [b, h // window_size, w // window_size, window_size, window_size, -1])
    x = paddle.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = paddle.reshape(x, [b, h, w, -1])
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.create_parameter([size, num_heads], default_initializer=nn.initializer.Constant(0.0))

        self.proj = nn.Linear(dim, dim)
        # initialize relative_position_bias_table similar to trunc_normal_
        # you may want to use paddle.nn.initializer.TruncatedNormal if needed

        self.softmax = nn.Softmax(axis=-1)

        self.mymodule_q = nn.Identity()
        self.mymodule_k = nn.Identity()
        self.mymodule_v = nn.Identity()
        self.mymodule_a = nn.Identity()

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = paddle.reshape(qkv, [b_, n, 3, self.num_heads, c // self.num_heads])
        qkv = paddle.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        q = self.mymodule_q(q)
        k = self.mymodule_k(k)
        v = self.mymodule_v(v)
        attn = paddle.matmul(q, paddle.transpose(k, perm=[0,1,3,2]))

        # relative position bias: rpi is index tensor
        rp_flat = paddle.reshape(rpi, [-1]).astype('int64')
        table = self.relative_position_bias_table
        relative_position_bias = paddle.gather(table, rp_flat, axis=0)
        relative_position_bias = paddle.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1],
                                                                         self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = paddle.transpose(relative_position_bias, perm=[2,0,1])
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = paddle.reshape(attn, [b_ // nw, nw, self.num_heads, n, n]) + mask.unsqueeze(1).unsqueeze(0)
            attn = paddle.reshape(attn, [-1, self.num_heads, n, n])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.mymodule_a(attn)
        x = paddle.matmul(attn, v)
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
        x = paddle.reshape(x, [b_, n, c])
        x = self.proj(x)
        return x

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


# ---------- ASSM (核心结构) ----------
class ASSM(nn.Layer):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state

        # 1. 加回 selectiveScan 模块
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)

        # 2. 其余模块
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.Swish()
        self.out_proj = nn.Linear(hidden, dim, bias_attr=True)

        self.in_proj = nn.Sequential(
            nn.Conv2D(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2D(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.set_value(
            paddle.uniform(
                shape=self.embeddingB.weight.shape,
                min=-1 / self.num_tokens,
                max=1 / self.num_tokens
            )
        )

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(axis=-1)
        )

        # placeholders for fake-quant modules
        self.fq_embeddingB_w = nn.Identity()
        self.fq_token_w = nn.Identity()
        self.fq_cls_policy_a = nn.Identity()
        self.fq_full_embedding_a = nn.Identity()

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = paddle.matmul(self.fq_embeddingB_w(self.embeddingB.weight), self.fq_token_w(token.weight))
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, axis=-1)

        prompt = paddle.matmul(self.fq_cls_policy_a(cls_policy), self.fq_full_embedding_a(full_embedding)).reshape([B, n, self.d_state])
        detached_index = paddle.argmax(cls_policy, axis=-1).reshape([B, n])
        x_sort_indices = paddle.argsort(detached_index, axis=-1)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = paddle.transpose(x, perm=[0, 2, 1])
        x = paddle.reshape(x, [B, C, H, W])
        x = self.in_proj(x)
        x = x * paddle.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = paddle.reshape(x, [B, cc, -1])
        x = paddle.transpose(x, perm=[0, 2, 1])

        semantic_x = semantic_neighbor(x, x_sort_indices)

        # 3. 用 selectiveScan 处理
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x

def check_bit(tensor, name):
    """一个简单的辅助函数，用来检查 Paddle 张量是否量化，并打印状态。"""
    if not paddle.is_tensor(tensor) or tensor.numel() == 0:
        print(f"--- 检查 '{name}': [非张量或空张量，跳过检查] ---")
        return
    # 获取唯一值
    unique_vals = paddle.unique(tensor)
    num_unique = unique_vals.shape[0]
    # 转 numpy 方便打印
    unique_vals_np = np.round(unique_vals.numpy(), 4)
    first_20_vals_np = np.round(paddle.flatten(tensor)[:20].numpy(), 4)
    print(
        f"--- 检查张量 (Checking Tensor): '{name}' ---\n"
        f"  - 形状 (Shape): {tensor.shape}\n"
        f"  - 数据类型 (dtype): {tensor.dtype}\n"
        f"  - 前20个值 (First 20 Vals): {first_20_vals_np}\n"
        f"  - 唯一值数量 (Num Unique): {num_unique}\n"
        f"  - 唯一值 (Unique Vals): {unique_vals_np}\n"
        f"  - 是否为4-bit? (Is 4-bit?): {'✅ 是 (Yes)' if num_unique <= 16 else '❌ 否 (No)'}\n"
        f"-------------------------------------------"
    )

class Selective_Scan(nn.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x_proj
        x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias_attr=False)
        transposed_x_weight = x_proj.weight.transpose([1, 0])
        self.x_proj_weight = self.create_parameter(
            shape=[1] + list(transposed_x_weight.shape),
            default_initializer=nn.initializer.Assign(transposed_x_weight)
        )
        del x_proj

        # dt_projs
        dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        transposed_dt_weight = dt_proj.weight.transpose([1, 0])
        self.dt_projs_weight = self.create_parameter(
            shape=[1] + list(transposed_dt_weight.shape),
            default_initializer=nn.initializer.Assign(transposed_dt_weight)
        )
        self.dt_projs_bias = self.create_parameter(
            shape=[1] + list(dt_proj.bias.shape),
            default_initializer=nn.initializer.Assign(dt_proj.bias)
        )
        del dt_proj

        # A_logs 和 Ds 的修改部分
        A_logs_tensor = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.A_logs = self.create_parameter(shape=A_logs_tensor.shape, default_initializer=nn.initializer.Assign(A_logs_tensor))

        Ds_tensor = self.D_init(self.d_inner, copies=1, merge=True)
        self.Ds = self.create_parameter(shape=Ds_tensor.shape, default_initializer=nn.initializer.Assign(Ds_tensor))
        
        self.selective_scan = None  # placeholder
        self.pscan = None  # placeholder

        # Identity placeholders
        self.dt_weight = nn.Identity()
        self.x_weight = nn.Identity()
        self.mymodule_xs = nn.Identity()
        self.mymodule_dts = nn.Identity()
        self.mymodule_delta = nn.Identity()
        self.mymodule_Bs = nn.Identity()
        self.mymodule_Cs = nn.Identity()
        self.mymodule_As = nn.Identity()
        self.mymodule_dt_projs_bias = nn.Identity()
        self.mymodule_hs = nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias_attr=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.initializer.Constant(dt_init_std)(dt_proj.weight)
        elif dt_init == "random":
            nn.initializer.Uniform(-dt_init_std, dt_init_std)(dt_proj.weight)
        else:
            raise NotImplementedError

        dt = paddle.exp(
            paddle.rand([d_inner]) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clip(min=dt_init_floor)

        # inverse softplus
        inv_dt = dt + paddle.log(-paddle.expm1(-dt))
        dt_proj.bias.set_value(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True):
        A = paddle.arange(1, d_state + 1, dtype='float32').unsqueeze(0).expand([d_inner, d_state])
        A_log = paddle.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).expand([copies, -1, -1])
            if merge:
                A_log = A_log.reshape([-1, d_state])
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True):
        D = paddle.ones([d_inner], dtype='float32')
        if copies > 1:
            D = D.unsqueeze(0).expand([copies, -1])
            if merge:
                D = D.reshape([-1])
        return D

    def forward_core(self, x: paddle.Tensor, prompt):
        B, L, C = x.shape
        K = 1
        xs = x.transpose([0, 2, 1]).reshape([B, 1, C, L])
        xs = self.mymodule_xs(xs.reshape([B, K, -1, L]))
        # check_bit(xs, "forward_core: xs")
        x_dbl = paddle.einsum("b k d l, k c d -> b k c l", xs, self.x_weight((self.x_proj_weight).unsqueeze(0)))
        dts, Bs, Cs = paddle.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=2)
        dts = paddle.einsum("b k r l, k d r -> b k d l", self.mymodule_dts(dts.reshape([B, K, -1, L])), 
                            self.dt_weight((self.dt_projs_weight.unsqueeze(0))))
        # check_bit(self.mymodule_dts(dts.view(B, K, -1, L)), "forward_core: quant_dts_input") 
        # check_bit(self.dt_weight((self.dt_projs_weight.unsqueeze(0))), "forward_core: quant_dt (dt_projs_weight)")
        xs = xs.astype('float32').reshape([B, -1, L])
        dts = dts.reshape([B, -1, L])
        Bs = Bs.reshape([B, K, -1, L])
        Cs = Cs.reshape([B, K, -1, L]) + prompt
        Ds = self.Ds.astype('float32').reshape([-1])
        As = -paddle.exp(self.A_logs.astype('float32')).reshape([-1, self.d_state])
        dt_projs_bias = self.dt_projs_bias.astype('float32').reshape([-1])

        out_y = self.selective_scan_ref(
            xs, self.mymodule_delta(dts),
            self.mymodule_As(As), self.mymodule_Bs(Bs), self.mymodule_Cs(Cs), Ds, z=None,
            delta_bias=self.mymodule_dt_projs_bias(dt_projs_bias),
            delta_softplus=True,
            return_last_state=False,
        ).reshape([B, K, -1, L])
        return out_y[:, 0]

    def forward(self, x: paddle.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.transpose([0, 2, 1]).reshape([b, 1, c, l])
        y = self.forward_core(x, prompt)
        y = y.transpose([0, 2, 1])
        return y
    

    def selective_scan_ref(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                        return_last_state=False):
        dtype_in = u.dtype
        u = u.astype('float32')
        delta = delta.astype('float32')
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].astype('float32')
        if delta_softplus:
            delta = F.softplus(delta)
        batch = u.shape[0]
        dim = A.shape[0]
        dstate = A.shape[1]
        is_variable_B = len(B.shape) >= 3
        is_variable_C = len(C.shape) >= 3
        B = B.astype('float32')
        C = C.astype('float32')
        # deltaA = exp(delta * A)
        to_exp = paddle.einsum('bdl,dn->bdln', delta, A)
        deltaA = paddle.exp(to_exp)
        # deltaB_u computation
        if not is_variable_B:
            deltaB_u = paddle.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if len(B.shape) == 3:
                deltaB_u = paddle.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                # repeat B
                H = dim // B.shape[1]
                repeat_pattern = [1, H, 1, 1]
                B = paddle.tile(B, repeat_pattern)
                deltaB_u = paddle.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

        if is_variable_C and len(C.shape) == 4:
            H = dim // C.shape[1]
            repeat_pattern = [1, H, 1, 1]  # 沿 G 维重复 H 次
            C = paddle.tile(C, repeat_pattern)

        last_state = None
        # call pscan
        hs = pscan(deltaA.transpose([0, 2, 1, 3]).contiguous(), deltaB_u.transpose([0, 2, 1, 3]).contiguous())
        hs = hs.transpose([0, 2, 1, 3]).contiguous()
        hs = self.mymodule_hs(hs)
        # check_bit(hs, "forward_core: quant_hs")
        y = paddle.einsum('bhln,bhnl->bhl', hs, C)
        out = y if D is None else y + u * D.reshape([-1, 1])
        if z is not None:
            out = out * F.silu(z)
        out = out.astype(dtype_in)
        if return_last_state:
            return out, last_state
        else:
            return out


class AttentiveLayer(nn.Layer):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_tokens = num_tokens
        self.is_last = is_last
        self.inner_rank = inner_rank

        self.softmax = nn.Softmax(axis=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = self.create_parameter(
            shape=[dim],
            default_initializer=nn.initializer.Constant(layer_scale)
        )
        self.scale2 = self.create_parameter(
            shape=[dim],
            default_initializer=nn.initializer.Constant(layer_scale)
        )

        self.wqkv = nn.Linear(dim, 3 * dim, bias_attr=qkv_bias)

        self.win_mhsa = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = ASSM(
            dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn1 = GatedMLP(dim, mlp_hidden_dim, dim)
        self.convffn2 = GatedMLP(dim, mlp_hidden_dim, dim)

        self.embeddingA = nn.Embedding(inner_rank, d_state)
        nn.initializer.Uniform(-1 / inner_rank, 1 / inner_rank)(self.embeddingA.weight)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x) 
        qkv = qkv.reshape([b, h, w, c3])

        if self.shift_size > 0:
            shifted_qkv = paddle.roll(qkv, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None

        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, c3])
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, c])

        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x.reshape([b, n, c]) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x
        return x


class BasicBlock(nn.Layer):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.LayerList([
            AttentiveLayer(
                dim=dim,
                d_state=d_state,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                is_last=i == depth - 1,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class ASSB(nn.Layer):
    """
    Attentive State Space Block (Residual Group)
    """

    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # patch embed/unembed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None
        )

        # residual group
        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        # residual connection
        if resi_connection == '1conv':
            self.conv = nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2D(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

    def forward(self, x, x_size, params):
        # residual group forward
        out = self.residual_group(x, x_size, params)
        # unembed -> conv -> embed
        out = self.patch_unembed(out, x_size)
        out = self.conv(out)
        out = self.patch_embed(out)
        # residual connection
        return out + x


class PatchEmbed(nn.Layer):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.flatten(2).transpose([0, 2, 1])  # -> (B, H*W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Layer):
    """
    Patch Unembedding (Token to Image)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # x: (B, H*W, C)
        # x_size: (H, W)
        x = x.transpose([0, 2, 1])
        x = x.reshape([x.shape[0], self.embed_dim, x_size[0], x_size[1]])
        return x

    def flops(self, input_resolution=None):
        return 0


class Upsample(nn.Sequential):
    """Upsample module (for 2^n and 3x upsampling).

    Args:
        scale (int): Upsampling factor. Supported: 2^n and 3.
        num_feat (int): Number of input feature channels.
    """

    def __init__(self, scale, num_feat):
        self.scale = scale
        self.num_feat = num_feat

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale {scale}. Only 2^n and 3 are supported.")

        super().__init__(*m)

    def flops(self, input_resolution):
        """FLOPs estimation (approximate)."""
        h, w = input_resolution
        flops = 0
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * h * w * int(math.log2(self.scale))
        elif self.scale == 3:
            flops += self.num_feat * 9 * self.num_feat * 9 * h * w
        return flops


class UpsampleOneStep(nn.Sequential):
    """Single-step upsampling (1 conv + 1 pixelshuffle).
    
    Used for lightweight SR models to save parameters.

    Args:
        scale (int): Upsampling factor. Supported: 2^n or 3.
        num_feat (int): Number of input feature channels.
        num_out_ch (int): Output channels (e.g., RGB = 3).
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.scale = scale
        self.input_resolution = input_resolution

        m = [
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        ]
        super().__init__(*m)

    def flops(self, input_resolution):
        """Approximate FLOPs estimation."""
        h, w = self.input_resolution if input_resolution is None else input_resolution
        # conv: h*w*Cin*Cout*k*k ≈ h*w*num_feat*(scale^2)*9
        flops = h * w * self.num_feat * (self.scale ** 2) * 9
        return flops


@ARCH_REGISTRY.register()
class MambaIRv2Light(paddle.nn.Layer):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 d_state=8,
                 depths=(6, 6, 6, 6,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            # register as buffer so it moves with model
            self.register_buffer('mean', paddle.to_tensor(rgb_mean, dtype='float32').reshape([1, 3, 1, 1]))
        else:
            self.register_buffer('mean', paddle.zeros([1, 1, 1, 1], dtype='float32'))

        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2D(num_in_ch, embed_dim, kernel_size=3, stride=1, padding=1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # create parameter and initialize with truncated normal
            self.absolute_pos_embed = self.create_parameter(
                shape=[1, num_patches, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02)
            )
        else:
            self.absolute_pos_embed = None

        # relative position index (SW-MSA)
        relative_position_index_SA = self.calculate_rpi_sa()
        # register as buffer so it's saved and moved with model
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # build layers (ASSB blocks in Paddle)
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = ASSB(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2D(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2D(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2D(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2D(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2D(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2))
            self.conv_up1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2D(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2D(embed_dim, num_out_ch, 3, 1, 1)

        # weight init: mimic your _init_weights behavior
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # only apply to nn.Linear and LayerNorm like original
        if isinstance(m, nn.Linear):
            # Paddle's TruncatedNormal initializer usage during create_parameter, but for existing layers:
            nn.initializer.TruncatedNormal(std=0.02)(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.set_value(paddle.zeros_like(m.bias))
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, 'bias'):
                m.bias.set_value(paddle.zeros_like(m.bias))
            if hasattr(m, 'weight'):
                m.weight.set_value(paddle.ones_like(m.weight))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape and (self.absolute_pos_embed is not None):
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA (Paddle version)
        coords_h = paddle.arange(self.window_size, dtype='int64')
        coords_w = paddle.arange(self.window_size, dtype='int64')
        # meshgrid: note order, we want shape (2, Wh, Ww)
        coords = paddle.stack(paddle.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = coords.reshape([2, -1])  # 2, Wh*Ww
        # relative coords: 2, Wh*Ww, Wh*Ww
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(axis=-1).astype('int64')  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA (Paddle)
        h, w = x_size
        # img_mask shape (1, h, w, 1)
        img_mask = paddle.zeros([1, h, w, 1], dtype='int64')
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask.astype('float32'), self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.reshape([-1, self.window_size * self.window_size])
        # attn_mask shape: (num_windows, 1, Wh*Ww) - (num_windows, Wh*Ww, 1) -> (num_windows, Wh*Ww, Wh*Ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = paddle.where(attn_mask != 0, paddle.full_like(attn_mask, -100.0), paddle.full_like(attn_mask, 0.0))
        return attn_mask

    def forward(self, x):
        # padding
        h_ori, w_ori = x.shape[-2], x.shape[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        # pad by reflecting (paddle.flip)
        x = paddle.concat([x, paddle.flip(x, axis=[2])], axis=2)[:, :, :h, :]
        x = paddle.concat([x, paddle.flip(x, axis=[3])], axis=3)[:, :, :, :w]

        # ensure mean has same dtype and place as x
        mean = self.mean.astype(x.dtype)
        # if mean is single-channel but input has 3 channels, broadcasting will handle
        x = (x - mean) * self.img_range

        attn_mask = self.calculate_mask([h, w])
        # ensure mask on same device/dtype - in Paddle register_buffer is used so it should already be on model device
        attn_mask = attn_mask.astype(x.dtype)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x


# Example usage (Paddle):
if __name__ == '__main__':
    paddle.set_device('cpu')  # or 'gpu' if available
    model = MambaIRv2Light(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect')
    # params count
    total = sum([p.size.numel() for p in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    trainable_num = sum([p.size.numel() for p in model.parameters() if p.trainable])
    print(trainable_num)

    # test
    _input = paddle.randn([2, 3, 64, 64], dtype='float32')
    output = model(_input)
    print(output.shape)

