import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Wrappers for Hierarchy and ONNX Parity
# These ensure that the state dict keys match the production ONNX model's 
# hierarchy (e.g., <name>.linear.weight) while using standard PyTorch layers.
# -----------------------------------------------------------------------------

class Linear(nn.Module):
    """Matches hierarchy: <name>.linear.weight / bias"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class Conv1d(nn.Module):
    """Matches hierarchy: <name>.net.weight / bias or <name>.weight / bias"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, padding: int = 0, groups: int = 1, dilation: int = 1, wrap_net: bool = True):
        super().__init__()
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, padding=padding, groups=groups, dilation=dilation)
        if wrap_net:
            self.net = conv
        else:
            self.weight = conv.weight
            if bias:
                self.bias = conv.bias
            self.conv = conv

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.net(x) if hasattr(self, "net") else self.conv(x)
        # Optional mask multiply lives in THIS module's scope, so the export
        # emits e.g. `proj_in/Mul` / `proj_out/Mul` (matches checks/vf.txt).
        if mask is not None:
            y = y * mask
        return y

class LayerNorm(nn.Module):
    """
    1D LayerNorm [B, C, L] -> [B, L, C] -> [B, C, L]
    Matches hierarchy: norm.norm.weight / norm.norm.bias
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Hierarchy: <scope>.norm.weight/bias
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ONNX Flow: Transpose(1,2) -> LayerNormalization -> Transpose(1,2)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)



# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding under its own `sinusoidal` scope.

    Owning module is named `sinusoidal` so the export emits
    /time_encoder/sinusoidal/{Reshape, Unsqueeze, Mul, Mul_1, Sin, Cos, Concat}
    exactly like the production trace (checks/vf.txt). `inv_freq` is a
    deterministic non-persistent buffer (production folds it into the constant
    /time_encoder/sinusoidal/Constant_3_output_0, so it never appears in
    checkpoints either).
    """
    def __init__(self, half_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim).float() / (half_dim - 1)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: normalized time [2*B, 1, 1] -> /sinusoidal/Reshape -> /sinusoidal/Unsqueeze
        x = x.view(-1).unsqueeze(1)
        # Aligns with /sinusoidal/Mul (1000.0) and /sinusoidal/Mul_1 (inv_freq)
        x = x * 1000.0
        emb = x * self.inv_freq.unsqueeze(0)
        # Concat sin and cos matching /sinusoidal/Concat
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimeEncoder(nn.Module):
    """
    Encodes normalized time into embeddings.
    Hierarchy matches /time_encoder/sinusoidal/ and /time_encoder/mlp/
    """
    def __init__(self, embed_dim: int, hdim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(embed_dim // 2)
        # nn.Sequential named `mlp` yields the /time_encoder/mlp/mlp.{0,1,2}/
        # double scope of the production trace while keeping state-dict keys
        # mlp.0.linear.* / mlp.2.linear.* unchanged.
        self.mlp = nn.Sequential(
            Linear(embed_dim, hdim),
            nn.Mish(),
            Linear(hdim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: normalized time [2*B, 1, 1]
        x = self.sinusoidal(x)
        x = self.mlp(x)
        # Aligns with /time_encoder/Unsqueeze [2*B, 64, 1]
        return x.unsqueeze(-1)

class TimeCondBlock(nn.Module):
    """
    Injects time embeddings into the latent flow.
    Hierarchy: main_blocks.X/linear/linear
    """
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.linear = Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [2*B, channels, L]
        # time_emb: [2*B, time_dim, 1]
        # Aligns with MatMul -> Add -> Transpose -> Add(x) sequence
        cond = self.linear(time_emb.transpose(1, 2)) # [2*B, 1, channels]
        x = x + cond.transpose(1, 2)
        # Mask multiply in this module's scope -> `main_blocks.<i>/Mul`.
        if mask is not None:
            x = x * mask
        return x

class DepthwiseConv1d(nn.Conv1d):
    """Depthwise Conv1d that replicate-pads INSIDE its own module scope.

    This makes the export emit `<scope>/dwconv/Pad` followed by
    `<scope>/dwconv/Conv` (both under the dwconv scope), matching the production
    trace (checks/vf.txt). Weight/bias remain `dwconv.weight` / `dwconv.bias`, so
    checkpoint keys and parameter order are unchanged.
    """
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1):
        super().__init__(dim, dim, kernel_size, groups=dim, dilation=dilation, padding=0)
        self.replicate_pad = ((kernel_size - 1) // 2) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches /dwconv/Pad -> /dwconv/Conv (ONNX uses edge/replicate padding).
        x = F.pad(x, (self.replicate_pad, self.replicate_pad), mode='replicate')
        return super().forward(x)


class ConvNeXtBlock1D(nn.Module):
    """
    1D ConvNeXt block with depthwise conv, layer norm, and pointwise projections.
    Hierarchy: main_blocks.X/convnext.Y/
    """
    def __init__(self, dim: int, kernel_size: int = 5, expansion: int = 4, dilation: int = 1):
        super().__init__()
        # dwconv: Matches /dwconv/Pad -> /dwconv/Conv scope (pads internally).
        self.dwconv = DepthwiseConv1d(dim, kernel_size=kernel_size, dilation=dilation)
        
        # norm: Matches /norm/norm/LayerNormalization scope
        self.norm = LayerNorm(dim)
        
        # pwconv1: Pointwise expansion
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        # act: Matches /act/ scope with GELU logic
        self.act = nn.GELU() 
        # pwconv2: Pointwise reduction
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        
        # gamma: Matches /gamma scaling parameter
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 1e-6)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None: x = x * mask
        res = x
        
        # Depthwise conv (pads internally -> /dwconv/Pad -> /dwconv/Conv).
        x = self.dwconv(x)
        if mask is not None: x = x * mask
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        
        x = x + res
        if mask is not None: x = x * mask
        return x

class ConvNeXtStack(nn.Module):
    """
    A stack of ConvNeXt blocks. 
    Hierarchy: main_blocks.{0,2,4,6...}/convnext.{0,1,2...}
    """
    def __init__(self, dim: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        # Blocks are named convnext.0, convnext.1, etc.
        self.convnext = nn.ModuleList([
            ConvNeXtBlock1D(dim, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.convnext:
            x = block(x, mask)
        return x

# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, half: int) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE).
    Matches ONNX flow: x1*cos - x2*sin and x1*sin + x2*cos.
    `half` (= head_dim // 2) is a Python int so the export emits STATIC slice
    bounds ([0:32] / [32:64]) exactly like the production trace, instead of a
    Shape -> Gather -> Div -> Cast chain per block (checks/vf.txt Mul_8/Mul_9).
    """
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return torch.cat([x1_rot, x2_rot], dim=-1)



class webgpu_shape_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.tensor(x.shape, device=x.device, dtype=torch.long)
    @staticmethod
    def backward(ctx, grad_output):
        return None
    @staticmethod
    def symbolic(g, x):
        return g.op("webgpu_shape", x)

class webgpu_slice_shape_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape):
        return shape[:2]
    @staticmethod
    def backward(ctx, grad_output):
        return None
    @staticmethod
    def symbolic(g, shape):
        return g.op("webgpu_slice_shape", shape)

class webgpu_concat_shape_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape12, shape01):
        return torch.cat([shape01, shape12])
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    @staticmethod
    def symbolic(g, shape12, shape01):
        # Target ONNX node has 2 inputs: dynamic [B*2, T] slice first, then the
        # baked [heads, -1] constant (webgpu_head_shape). See checks/vf.txt.
        return g.op("webgpu_concat_shape", shape01, shape12)

class webgpu_reshape_heads_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(torch.tensor(x.shape, device=x.device))
        return x.view(shape.tolist())
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return grad_output.reshape(orig_shape.tolist()), None
    @staticmethod
    def symbolic(g, x, shape):
        return g.op("webgpu_reshape_heads", x, shape)

class webgpu_transpose_heads_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.permute(2, 0, 1, 3)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.permute(1, 2, 0, 3)
    @staticmethod
    def symbolic(g, x):
        return g.op("webgpu_transpose_heads", x)

class WebGPUSplit(nn.Module):
    """
    Splits heads using view + permute.
    In the ONNX trace, this is represented by the /Split/, /Split_1/ and /Split_2/
    scopes for RoPE blocks. The webgpu_* autograd functions are applied directly
    (not via wrapper sub-modules) so each op is named `<Split>/webgpu_*` with a
    single scope level, exactly matching checks/vf.txt. The baked [heads, -1]
    reshape constant is held here as `webgpu_head_shape` (kept as a non-trained
    Parameter so the optimizer parameter list is unchanged).
    """
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads
        self.webgpu_head_shape = nn.Parameter(torch.tensor([heads, -1], dtype=torch.long), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = webgpu_shape_fn.apply(x)
        shape01 = webgpu_slice_shape_fn.apply(shape)
        shape4 = webgpu_concat_shape_fn.apply(self.webgpu_head_shape, shape01)
        x = webgpu_reshape_heads_fn.apply(x, shape4)
        return webgpu_transpose_heads_fn.apply(x)

class webgpu_slice_shape_merge_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape):
        return shape[1:3]
    @staticmethod
    def backward(ctx, grad_output):
        return None
    @staticmethod
    def symbolic(g, shape):
        return g.op("webgpu_slice_shape", shape)

class webgpu_concat_shape_merge_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape12, width):
        return torch.cat([shape12, width])
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    @staticmethod
    def symbolic(g, shape12, width):
        # Target ONNX node has 2 inputs: dynamic [B*2, T] slice first, then the
        # baked [heads*head_dim] constant (webgpu_width). See checks/vf.txt.
        return g.op("webgpu_concat_shape", shape12, width)

class webgpu_transpose_merge_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.permute(1, 2, 0, 3).contiguous()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.permute(2, 0, 1, 3)
    @staticmethod
    def symbolic(g, x):
        return g.op("webgpu_transpose_merge", x)

class webgpu_reshape_merge_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(torch.tensor(x.shape, device=x.device))
        return x.view(shape.tolist())
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return grad_output.reshape(orig_shape.tolist()), None
    @staticmethod
    def symbolic(g, x, shape):
        return g.op("webgpu_reshape_merge", x, shape)

class WebGPUMerge(nn.Module):
    """
    Merges heads back using transpose + reshape.
    In the ONNX trace, this is the /Split_3/ scope. The webgpu_* autograd
    functions are applied directly (not via wrapper sub-modules) so each op is
    named `<Split_3>/webgpu_*` with a single scope level, matching checks/vf.txt.
    The baked [heads*head_dim] reshape constant is held here as `webgpu_width`
    (a non-trained Parameter so the optimizer parameter list is unchanged).
    """
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.webgpu_width = nn.Parameter(torch.tensor([heads * head_dim], dtype=torch.long), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [H, B, T, D]
        shape = webgpu_shape_fn.apply(x)
        shape12 = webgpu_slice_shape_merge_fn.apply(shape)
        shape3 = webgpu_concat_shape_merge_fn.apply(shape12, self.webgpu_width)
        x = webgpu_transpose_merge_fn.apply(x)
        return webgpu_reshape_merge_fn.apply(x, shape3)


class StaticSqueeze(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        ctx.save_for_backward(torch.tensor(x.shape, device=x.device))
        return x.squeeze(dim)
    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return grad_output.reshape(orig_shape.tolist()), None
    @staticmethod
    def symbolic(g, x, dim):
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        return g.op("Squeeze", x, axes)

class StaticTile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, repeats):
        ctx.repeats = repeats
        ctx.in_shape = x.shape
        return x.tile(repeats)
    @staticmethod
    def backward(ctx, grad_output):
        repeats, in_shape = ctx.repeats, ctx.in_shape
        # Fold each tiled dim into (repeat, orig) and sum over the repeat axis,
        # so grad keeps the ORIGINAL size (not collapsed to 1).
        folded = []
        sum_dims = []
        for dim, (r, s) in enumerate(zip(repeats, in_shape)):
            if r > 1:
                sum_dims.append(len(folded))
                folded.extend([r, s])
            else:
                folded.append(s)
        if sum_dims:
            grad_output = grad_output.reshape(folded).sum(dim=sum_dims)
        return grad_output, None
    @staticmethod
    def symbolic(g, x, repeats):
        repeats_t = g.op("Constant", value_t=torch.tensor(repeats, dtype=torch.long))
        return g.op("Tile", x, repeats_t)

class DynamicBatchTile(torch.autograd.Function):
    """Tile `x` along the batch axis to match `ref`'s batch dim, emitting a
    DYNAMIC repeat (Shape -> Gather -> Concat([B,1,1]) -> Tile) so the exported
    graph supports any batch size — exactly the production `/Tile` node that
    expands the baked style-key constant [1, 50, 256] to [B, 50, 256]
    (see checks/vf.txt). A static `.repeat(int, ...)` would instead bake B and
    break dynamic-batch export.
    """
    @staticmethod
    def forward(ctx, x, ref):
        ctx.batch = int(ref.shape[0])
        return x.repeat(ctx.batch, 1, 1)
    @staticmethod
    def backward(ctx, grad_output):
        # x was broadcast over the batch dim; sum grads back to a single row.
        return grad_output.sum(dim=0, keepdim=True), None
    @staticmethod
    def symbolic(g, x, ref):
        shape = g.op("Shape", ref)
        gather_idx = g.op("Constant", value_t=torch.tensor(0, dtype=torch.long))
        b = g.op("Gather", shape, gather_idx, axis_i=0)
        unsqueeze_axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        b1 = g.op("Unsqueeze", b, unsqueeze_axes)
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        repeats = g.op("Concat", b1, one, one, axis_i=0)
        return g.op("Tile", x, repeats)

class AttentionModule(nn.Module):
    """
    Cross-attention module with optional RoPE support.
    Hierarchy: matches /attn/ or /attention/ scopes
    """
    def __init__(self, d_model: int, d_ctx: int, heads: int, head_dim: int, 
                 use_rope: bool, rope_gamma: float = 10.0, rotary_base: float = 10000.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        # ONNX divides logits by sqrt(d_ctx) (=16 for d_ctx=256), not sqrt(head_dim).
        # Kept as a division (Div node) to match the production trace exactly.
        self.scale_denom = d_ctx ** 0.5
        
        # Wrapped projections match hierarchy: W_query.linear.weight etc.
        self.W_query = Linear(d_model, heads * head_dim)
        self.W_key = Linear(d_ctx, heads * head_dim)
        self.W_value = Linear(d_ctx, heads * head_dim)
        self.out_fc = Linear(heads * head_dim, d_model)
        
        if use_rope:
            half_dim = head_dim // 2
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, half_dim).float() / half_dim))
            # Buffers match /theta and /increments hierarchy
            self.register_buffer("theta", (inv_freq * rope_gamma).view(1, 1, -1))
            self.register_buffer("increments", torch.arange(1000).view(1, 1000, 1))
            
            # Match /Split, /Split_1, /Split_2 (head split) and /Split_3 (head merge) scopes in ONNX for RoPE blocks
            self.Split = WebGPUSplit(heads)
            self.Split_1 = WebGPUSplit(heads)
            self.Split_2 = WebGPUSplit(heads)
            self.Split_3 = WebGPUMerge(heads, head_dim)
        else:
            # Style attention uses Tanh activation on projected keys
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                ctx_mask: Optional[torch.Tensor] = None, 
                ctx_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x arrives PRE-TRANSPOSED as [B, T, C] (the [B,C,T]->[B,T,C] transpose
        # lives in the CrossAttentionBlock scope, matching the production trace
        # `main_blocks.<i>/Transpose`). Output is [B, T, H*D] (block transposes back).
        H, D = self.heads, self.head_dim
        
        # 1. Projections
        q = self.W_query(x) # [B, T, H*D]
        k = self.W_key(ctx_keys if ctx_keys is not None else ctx) # [B, L, H*D]
        v = self.W_value(ctx) # [B, L, H*D]
        
        # 2. Split heads and transpose
        if self.use_rope:
            # Matches /Split/, /Split_1/ and /Split_2/ webgpu_reshape_heads scopes in ONNX
            q = self.Split(q) # [H, B, T, D]
            k = self.Split_1(k) # [H, B, L, D]
            v = self.Split_2(v) # [H, B, L, D]
        else:
            # Style attention uses split + stack (matches ONNX Split + Unsqueeze + Concat)
            q_split = torch.split(q, q.size(-1) // H, dim=-1)
            k_split = torch.split(k, k.size(-1) // H, dim=-1)
            v_split = torch.split(v, v.size(-1) // H, dim=-1)
            q = torch.stack(q_split, dim=0) # [H, B, T, D]
            k = torch.stack(k_split, dim=0) # [H, B, L, D]
            v = torch.stack(v_split, dim=0) # [H, B, L, D]
        
        # 3. Positional Encoding / Activation
        if not self.use_rope:
            # Aligns with /tanh/Tanh in ONNX style attention applied AFTER transpose
            k = k.transpose(-1, -2) # [H, B, D, L]
            k = self.tanh(k)
            # Dot-product attention (Div scale -> `attention/Div`)
            logits = torch.matmul(q, k) / self.scale_denom
        else:
            # Aligns with RoPE sequence: ReduceSum -> Div -> Mul(theta) -> Sin/Cos.
            # Sequence lengths come from the MASK shapes (production `attn/Shape` /
            # `attn/Shape_1` read the transposed masks) so the whole RoPE angle
            # chain is identical across blocks and dedupes like checks/vf.txt.
            len_q = x_mask.sum(dim=(1, 2)).view(-1, 1, 1) if x_mask is not None else torch.tensor([float(x.shape[1])], device=x.device, dtype=x.dtype).view(1, 1, 1)
            len_k = ctx_mask.sum(dim=(1, 2)).view(-1, 1, 1) if ctx_mask is not None else torch.tensor([float(ctx.shape[1])], device=x.device, dtype=x.dtype).view(1, 1, 1)
            
            T_pos = x_mask.shape[1] if x_mask is not None else x.shape[1]
            L_pos = ctx_mask.shape[1] if ctx_mask is not None else ctx.shape[1]
            pos_q = self.increments[:, :T_pos, :].to(x.dtype)
            pos_k = self.increments[:, :L_pos, :].to(x.dtype)
            
            # Angles [2*B, T, D/2]
            arg_q = (pos_q / len_q) * self.theta
            arg_k = (pos_k / len_k) * self.theta
            
            # No unsqueeze needed; native broadcasting against [H, 2*B, T, D/2]
            cos_q, sin_q = arg_q.cos(), arg_q.sin()
            cos_k, sin_k = arg_k.cos(), arg_k.sin()
            
            # Static half bound (32) -> constant-bound Slices, as in the trace.
            q = apply_rotary_pos_emb(q, cos_q, sin_q, D // 2)
            k = apply_rotary_pos_emb(k, cos_k, sin_k, D // 2)
            
            # Dot-product attention (Div scale -> `attn/Div_4`)
            logits = torch.matmul(q, k.transpose(-1, -2)) / self.scale_denom
        
        # 5. Masking
        if ctx_mask is not None:
            # Aligns with /attn/Where masking padding in context
            logits = logits.masked_fill(ctx_mask.unsqueeze(0).transpose(-1, -2) == 0.0, float('-inf'))
        
        attn = F.softmax(logits, dim=-1)
        
        if x_mask is not None:
            # Masking query positions
            attn = attn.masked_fill(x_mask.unsqueeze(0) == 0.0, 0.0)
            
        # 6. Aggregation and Output Projection
        out = torch.matmul(attn, v) # [H, B, T, D]
        if self.use_rope:
            # Matches /Split_3/ webgpu_transpose_merge -> webgpu_reshape_merge scope
            out = self.Split_3(out) # [B, T, H*D]
        else:
            # Style attention merges via Split (split) -> Concat -> Squeeze (matches ONNX /Split_3, /Concat_3, /Squeeze)
            out_split = torch.split(out, out.size(0) // H, dim=0)
            out = StaticSqueeze.apply(torch.cat(out_split, dim=-1), 0) # [B, T, H*D]
        out = self.out_fc(out)
        
        # Query-mask multiply in the attn scope (`attn/Mul_14` / `attention/Mul`).
        if x_mask is not None:
            out = out * x_mask
            
        return out

class CrossAttentionBlock(nn.Module):
    """
    Standard residual block wrapping AttentionModule.
    Matches main_blocks hierarchy.
    """
    def __init__(self, d_model: int, d_ctx: int, heads: int, head_dim: int, use_rope: bool, rope_gamma: float = 10.0, rotary_base: float = 10000.0, transpose_ctx: bool = False):
        super().__init__()
        self.use_rope = use_rope
        self.transpose_ctx = transpose_ctx
        attn_mod = AttentionModule(d_model, d_ctx, heads, head_dim, use_rope, rope_gamma, rotary_base)
        if use_rope:
            # Hierarchy: main_blocks.{3,9,15,21}.attn
            self.attn = attn_mod 
        else:
            # Hierarchy: main_blocks.{5,11,17,23}.attention
            self.attention = attn_mod 
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, 
                x_mask: Optional[torch.Tensor] = None, 
                ctx_mask: Optional[torch.Tensor] = None, 
                ctx_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_mask is not None: x = x * x_mask
        res = x
        
        # Mask transposes ([B,1,T] -> [B,T,1]) in the block scope. Identical
        # across blocks, so the simplifier dedupes them into main_blocks.3's
        # Transpose_2 / Transpose_3 exactly like the production trace.
        attn_x_mask = x_mask.transpose(1, 2) if x_mask is not None else None
        attn_ctx_mask = ctx_mask.transpose(1, 2) if ctx_mask is not None else None
        
        if self.transpose_ctx:
            ctx = ctx.transpose(1, 2)
            if ctx_keys is not None:
                ctx_keys = ctx_keys.transpose(1, 2)
        
        # Query transpose [B,C,T] -> [B,T,C] in the BLOCK scope
        # (`main_blocks.<i>/Transpose` in the trace), not inside attn.
        x_t = x.transpose(1, 2)
        if self.use_rope:
            attn_out = self.attn(x_t, ctx, x_mask=attn_x_mask, ctx_mask=attn_ctx_mask, ctx_keys=ctx_keys)
        else:
            attn_out = self.attention(x_t, ctx, x_mask=attn_x_mask, ctx_mask=attn_ctx_mask, ctx_keys=ctx_keys)
        # Back to [B,C,T] in the block scope (`Transpose_4` / `Transpose_1`),
        # then residual Add with attn output as the FIRST operand (trace order).
        attn_out = attn_out.transpose(1, 2)
        
        x = attn_out + res
        x = self.norm(x)
        if x_mask is not None: x = x * x_mask
        return x

# -----------------------------------------------------------------------------
# Main Estimator
# -----------------------------------------------------------------------------

class UncondParams(nn.Module):
    """Learnable unconditional tokens for CFG. Dims from ttl.uncond_masker config."""
    def __init__(self, text_dim=256, n_style=50, style_value_dim=256, init_std=0.1):
        super().__init__()
        self.text_special_token = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.style_value_special_token = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)
        self.style_key_special_token = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)


class VectorFieldTrunk(nn.Module):
    """The vector-field network itself (ONNX scope `/vector_estimator/vector_field/`).

    Lives in its own module so the exported graph carries the `vector_field`
    scope level exactly like the production trace (checks/vf.txt). Its forward is
    the shared trunk used by BOTH the training (`return_velocity`) and ONNX
    inference paths, so node scope/names and the trained computation never drift:
        time_encoder -> proj_in -> main_blocks -> last_convnext -> proj_out
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        text_dim: int,
        style_dim: int,
        num_superblocks: int,
        time_embed_dim: int,
        rope_gamma: float,
        text_n_heads: int,
        time_hdim: int,
        rotary_base: float,
    ):
        super().__init__()
        # Initial Projection: Hierarchy matches proj_in.net.weight
        self.proj_in = Conv1d(in_channels, hidden_channels, 1, bias=False)

        # Time Encoder: Hierarchy matches /time_encoder/sinusoidal/ and /time_encoder/mlp/
        self.time_encoder = TimeEncoder(time_embed_dim, hdim=time_hdim)

        # main_blocks sequence matches trace: 0, 1, 2, 3, 4, 5 repeats num_superblocks times.
        self.main_blocks = nn.ModuleList()
        for i in range(num_superblocks):
            # Index 6*i + 0: ConvNeXt Stack (matches trace main_blocks.{0,6,12,18}/convnext.0..3)
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1, 2, 4, 8]))

            # Index 6*i + 1: Time Conditioning
            self.main_blocks.append(TimeCondBlock(time_embed_dim, hidden_channels))

            # Index 6*i + 2: ConvNeXt Stack
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1]))

            # Index 6*i + 3: Text Attention with RoPE (matches trace main_blocks.3/attn)
            self.main_blocks.append(CrossAttentionBlock(hidden_channels, text_dim, heads=text_n_heads, head_dim=64, use_rope=True, rope_gamma=rope_gamma, rotary_base=rotary_base, transpose_ctx=True))

            # Index 6*i + 4: ConvNeXt Stack
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1]))

            # Index 6*i + 5: Style Attention with Tanh (matches trace main_blocks.5/attention)
            self.main_blocks.append(CrossAttentionBlock(hidden_channels, style_dim, heads=2, head_dim=128, use_rope=False, rotary_base=rotary_base))

        # Final ConvNeXt Stack: Hierarchy matches last_convnext/convnext.0..3 (4 blocks, dilation 1)
        self.last_convnext = ConvNeXtStack(hidden_channels, kernel_size=5, dilations=[1, 1, 1, 1])

        # Final Projection: Hierarchy matches proj_out.net.weight
        self.proj_out = Conv1d(hidden_channels, out_channels, 1, bias=False)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        style_key: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> torch.Tensor:
        # Time embedding (ONNX scope /vector_field/time_encoder/...). Computed
        # here, inside the vector_field scope, exactly as in checks/vf.txt.
        t_emb = self.time_encoder(t_norm)

        # proj_in applies the mask in its own scope -> `proj_in/Mul`.
        x = self.proj_in(noisy_latent, latent_mask)

        for i, block in enumerate(self.main_blocks):
            idx = i % 6
            if idx in (0, 2, 4):
                x = block(x, mask=latent_mask)
            elif idx == 1:
                # TimeCondBlock applies the mask in its scope -> `main_blocks.<i>/Mul`.
                x = block(x, t_emb, mask=latent_mask)
            elif idx == 3:
                x = block(x, text_emb, x_mask=latent_mask, ctx_mask=text_mask)
            elif idx == 5:
                x = block(x, style_ttl, x_mask=latent_mask, ctx_keys=style_key)

        x = self.last_convnext(x, mask=latent_mask)
        # proj_out applies the mask in its own scope -> `proj_out/Mul`.
        return self.proj_out(x, latent_mask)


class VectorFieldEstimator(nn.Module):
    """
    Vector Field Estimator for Text-to-Latent generation.
    Follows production ONNX 1-to-1 computation trace.
    """
    def __init__(
        self,
        in_channels: int = 144,
        hidden_channels: int = 512,
        out_channels: int = 144,
        text_dim: int = 256,
        style_dim: int = 256,
        num_style_tokens: int = 50,
        num_superblocks: int = 4,
        time_embed_dim: int = 64,
        rope_gamma: float = 10.0,
        text_n_heads: int = 8,
        time_hdim: int = 256,
        rotary_base: float = 10000.0,
        cfg_scale: float = 3.0,
        uncond_init_std: float = 0.1,
    ):
        super().__init__()
        self.cfg_scale = cfg_scale
        
        # Matches hierarchy: tts.ttl
        self.tts = nn.Module()
        self.tts.ttl = nn.Module()
        
        # 1. Uncond Masker (Matches Expand/Where/Mul logic for CFG)
        self.tts.ttl.uncond_masker = UncondParams(
            text_dim,
            num_style_tokens,
            style_dim,
            init_std=uncond_init_std,
        )
        
        self.tile = nn.Parameter(torch.randn(1, num_style_tokens, style_dim) * 0.02)
        
        # 2. Vector Field. A dedicated module so its forward runs under the
        #    `/vector_estimator/vector_field/` ONNX scope, exactly matching the
        #    production trace (checks/vf.txt). Submodule names are unchanged so
        #    state_dict keys stay `tts.ttl.vector_field.{proj_in,time_encoder,
        #    main_blocks,last_convnext,proj_out}.*`.
        self.tts.ttl.vector_field = VectorFieldTrunk(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            style_dim=style_dim,
            num_superblocks=num_superblocks,
            time_embed_dim=time_embed_dim,
            rope_gamma=rope_gamma,
            text_n_heads=text_n_heads,
            time_hdim=time_hdim,
            rotary_base=rotary_base,
        )

    @staticmethod
    def remap_legacy_state_dict(state_dict: dict) -> dict:
        """Map the former flat vector-field layout into the ONNX-parity tree."""
        remapped = {}
        vector_prefixes = (
            "proj_in.", "time_encoder.", "main_blocks.", "last_convnext.", "proj_out.",
        )
        for key, value in state_dict.items():
            if key.startswith(vector_prefixes):
                key = f"tts.ttl.vector_field.{key}"
            remapped[key] = value
        return remapped

    def _forward_velocity(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
        drop_text: Optional[torch.Tensor] = None,
        drop_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training-only path (NEVER exported to ONNX): a single network
        evaluation returning the raw vector field v_theta. There is no CFG
        combination and no Euler integration here (the flow-matching target
        lives in z-space). Per-sample unconditional dropout substitutes the
        model's OWN learnable uncond tokens — the same tokens used for CFG at
        inference — so the train-time objective and inference-time guidance stay
        consistent.
        """
        B = noisy_latent.shape[0]
        t_norm = current_step / total_step

        # Conditional style key is the learned constant `self.tile`.
        style_key = StaticTile.apply(self.tile, (1, 1, 1))

        if drop_text is not None:
            dt = drop_text.view(B, 1, 1).to(text_emb.dtype)
            # Default unconditional text token is slot 0
            u_text = self.tts.ttl.uncond_masker.text_special_token[:, :, :1]
            text_uncond = u_text.expand(B, -1, text_emb.shape[-1])
            
            text_emb = text_emb * (1.0 - dt) + text_uncond * dt

        if drop_style is not None:
            ds = drop_style.view(B, 1, 1).to(style_ttl.dtype)
            
            u_style_val = self.tts.ttl.uncond_masker.style_value_special_token[0:1]
            style_uncond = u_style_val.expand(B, -1, -1)
            style_ttl = style_ttl * (1.0 - ds) + style_uncond * ds
            
            u_style_key = self.tts.ttl.uncond_masker.style_key_special_token[0:1]
            style_key_uncond = u_style_key.expand(B, -1, -1)
            style_key = style_key * (1.0 - ds) + style_key_uncond * ds

        return self.tts.ttl.vector_field(
            noisy_latent, text_emb, style_ttl, style_key,
            latent_mask, text_mask, t_norm.view(B, 1, 1),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: torch.Tensor,
        drop_text: Optional[torch.Tensor] = None,
        drop_style: Optional[torch.Tensor] = None,
        return_velocity: bool = False,
    ) -> torch.Tensor:
        # Training-only branch. `return_velocity` is a Python bool, so during
        # ONNX tracing (which always calls with the default False) this whole
        # branch is constant-folded away and never appears in the exported graph.
        if return_velocity:
            return self._forward_velocity(
                noisy_latent, text_emb, style_ttl, latent_mask, text_mask,
                current_step, total_step, drop_text, drop_style
            )

        # ====================================================================
        # Inference path — EXACT ONNX 1-to-1 trace. The outer scope handles CFG
        # batch expansion, uncond masking and the final CFG/Euler combination;
        # the network trunk runs inside `vector_field` (the /vector_field/ scope).
        # Do NOT reorder these ops; node ordering/names depend on this.
        # ====================================================================

        # Aligns with /Div -> /Reshape and /Reciprocal -> /Reshape_1. view(-1,1,1)
        # emits a single constant-shape Reshape exactly like the trace (a
        # view(B,1,1) would emit a Shape->Gather chain instead).
        t_norm = (current_step / total_step).view(-1, 1, 1)
        step_size = (1.0 / total_step).view(-1, 1, 1)

        # CFG Batch Expansion (Tile_1, Tile_3, Tile_4 in ONNX trace)
        noisy_latent_2 = StaticTile.apply(noisy_latent, (2, 1, 1))
        latent_mask_2 = StaticTile.apply(latent_mask, (2, 1, 1))
        text_mask_2 = StaticTile.apply(text_mask, (2, 1, 1))

        # Time normalization expansion ([2B,1,1]); time_encoder runs inside
        # vector_field (Tile_2 -> /vector_field/time_encoder/... in the trace).
        t_norm_2 = StaticTile.apply(t_norm, (2, 1, 1))

        # Uncond masking (Concat_5): text_emb and text_special_token.
        # B/T are sourced from text_emb (trace: Shape_3 -> Gather/Gather_2),
        # ones_like(text_emb[:, :1, :]) is Slice -> Shape_2 -> ConstantOfShape_1,
        # the multiply is /Mul_1.
        u_text = self.tts.ttl.uncond_masker.text_special_token
        text_uncond = u_text.expand(text_emb.shape[0], -1, text_emb.shape[2])
        text_uncond = text_uncond * torch.ones_like(text_emb[:, :1, :])
        text_emb_2 = torch.cat([text_emb, text_uncond], dim=0)

        # Style value expansion (Concat_7): B sourced from style_ttl
        # (trace: Shape_8 -> Gather_4 -> Expand_3).
        style_uncond = self.tts.ttl.uncond_masker.style_value_special_token.expand(style_ttl.shape[0], -1, -1)
        style_ttl_2 = torch.cat([style_ttl, style_uncond], dim=0)

        # Style key expansion (Concat_6): ONNX /Tile repeats the baked key
        # constant [1, 50, 256] to [B, 50, 256] with dynamic repeats
        # Concat([B, 1, 1]) from Shape(noisy_latent). The uncond key expands
        # with B from the tiled cond key (trace: Shape_6 -> Gather_3 -> Expand_2).
        style_key_cond = DynamicBatchTile.apply(self.tile, noisy_latent)
        style_key_uncond = self.tts.ttl.uncond_masker.style_key_special_token.expand(style_key_cond.shape[0], -1, -1)
        style_key_2 = torch.cat([style_key_cond, style_key_uncond], dim=0)

        # 1-4. Network trunk (proj_in -> main_blocks -> last_convnext -> proj_out),
        # with time_encoder computed inside. Runs under the `/vector_field/` scope.
        v_out_2 = self.tts.ttl.vector_field(
            noisy_latent_2, text_emb_2, style_ttl_2, style_key_2,
            latent_mask_2, text_mask_2, t_norm_2,
        )

        # 5. CFG Combination
        # ONNX trace uses the EXPANDED form (Mul, Mul, Sub):
        #   (1 + cfg) * v_cond - cfg * v_uncond
        # which is algebraically identical to v_cond + cfg * (v_cond - v_uncond)
        # but emits the same node pattern as the production graph.
        v_cond, v_uncond = v_out_2.chunk(2, dim=0)
        v_final = (1.0 + self.cfg_scale) * v_cond - self.cfg_scale * v_uncond

        # 6. Euler Integration Step
        denoised = noisy_latent + step_size * v_final
        return denoised * latent_mask


if __name__ == "__main__":
    B = 2
    L_length = 144
    T_latent = 100
    T_text = 60
    
    device = "cpu"
    print(f"Testing VectorFieldEstimator on {device}")
    
    model = VectorFieldEstimator().to(device).eval()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    noisy_latent = torch.randn(B, L_length, T_latent, device=device)
    text_emb     = torch.randn(B, 256, T_text, device=device)
    style_ttl    = torch.randn(B, 50, 256, device=device)
    latent_mask  = torch.ones(B, 1, T_latent, device=device)
    text_mask    = torch.ones(B, 1, T_text, device=device)
    current_step = torch.tensor([10.0, 10.0], device=device)
    total_step   = torch.tensor([50.0, 50.0], device=device)
    
    with torch.no_grad():
        out = model(noisy_latent, text_emb, style_ttl,
                    latent_mask, text_mask, current_step, total_step)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, 144, T_latent)
    print("Inference Success! Architecture follows ONNX V3 trace 1-to-1.")