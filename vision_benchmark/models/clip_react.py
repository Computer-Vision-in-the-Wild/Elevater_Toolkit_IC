from collections import OrderedDict
from multiprocessing.sharedctypes import Value
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from timm.models.layers import DropPath, trunc_normal_
from vision_benchmark.datasets.languages.build import build_tokenizer


logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, LayerNorm)):
            # if is_main_process():
            #     logger.info('=> init {} gamma to 1'.format(m))
            #     logger.info('=> init {} beta to 0'.format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            # if is_main_process():
            #     logger.info('=> init weight of Linear/Conv2d from tranc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if is_main_process():
                #     logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 use_quick_gelu: bool = True,
                 mlp_ratio: float = 4.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
            ("gelu", QuickGELU() if use_quick_gelu else nn.GELU()),
            ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class GatedResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 use_quick_gelu: bool = True,
                 mlp_ratio: float = 4.0,
                 rcp_block_cfg: dict = {}):
        super().__init__()

        self.use_ffn = rcp_block_cfg.get('USE_FFN', True)
        self.d_block = rcp_block_cfg.get('WIDTH', d_model)
        self.use_gumbel_sample = rcp_block_cfg.get('GUMBEL_SAMPLE', False)

        self.use_block_proj = False
        if self.d_block != d_model:
            self.use_block_proj = True
            self.ln_pre = LayerNorm(d_model)
            self.proj_pre = nn.Linear(d_model, self.d_block)
            self.ln_post = LayerNorm(self.d_block)
            self.proj_post = nn.Linear(self.d_block, d_model)

        self.attn = nn.MultiheadAttention(self.d_block, n_head)
        self.ln_1 = LayerNorm(self.d_block)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        if self.use_gumbel_sample:
            self.register_parameter('gumbel_attn', nn.Parameter(torch.tensor(0.)))
        else:
            self.gumbel_attn = None

        if self.use_ffn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(self.d_block, int(self.d_block * mlp_ratio))),
                ("gelu", QuickGELU() if use_quick_gelu else nn.GELU()),
                ("c_proj", nn.Linear(int(self.d_block * mlp_ratio), self.d_block))
            ]))
            self.ln_2 = LayerNorm(self.d_block)
            self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))
            if self.use_gumbel_sample:
                self.register_parameter('gumbel_dense', nn.Parameter(torch.tensor(0.)))
            else:
                self.gumbel_dense = None
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def prob_gate(self, gumbel_val, eps=1e-10):
        if not self.use_gumbel_sample:
            return 1.0
        prob = (1.0 + gumbel_val.tanh()) / 2
        prob = torch.clamp(prob, eps, 1.0 - eps)
        log_probs = torch.stack([prob, 1.0 - prob]).log()
        return F.gumbel_softmax(log_probs, hard=True)[0]

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.use_block_proj:
            x0 = x
            x = self.proj_pre(self.ln_pre(x))
        x = x + self.prob_gate(self.gumbel_attn) * self.alpha_attn.tanh() * self.drop_path(self.attention(self.ln_1(x)))
        if self.use_ffn:
            x = x + self.prob_gate(self.gumbel_dense) * self.alpha_dense.tanh() * self.drop_path(self.mlp(self.ln_2(x)))
        if self.use_block_proj:
            x = x0 + self.proj_post(self.ln_post(x))
        return x


class ResidualAttentionRCPBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 use_quick_gelu: bool = True,
                 mlp_ratio: float = 4.0,
                 rcp_block_cfg: dict = None):
        super().__init__()

        # reset drop path if there is rcp-specific drop path
        drop_path = rcp_block_cfg.get('DROP_PATH', drop_path)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
            ("gelu", QuickGELU() if use_quick_gelu else nn.GELU()),
            ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.rcp_block_mode = rcp_block_cfg['MODE']
        if self.rcp_block_mode == 'gated_attn':
            self.gated_attn_block = GatedResidualAttentionBlock(
                d_model=d_model,
                n_head=n_head,
                attn_mask=attn_mask,
                drop_path=drop_path,
                use_quick_gelu=use_quick_gelu,
                rcp_block_cfg=rcp_block_cfg,
            )
        elif self.rcp_block_mode == 'prompt_tuning':
            # initialize the prompt embeddings
            self.num_prompt_tokens = rcp_block_cfg['NUM_PROMPT_TOKENS']
            # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            val = np.sqrt(6. / float(self.num_prompt_tokens * 4)) # assuming n_prompt_tokens == patch size
            self.register_parameter('prompt_embeddings', nn.Parameter(torch.empty(self.num_prompt_tokens, 1, d_model)))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            raise ValueError(f'Unknown rcp block mode: {self.rcp_block_mode}')

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.rcp_block_mode == 'gated_attn':
            x = self.gated_attn_block(x)

        if self.rcp_block_mode == 'prompt_tuning':
            x = torch.cat((x[:1], self.prompt_embeddings.expand(-1, x.shape[1], -1), x[1:]), dim=0)

        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        if self.rcp_block_mode == 'prompt_tuning':
            x = torch.cat((x[:1], x[1+self.num_prompt_tokens:]), dim=0)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 mlp_ratio: float = 4.0,
                 use_quick_gelu: bool = True,
                 use_rcp_block: bool = False,
                 rcp_block_cfg: dict = None):
        super().__init__()
        self.width = width
        self.layers = layers

        if use_rcp_block:
            use_last_k_layers = rcp_block_cfg.get('USE_LAST_K', -1)
            if use_last_k_layers == -1:
                use_last_k_layers = layers
            rcp_layer_indices = list(range(layers))[-use_last_k_layers:]
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionRCPBlock(width, heads, attn_mask, drop_path, use_quick_gelu, mlp_ratio, rcp_block_cfg)
                    if layer_idx in rcp_layer_indices
                    else ResidualAttentionBlock(width, heads, attn_mask, drop_path, use_quick_gelu, mlp_ratio)
                    for layer_idx in range(layers)
                ]
            )
        else:
            self.resblocks = nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask, drop_path, use_quick_gelu, mlp_ratio)
                    for _ in range(layers)
                ]
            )

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # if is_main_process():
            #     logger.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if is_main_process():
                #     logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 output_dim: int,
                 pool_type: str = 'default',
                 skip_cls: bool = False,
                 drop_path: float = 0.0,
                 use_rcp_block: bool = False,
                 use_quick_gelu: bool = True,
                 rcp_block_cfg: dict = None):
        super().__init__()
        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.sequence_length = (input_resolution // patch_size) ** 2 + 1

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            if (not self.skip_cls):
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length, stride=self.sequence_length, groups=width)
            else:
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length-1, stride=self.sequence_length, groups=width)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                self.sequence_length, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            use_rcp_block=use_rcp_block,
            use_quick_gelu=use_quick_gelu,
            rcp_block_cfg=rcp_block_cfg,
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # if is_main_process():
            #     logger.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if is_main_process():
                #     logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if (self.pool_type == 'average'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = torch.mean(x,dim=1)
        elif (self.pool_type == 'linear'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[:, 0, :]

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_drop_path: int,
                 # text
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_style: str = 'clip',
                 gather_tensors: bool = False,
                 tokenizer_style: str = 'clip',
                 pool_type: str = 'default',
                 skip_cls: bool = False,
                 config: dict = {},
                 ):
        super().__init__()

        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.context_length = context_length
        self.transformer_style = transformer_style
        self.transformer_width = transformer_width
        self.gather_tensors = gather_tensors

        self.tokenizer_style = tokenizer_style
        self.tokenizer = build_tokenizer(tokenizer_style)

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = config['MODEL']['SPEC']['VISION'].get('HEADS', vision_width // 64)
            mlp_ratio = config['MODEL']['SPEC']['VISION'].get('MLP_RATIO', 4)
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                mlp_ratio=mlp_ratio,
                output_dim=embed_dim,
                pool_type = self.pool_type,
                skip_cls=self.skip_cls,
                drop_path=vision_drop_path,
                use_rcp_block=config['MODEL']['SPEC']['VISION'].get('USE_RCP_BLOCK', False),
                use_quick_gelu=config['MODEL']['SPEC'].get('USE_QUICK_GELU', True),
                rcp_block_cfg=config['MODEL']['SPEC'].get('RCP_BLOCK', {}),
            )

        if self.transformer_style == 'clip':
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                use_rcp_block=config['MODEL']['SPEC']['TEXT'].get('USE_RCP_BLOCK', False),
                use_quick_gelu=config['MODEL']['SPEC'].get('USE_QUICK_GELU', True),
                rcp_block_cfg=config['MODEL']['SPEC'].get('RCP_BLOCK', {}),
            )
            self.token_embedding = nn.Embedding(
                self.tokenizer.get_vocab_size(), transformer_width
            )
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, transformer_width)
            )
            # trunc_normal_(self.positional_embedding, std=.02)
        elif self.transformer_style.startswith('hf_') or self.transformer_style.startswith('hfc_'):
            logger.info('=> Using HuggingFace model {}'.format(self.transformer_style))
            if (not self.transformer_style.startswith('hfc_')):
                self.transformer = AutoModel.from_pretrained(
                    self.transformer_style[3:]
                )
            else:
                self.transformer = AutoModel.from_pretrained(
                    self.transformer_style[4:]
                )

            transformer_width = self.transformer(
                torch.zeros((1,1)).type(torch.LongTensor)
            )['last_hidden_state'].shape[2]

            self.transformer_width = transformer_width

        if self.transformer_style.startswith('hfc_'):
            self.transformer.init_weights()

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            self.conv_pool = nn.Conv1d(
                self.transformer_width,
                self.transformer_width,
                self.context_length,
                stride=self.context_length,
                groups=self.transformer_width
            )

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # for hybrid contrastive loss
        if 'USE_CLS_BIAS' in config['MODEL'] and config['MODEL']['USE_CLS_BIAS']:
            self.cls_bias = nn.Parameter(torch.zeros(1, config['MODEL']['NUM_CLASSES']))

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, norm=True):
        x = self.visual(image.type(self.dtype))

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
    # 'text' is not the raw text, it is the tokens.

        if self.transformer_style == 'clip':
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        elif self.transformer_style.startswith('hf_'):
            x = self.transformer(text)['last_hidden_state']

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if (self.pool_type == 'default'):
            if self.tokenizer_style == 'clip':
                x = x[
                    torch.arange(x.shape[0]),
                    text.argmax(dim=-1)
                ]
            elif (
                self.tokenizer_style.startswith('hf_')
                or self.tokenizer_style.startswith('hfc_')
            ):
                x = x[
                    torch.arange(x.shape[0]),
                    (text == self.tokenizer.get_eot_token()).nonzero(as_tuple=True)[0][0]
                ]
        elif (self.pool_type == 'linear'):
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[
                    torch.arange(x.shape[0]),
                    :
                ]
            x = torch.mean(x, dim=1)

        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def get_cls_bias(self, target):
        # get category bias
        if not hasattr(self, "cls_bias"):
            return None
        else:
            return torch.gather(self.cls_bias, 1, target.view(1, -1))

    def forward(self, image, text, target):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        self.logit_scale.data.clamp_(0, 4.60517)
        T = self.logit_scale.exp()

        cls_bias = self.get_cls_bias(target-1)

        return features_image, features_text, T, cls_bias


def get_zeroshot_model(config, **kwargs):
    embed_dim = config['MODEL']['SPEC']['EMBED_DIM']

    image_resolution = config['TRAIN']['IMAGE_SIZE'][0]
    spec_vision = config['MODEL']['SPEC']['VISION']
    vision_width = spec_vision['WIDTH']
    vision_drop_path = spec_vision.get('DROP_PATH', 0.0)

    if (spec_vision['MODEL'] == 'vit'):
        vision_layers = spec_vision['LAYERS']
        vision_patch_size = spec_vision['PATCH_SIZE']
    else:
        vision_layers = tuple(spec_vision['LAYERS'])
        vision_patch_size = None

    spec_text = config['MODEL']['SPEC']['TEXT']
    context_length = spec_text['CONTEXT_LENGTH']

    transformer_width = spec_text['WIDTH']
    transformer_heads = spec_text['HEADS']
    transformer_layers = spec_text['LAYERS']
    transformer_style = spec_text['STYLE']
    tokenizer_style = spec_text['TOKENIZER']

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        vision_drop_path, context_length, transformer_width,
        transformer_heads, transformer_layers, transformer_style,
        tokenizer_style=tokenizer_style,
        pool_type=config['MODEL']['SPEC'].get('POOL_TYPE', 'default'),
        skip_cls=config['MODEL']['SPEC'].get('SKIP_CLS', False),
        config=config,
    )

    return model
