# generic imports
from typing import Dict, Optional

# torch imports
import torch
from torch import nn
import numpy as np

# custom imports
from bpemb import BPEmb
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders import get_encoder
from i2t.utils import instantiate, ClassDescription
from i2t.clip_model import ModifiedResNet, VisionTransformer, Transformer, LayerNorm


class TextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        context_length: int = 49,
        vocab_size: int = 200_000,
        transformer_width: int = 200,  # 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        pretrained_bpemb_embeddings: bool = True,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        # transformer_heads = transformer_width // 64

        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        if pretrained_bpemb_embeddings:
            emb = BPEmb(lang="ru", dim=transformer_width, vs=vocab_size)
            vctrs = torch.cat([torch.tensor(emb.vectors), torch.zeros(1, transformer_width)], dim=0)
            self.token_embedding = nn.Embedding.from_pretrained(
                vctrs,
                freeze=freeze_embeddings,
                padding_idx=vocab_size,
                sparse=False,
            )
        else:
            self.token_embedding = nn.Embedding(vocab_size + 1, transformer_width, padding_idx=vocab_size)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.output_dim = transformer_width
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text_data):
        text = text_data["ids"]  # NL
        key_padding_mask = text_data["mask"]  # NL
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        mask = ~key_padding_mask
        x = torch.sum(x * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=1, keepdim=True)  # ND
        x = x @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask


class ModalityEncoder(nn.Module):
    """Simple wrapper around encoder, adds output projection layer.
    """
    def __init__(
        self,
        encoder: ClassDescription,
        output_dim: int,
        normalize: bool = True
    ):
        super().__init__()
        self.encoder = instantiate(encoder)

        # if not isinstance(self.encoder, (ModifiedResNet, VisionTransformer)):
            # self.projector = nn.Linear(self.encoder.output_dim, output_dim, bias=False)
        self.normalize = nn.functional.normalize if normalize else (lambda x: x)

        self.initialize_parameters()
    
    def forward(self, *args, **kwargs):
        features = self.encoder(*args, **kwargs)
        features = self.normalize(features)

        # if hasattr(self.encoder, "logit_scale"):
            # features = self.encoder.logit_scale.exp() * features

        return features
        # projected_features = self.projector(features)
        # return self.normalize(projected_features)

    def initialize_parameters(self):
        if isinstance(self.encoder, TextEncoder):
            nn.init.normal_(self.encoder.token_embedding.weight, std=0.02)
            nn.init.normal_(self.encoder.positional_embedding, std=0.01)

        if isinstance(self.encoder, ModifiedResNet):
            if self.encoder.attnpool is not None:
                std = self.encoder.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.encoder.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        if isinstance(self.encoder, TextEncoder):
            proj_std = (self.encoder.transformer.width ** -0.5) * ((2 * self.encoder.transformer.layers) ** -0.5)
            attn_std = self.encoder.transformer.width ** -0.5
            fc_std = (2 * self.encoder.transformer.width) ** -0.5
            for block in self.encoder.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.encoder.text_projection is not None:
                nn.init.normal_(self.encoder.text_projection, std=self.encoder.transformer.width ** -0.5)



class ImageModel(nn.Module):
    """Thin wrapper around SMP encoders.
    """
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        weights: Optional[str] = 'imagenet',
    ):
        super().__init__()
        self.encoder = get_encoder(name=encoder_name, weights=weights)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self.encoder.out_channels[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)[-1]
        x = self.avgpool(x)
        return torch.flatten(x, start_dim=1)


class TextModel(nn.Module):
    """Simple BoW-based text encoder.
    """
    def __init__(
        self,
        hidden_size: int = 200,
        hidden_layers: int = 3,
        embedding_size: int = 200,
        vocab_size: int = 200000,
        pretrained_bpemb_embeddings: bool = True,
        freeze_embeddings: bool = False
    ):
        super().__init__()

        if pretrained_bpemb_embeddings:
            emb = BPEmb(lang="ru", dim=embedding_size, vs=vocab_size)
            self.embedding = nn.EmbeddingBag.from_pretrained(
                torch.tensor(emb.vectors),
                freeze=freeze_embeddings,
                sparse=False,
            )
        else:
            self.embedding = nn.EmbeddingBag(
                vocab_size,
                embedding_dim=embedding_size,
                sparse=False
            )

        self.output_dim = hidden_size

        in_channels = [embedding_size, *(hidden_size for _ in range(hidden_layers))]
        out_channels = [hidden_size for _ in range(hidden_layers + 1)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inc, outc),
                nn.BatchNorm1d(outc),
                nn.ReLU(inplace=True),
            )
            for inc, outc in zip(in_channels, out_channels)
        ])

    def forward(self, text_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embedding(text_data['ids'], text_data['offsets'])
        for block in self.blocks:
            x = block(x)
        return x
