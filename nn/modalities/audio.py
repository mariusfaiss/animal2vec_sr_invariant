#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

from functools import partial
import torch
import torch.nn as tnn
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from nn import ConvFeatureExtractionModel
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
    Fp32LayerNorm
)
from fairseq.tasks import FairseqTask
# from fairseq.modules import SinusoidalPositionalEmbedding
from .base import D2vModalityConfig, ModalitySpecificEncoder, get_alibi_bias
from .modules import BlockEncoder, Decoder1d
from nn import Modality, get_conv_size
from omegaconf import II
from nn.utils import TemporalEmbedding
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class D2vAudioConfig(D2vModalityConfig):
    type: Modality = Modality.AUDIO
    extractor_mode: str = "layer_norm"
    conv_feature_layers: str = II("task.conv_feature_layers")
    sample_rate: int = II("task.sample_rate")
    max_sample_size: int = II("task.max_sample_size")
    conv_pos_width: int = field(
        default=95,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": "depth of positional encoder network"},
    )
    conv_pos_pre_ln: bool = False
    sinc_input: bool = False
    apply_window_to_root: bool = False
    sinc_norm: str = "instance"
    use_pswish: bool = False
    positional_encoder_type: str = II("task.positional_encoder_type")
    all_sample_rates: str = II("task.all_sample_rates")
    # dummy_data: bool = II("task.dummy_data")


class AudioEncoder(ModalitySpecificEncoder):
    modality_cfg: D2vAudioConfig

    def __init__(
            self,
            modality_cfg: D2vAudioConfig,
            embed_dim: int,
            make_block: Callable[[float], tnn.ModuleList],
            norm_layer: Callable[[int], tnn.LayerNorm],
            layer_norm_first: bool,
            alibi_biases: Dict,
            task: Optional[FairseqTask],
    ):

        self.feature_enc_layers = eval(modality_cfg.conv_feature_layers)
        feature_embed_dim = self.feature_enc_layers[-1][0]
        self.positional_encoder_type = modality_cfg.positional_encoder_type
        # self.dummy_data = modality_cfg.dummy_data
        self.all_sample_rates = eval(modality_cfg.all_sample_rates)
        self.max_sample_size = modality_cfg.max_sample_size

        local_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=modality_cfg.extractor_mode,
            conv_bias=False,
            sinc_input=modality_cfg.sinc_input,
            apply_window_to_root=modality_cfg.apply_window_to_root,
            sample_rate=modality_cfg.sample_rate,
            sinc_norm=modality_cfg.sinc_norm,
            use_pswish=modality_cfg.use_pswish,
        )

        project_features = tnn.Sequential(
            TransposeLast(),
            # tnn.LayerNorm(feature_embed_dim),
            Fp32LayerNorm(feature_embed_dim, elementwise_affine=True),
            tnn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)


        positional_encoder = self.positional_encoder_type.lower()
        if positional_encoder == "learnable":
            fixed_positional_encoder = None
            relative_positional_encoder = tnn.Sequential(
                TransposeLast(),
                *[
                    tnn.Sequential(
                        tnn.Conv1d(
                            embed_dim,
                            embed_dim,
                            kernel_size=k,
                            padding=k // 2,
                            groups=modality_cfg.conv_pos_groups,
                        ),
                        SamePad(k),
                        TransposeLast(),
                        Fp32LayerNorm(embed_dim, elementwise_affine=False),
                        TransposeLast(),
                        tnn.GELU(),
                    )
                    for _ in range(num_pos_layers)
                ],
                TransposeLast(),
            )
        elif positional_encoder in ["none", "rotary"]:
            fixed_positional_encoder = None
            relative_positional_encoder = None
        else:
            # calculate maximum allowed length after feature encoder
            max_length = [self.max_sample_size]
            for xx in self.feature_enc_layers:
                max_length = [get_conv_size(max_length, [min(10, xx[1])], [0], [1], [xx[2]], dim=1)[0]]

            if positional_encoder == "temporal":
                relative_positional_encoder = None
                fixed_positional_encoder = TemporalEmbedding(embed_dim=embed_dim, max_length=max_length[0],
                                                             ref_sr=modality_cfg.sample_rate,
                                                             list_of_rates=self.all_sample_rates, adaptive_sr=True)
            elif positional_encoder == "temporaltape":
                relative_positional_encoder = None
                fixed_positional_encoder = TemporalEmbedding(embed_dim=embed_dim, max_length=max_length[0],
                                                             ref_sr=modality_cfg.sample_rate,
                                                             list_of_rates=self.all_sample_rates, adaptive_sr=True,
                                                             tape=True)
            elif positional_encoder == "tape":
                relative_positional_encoder = None
                fixed_positional_encoder = TemporalEmbedding(embed_dim=embed_dim, max_length=max_length[0],
                                                             ref_sr=modality_cfg.sample_rate, adaptive_sr=False,
                                                             tape=True)
            elif positional_encoder == "sinusoidal":
                relative_positional_encoder = None
                fixed_positional_encoder = TemporalEmbedding(embed_dim=embed_dim, max_length=max_length[0],
                                                             ref_sr=modality_cfg.sample_rate, adaptive_sr=False)
            else:
                raise ValueError(f"Invalid positional encoder type: {self.positional_encoder_type}\n"
                                 f"Possible types: learnable, temporal, temporaltape, tape, sinusoidal, rotary, none")

        if modality_cfg.conv_pos_pre_ln:
            if fixed_positional_encoder is not None:
                fixed_positional_encoder = tnn.Sequential(Fp32LayerNorm(embed_dim), fixed_positional_encoder)
            if relative_positional_encoder is not None:
                relative_positional_encoder = tnn.Sequential(Fp32LayerNorm(embed_dim), relative_positional_encoder)

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            tnn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )
        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else None
        )
        # print("\n\n decoder:", decoder)

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=relative_positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            # def _conv_out_length(input_length, kernel_size, stride):
            #     return torch.floor((input_length - kernel_size) / stride + 1)
            #
            # for i in range(len(self.feature_enc_layers)):
            #     input_lengths = _conv_out_length(
            #         input_lengths,
            #         self.feature_enc_layers[i][1],
            #         self.feature_enc_layers[i][2],
            #     )

            # ft_out_size = input_lengths.tolist()
            # for xx in self.feature_enc_layers:
            #      ft_out_size = [get_conv_size(ft_out_size, [xx[1]], [0], [1], [xx[2]], dim=1)[0]]

            # Start with the input tensor. Use float for the division operations.
            lengths = input_lengths.float()

            for _, kernel_size, stride in self.feature_enc_layers:
                # Apply the formula to the entire tensor of lengths at once
                lengths = torch.ceil(lengths / stride)

            # Convert back to LongTensor as required by many modules (e.g., nn.CTCLoss)
            return lengths.to(torch.long)
            # return torch.tensor(ft_out_size).to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to

                # place a 1 at the last valid position in each sequence
                padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1,)] = 1

                padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
            else:
                padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        return padding_mask

    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.project_features.children():
            if isinstance(mod, tnn.Linear):
                mod.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()
