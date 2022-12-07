from turtle import forward
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import math
import numpy as np
import torch
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor, device, nn

from vienna.models.encoders import resnet_encoders
from vienna.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vienna.models.utils import (
    CustomFixedCategorical,
    DotProductAttention,
    MultiHeadDotProductAttention,
    TemperatureTanh,
)
from vienna.models.encoders.audio_encoder import AudioCNN

from vienna.models.CLEnc import CLEnc
import traceback
PREV_ACTION_DIM = 4
PANO_ATTN_KEY_DIM = 128
ANGLE_FEATURE_SIZE = 4
GPS_FEATURE_SIZE = 2
from vienna.models.waypoint_policy import WaypointPolicy
from vienna.models.waypoint_predictors import WaypointPredictionNet
from vienna.models.rotograd import RotateOnly, RotoGrad, divide, rotate
from torch import distributed as distrib

def wrap_helper(tensor):
    T, N = tensor.shape[:2]
    return tensor.view(T * N, *(tensor.shape[2:]))

def wrap_helper_pano(tensor):
    T, N, M = tensor.shape[:3]
    return tensor.view(T * N * M, *(tensor.shape[3:]))


class GRU(nn.Module):
    def __init__(self, d_model):
        super(GRU, self).__init__()
        self.l_rx = nn.Linear(d_model, d_model)
        self.l_ry = nn.Linear(d_model, d_model)
        self.l_zx = nn.Linear(d_model, d_model)
        self.l_zy = nn.Linear(d_model, d_model)
        self.l_nx = nn.Linear(d_model, d_model)
        self.l_ny = nn.Linear(d_model, d_model)
    
    def forward(self, x, y):
        r = torch.sigmoid(self.l_rx(x) + self.l_ry(y))
        z = torch.sigmoid(self.l_zx(x) + self.l_zy(y))
        h = torch.tanh(self.l_ny(y)+self.l_nx(r * x))
        res = (1 - z) * x + z * h
        return res

class TFEL(nn.TransformerEncoderLayer):
    '''
        TransformerEncoderLayer
    '''
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(d_model, *args, **kwargs)
        self.g_mha = GRU(d_model)
        self.g_mlp = GRU(d_model)
        
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_input = self.norm1(src)
        src2 = self.self_attn(src_input, src_input, src_input, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src = src + self.dropout1(src2)
        src = self.g_mha(src, self.dropout1(src2))
        src_input = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_input))))
        # src = src + self.dropout2(src2)
        src = self.g_mlp(src, self.dropout2(src2))

        return src

class TFDL(nn.TransformerDecoderLayer):
    '''
        TransformerDecoderLayer
    '''
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(d_model, *args, **kwargs)
        self.g_sa = GRU(d_model)
        self.g_mha = GRU(d_model)
        self.g_mlp = GRU(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_input = self.norm1(tgt)
        tgt2 = self.self_attn(tgt_input, tgt_input, tgt_input, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        tgt = self.g_sa(tgt, self.dropout1(tgt2))

        tgt_input = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt_input, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout2(tgt2)
        tgt = self.g_mha(tgt, self.dropout2(tgt2))
        
        tgt_input = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_input))))
        # tgt = tgt + self.dropout3(tgt2)
        tgt = self.g_mlp(tgt, self.dropout3(tgt2))
        
        return tgt

class TFNet(Net):
    r""" A TransFormer (TF) network that contains:
        Instruction encoder
        Depth encoder
        RGB encoder
        A transformer encoder for sequence.
    """

    def __init__(self, observation_space: Space, model_config: Config):
        super().__init__()
        self.model_config = model_config
        self.wypt_cfg = model_config.WAYPOINT
        self._num_panos = self.model_config.num_panos
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.compass_embedding_size = 128
        self.gps_embedding_size = 128
        self.object_num = 21
        self.target_token_size = model_config.RGB_ENCODER.output_size
        self.pe_max_length = 505

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        # Init the audio encoder
        self.audio_encoder = AudioCNN([65, 26, 2], model_config.RGB_ENCODER.output_size)

        # Init the Imagegoal encoder
        # self.imagegoal_encoder = ImageGoalEncoder(
        #     observation_space, self.target_token_size, device,
        #     spatial_output=True
        # )

        # Init the object category embedding
        self.object_embedding = nn.Embedding(self.object_num + 1, self.target_token_size)

        # Init the task embedding for task tokens
        self.task_embedding = nn.Embedding(4, self.target_token_size)


        # self.rgb_linear = nn.Linear(64 * self.target_token_size, self.target_token_size)
        # self.depth_linear = nn.Linear(16 * self.target_token_size, self.target_token_size)
        

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        # self.depth_encoder = VlnResnetDepthEncoder(
        #     observation_space,
        #     output_size=model_config.DEPTH_ENCODER.output_size,
        #     checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
        #     backbone=model_config.DEPTH_ENCODER.backbone,
        #     spatial_output=True,
        # )
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet50'."

        # self.rgb_encoder = TorchVisionResNet50(
        #     observation_space,
        #     model_config.RGB_ENCODER.output_size,
        #     device,
        #     spatial_output=True,
        # )
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=True,
        )

        print('rgb shape', self.rgb_encoder.output_shape)
        print('depth shape', self.depth_encoder.output_shape)

        # self.rgb_linear = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(
        #         self.rgb_encoder.output_shape[0],
        #         model_config.RGB_ENCODER.output_size,
        #     ),
        #     nn.ReLU(True),
        # )
        # self.depth_linear = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(
        #         np.prod(self.depth_encoder.output_shape),
        #         model_config.DEPTH_ENCODER.output_size,
        #     ),
        #     nn.ReLU(True),
        # )

        # self.rgb_linear = nn.Sequential(
        #     nn.Linear(
        #         self.rgb_encoder.output_shape[0],
        #         model_config.RGB_ENCODER.output_size,
        #     ),
        #     nn.ReLU(True),
        # )
        # self.depth_linear = nn.Sequential(
        #     nn.Linear(
        #         self.depth_encoder.output_shape[0],
        #         model_config.DEPTH_ENCODER.output_size,
        #     ),
        #     nn.ReLU(True),
        # )
        

        self.rgb_spatial_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0])
        self.depth_spatial_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0])

        # self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        self.modal_type_embedding = nn.Embedding(3, self.target_token_size)

        # self.rcm_state_encoder = model_config.CMA.rcm_state_encoder

        
        self._hidden_size = model_config.RGB_ENCODER.output_size
        self.pe_dim = self.target_token_size

        self.pe = {}

        self.rgb_attn = nn.MultiheadAttention(self.target_token_size, 1)
        self.depth_attn = nn.MultiheadAttention(self.target_token_size, 1)
        self.audio_attn = nn.MultiheadAttention(self.target_token_size, 1)
        
        


        target_tf_layer = TFEL(self.target_token_size, nhead=1)
        self.target_tf = nn.TransformerEncoder(target_tf_layer, 2)

        msi_layer = TFEL(self.target_token_size, nhead=1)
        self.msi = nn.TransformerEncoder(msi_layer, 2)

        # self.linear_et = nn.Linear(self.target_token_size * 3 + PREV_ACTION_DIM + self.compass_embedding_size + self.gps_embedding_size, self.target_token_size)
        self.linear_et = nn.Linear(self.target_token_size * 3 + PREV_ACTION_DIM, self.target_token_size)

        ehe_layer = TFEL(self.target_token_size, nhead=4)
        self.ehe = nn.TransformerEncoder(ehe_layer, 2)

        self.mha = nn.MultiheadAttention(self.target_token_size, 4)

        mtp_layer = TFDL(self.target_token_size, nhead=4)
        self.mtp = nn.TransformerDecoder(mtp_layer, 2)

        final_feature_size = model_config.RGB_ENCODER.output_size + model_config.DEPTH_ENCODER.output_size + ANGLE_FEATURE_SIZE
        self.linear_ht = nn.Sequential(
            nn.Linear(
                self.target_token_size,
                final_feature_size,
            ),
            nn.ReLU(True),
        )
        self.stop_linear = nn.Linear(self.target_token_size, 1)


        in_dim = final_feature_size + self.target_token_size

        self._init_distance_linear(in_dim, final_feature_size)
        self._init_offset_linear(in_dim, final_feature_size)

        self.train()
        self._init_pe()

    def distance_to_continuous(self, distance: Tensor) -> Tensor:
        """Maps a distance prediction to a continuous radius r in meters."""
        if self.wypt_cfg.continuous_distance:
            return distance

        range_dist = (
            self.wypt_cfg.max_distance_prediction
            - self.wypt_cfg.min_distance_prediction
        )
        meters_per_distance = range_dist / (
            self.wypt_cfg.discrete_distances - 1
        )
        return self.wypt_cfg.min_distance_prediction + (
            distance * meters_per_distance
        )

    def offset_to_continuous(self, offset: Tensor) -> Tensor:
        """Maps an offset prediction to a continuous offset in radians."""
        if self.wypt_cfg.continuous_offset:
            return offset

        radians_per_pano = 2 * np.pi / self._num_panos
        rad_per_offset = radians_per_pano / (
            self.wypt_cfg.discrete_offsets - 1
        )
        return (-radians_per_pano / 2) + (offset * rad_per_offset)

    def _map_pano_to_heading_features(self, pano: Tensor) -> Tensor:
        """Maps a tensor of pano ids to heading features
        Args:
            pano: size [B, 1]
        """
        delta_rot = (np.pi * 2) / self._num_panos
        heading = pano * delta_rot
        return torch.cat([torch.sin(heading), torch.cos(heading)], dim=1)

    def _init_distance_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the distance output to be either discrete or
        continuous. If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_distance:
            self.distance_linear = nn.Sequential(
                nn.Linear(in_dim, 1), nn.Sigmoid()
            )
            self.distance_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.distance_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_distances
            )

    def _init_offset_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the offset output to be either discrete or continuous.
        If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_offset:
            self.offset_linear = nn.Sequential(
                nn.Linear(in_dim, 1),
                TemperatureTanh(temperature=self.wypt_cfg.offset_temperature),
            )
            self.offset_scale = np.pi / self._num_panos
            self.offset_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.offset_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_offsets
            )

    def _init_pe(self):
        self.embed = torch.zeros(self.pe_max_length, self.pe_dim).float().cuda(self.device)
        for p in range(self.pe_max_length):
            for i in range(self.pe_dim):
                if i % 2 == 0:
                    self.embed[p, i] = math.sin(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
                else:
                    self.embed[p, i] = math.cos(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
        

    def position_embedding(self, length):
        return self.embed[:length].detach()

    def fix_modules(self):
        modules = [self.instruction_encoder, self.audio_encoder, self.imagegoal_encoder, self.object_embedding, self.task_embedding, self.depth_encoder, self.rgb_encoder, self.prev_action_embedding, self.modal_type_embedding,
        self.rgb_attn, self.depth_attn, self.audio_attn, self.target_tf, self.msi, self.ehe] # only except mha and mtp
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward_seq(self, OBS, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [T * batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [T * batch_size x RGB_ENCODER.output_size]
        
        """
        # rnn_hidden_states should be batch_second like [layers, batch, hidden_dim]
        # rnn_hidden_states = rnn_hidden_states.permute(1, 0, 2)
        observations = {}
        for k, v in OBS.items():
            observations[k] = v
        #     print(k, v.shape)
        # print('pre action', prev_actions.shape, 'masks', masks.shape)
        if 'rgb' in observations:
            T, N, M, _, _, _= observations['rgb'].shape
        else:
            T, N, M, _, _, _= observations['rgb_features'].shape
        # compass = observations['compass']
        # compass_embedding = torch.stack([torch.sin(compass),torch.cos(compass)], -1).view(T * N, 2, 1).expand(T * N, 2, self.compass_embedding_size//2).reshape(T * N, -1)
        # gps_embedding = observations['gps'].view(T * N, 2, 1).expand(T * N, 2, self.gps_embedding_size//2).reshape(T * N, -1)
        # device = compass.device

        if 'rgb' in observations:
            observations['rgb'] = wrap_helper_pano(observations['rgb'])
        if 'depth' in observations:
            observations['depth'] = wrap_helper_pano(observations['depth'])
        if 'rgb_features' in observations:
            observations['rgb_features'] = wrap_helper_pano(observations['rgb_features'])
        if 'depth_features' in observations:
            observations['depth_features'] = wrap_helper_pano(observations['depth_features'])
        

        depth_embedding = self.depth_encoder(observations) # T * N * M x dim x h x w
        rgb_embedding = self.rgb_encoder(observations) # T * N * M x dim x h x w
        # print(observations['depth_pano'].shape, depth_embedding.shape)
        rgb_embedding = torch.flatten(rgb_embedding, 2) # T * N * M x dim x K
        depth_embedding = torch.flatten(depth_embedding, 2) # T * N * M x dim x K

        device = rgb_embedding.device

        # rgb_embedding = self.rgb_linear(rgb_embedding) # T * N * M x dim
        # depth_embedding = self.depth_linear(depth_embedding) # T * N * M x dim
        
        # rgb_embedding = rgb_embedding.view(T * N, M, -1).permute(0, 2, 1) # T * N x dim x M
        # depth_embedding = depth_embedding.view(T * N, M, -1).permute(0, 2, 1) # T * N x dim x M

        
        
        

        if 'instruction' not in observations:
        # if instruction_embedding is None:
            instruction_embedding = torch.zeros(T * N, 1, self.instruction_encoder.output_size, device=device)
            vln_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 0)
        else:
            observations['instruction'] = wrap_helper(observations['instruction'])
            instruction_embedding = self.instruction_encoder(observations) # T * N x length x dim 
            instruction_embedding = instruction_embedding.permute(0, 2, 1)
            _, l, _ = instruction_embedding.shape
            vln_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 0)
            # print(instruction_embedding.shape)
            # pass
        
        if 'imagegoal' not in observations:
        # if imagegoal_embedding is None:
            imagegoal_embedding = torch.zeros(T * N, 16, self.target_token_size, device=device)
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 1)
        else:
            observations['imagegoal'] = wrap_helper(observations['imagegoal'])
            imagegoal_embedding = self.imagegoal_encoder(observations) # T * N x dim x h x w
            imagegoal_embedding = torch.flatten(imagegoal_embedding, 2).permute(0,2,1) # T * N x 16 x dim
            _, l, _ = imagegoal_embedding.shape
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 1)
            
        
        if 'spectrogram' not in observations:
        # if audio_embedding is None:
            audio_embedding = torch.zeros(T * N, M, self.audio_encoder.output_size, device=device)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            
        else:
            observations['spectrogram'] = wrap_helper(observations['spectrogram'])
            audio_embedding = self.audio_encoder(observations) # T * N * M x dim 
            audio_embedding = audio_embedding.view(T * N, M, -1)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            

        
        if 'objectgoal' in observations:
            objectgoal_embedding = self.object_embedding(observations['objectgoal'].long()).view(T * N, 1, -1)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
        else:
            objectgoal_embedding = torch.zeros(T * N, 1, self.object_embedding.embedding_dim, device=device)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
            

        t_mask = torch.cat([
            (instruction_embedding == 0.0).all(-1),
            (imagegoal_embedding == 0.0).all(-1),
            (audio_embedding == 0.0).all(-1),
            (objectgoal_embedding == 0.0).all(-1)
        ], dim=1
        ) # T * N x L

        TN, L = t_mask.size()
        attn_mask = t_mask.unsqueeze(1).repeat_interleave(L, dim=1) # T * N x L x L
        t_mask = t_mask.permute(1,0).unsqueeze(2) # L x T * N x 1
        

        instruction_embedding = instruction_embedding + vln_task_embedding
        
        imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        
        audio_embedding = audio_task_embedding + audio_embedding
        
        objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding
        
        

        target_ins = torch.cat([instruction_embedding, imagegoal_embedding, audio_embedding, objectgoal_embedding], 1) #   T * N x length x dim

        target_ins = target_ins.permute(1, 0, 2) # length x T * N x dim

        encoded_target = self.target_tf(target_ins, mask=attn_mask) # length x T * N x dim
        # encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim
        encoded_target_avg = (encoded_target * t_mask).sum(0) / t_mask.sum(0)

        _, _, K = rgb_embedding.shape # T * N * M x dim x K

        encoded_target_avg_M = encoded_target_avg.repeat_interleave(M).view(T * N, -1, M).permute(0,2, 1).reshape(T * N * M, -1)
        rgb_attn_embedding, _ = self.rgb_spatial_attn(encoded_target_avg_M.unsqueeze(0), rgb_embedding.permute(2, 0, 1), rgb_embedding.permute(2, 0, 1)) # T * N * M x dim
        rgb_attn_embedding = rgb_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # rgb_attn_embedding = self.rgb_linear(rgb_attn_embedding) # M x T * N x dim

        depth_attn_embedding, _ = self.depth_spatial_attn(encoded_target_avg_M.unsqueeze(0), depth_embedding.permute(2, 0, 1), depth_embedding.permute(2, 0, 1)) # T * N * M x dim
        depth_attn_embedding = depth_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # depth_attn_embedding = self.depth_linear(depth_attn_embedding) # M x T * N x dim

        attended_rgb, _ = self.rgb_attn(encoded_target_avg.unsqueeze(0), rgb_attn_embedding, rgb_attn_embedding)
        attended_rgb = attended_rgb.squeeze(0) # T * N x dim

        attended_depth, _ = self.depth_attn(encoded_target_avg.unsqueeze(0), depth_attn_embedding, depth_attn_embedding)
        attended_depth = attended_depth.squeeze(0) # T * N x dim
        # print(rgb_embedding.shape, depth_embedding.shape)
        # attended_rgb = torch.relu(self.rgb_linear(rgb_embedding))
        # attended_depth = torch.relu(self.depth_linear(depth_embedding))


        attended_audio, _ = self.audio_attn(encoded_target_avg.unsqueeze(0), audio_embedding.permute(1,0,2), audio_embedding.permute(1,0,2))
        attended_audio = attended_audio.squeeze(0) # T * N x dim
        
        # msi_input = torch.stack([attended_rgb, attended_depth, attended_audio], 0) # 3 x T * N x dim

        

        # o_t = self.msi(msi_input) # 3 x T * N x dim
        # o_t = o_t.permute(1,2,0).reshape(T * N, -1) # T * N x dim
        o_t = torch.cat([attended_rgb, attended_depth, attended_audio], -1) # T * N x 3 * dim
        
        # prev_actions = self.prev_action_embedding(
        #     ((prev_actions.float() + 1) * masks).long().view(-1)
        # ) # T * N x dim
        # print(self._map_pano_to_heading_features(prev_actions["pano"]).shape, masks.shape)
        prev_actions = (
            torch.cat(
                [
                    self._map_pano_to_heading_features(prev_actions["pano"]),
                    self.offset_to_continuous(prev_actions["offset"]),
                    self.distance_to_continuous(prev_actions["distance"]),
                ],
                dim=1,
            ).float()
            * masks
        ).view(T * N, -1 ) # T * N x dim



        # e_t = self.linear_et(torch.cat([o_t, prev_actions, compass_embedding, gps_embedding], -1)) # T * N x dim
        e_t = self.linear_et(torch.cat([o_t, prev_actions], -1)) # T * N x dim
        e_t = e_t.view(T, N, -1)

        position_embedding = self.position_embedding(T).expand(N, -1, -1).permute(1, 0, 2) # T x N x dim
        e_t_e = e_t + position_embedding

        idx_c = torch.linspace(0, T - 1, T, dtype=torch.long, device=device).repeat(T, 1)
        idx_r = idx_c.permute(1, 0)
        src_mask = idx_r < idx_c

        e_t_tilde = self.ehe(e_t_e, src_mask) # T x N x dim

        encoded_target = encoded_target.permute(1, 0, 2) # length x T * N x dim
        q_t, _ = self.mha(e_t_tilde.view(1, T * N, -1), encoded_target, encoded_target) # 1 x T * N x dim

        q_t = q_t.view(T, N, -1) # T x N x dim

        tgt_mask = src_mask
        hidden_states = self.mtp(q_t, e_t_tilde, tgt_mask, src_mask) # T x N x dim

        attended_visual_features = torch.cat(
            [
                rgb_attn_embedding.permute(1,0,2),
                depth_attn_embedding.permute(1,0,2),
                observations['angle_features'].view(T * N, M, -1)
            ],
            dim=2,
        ) # T * N x M x dim

        x = hidden_states.view(T * N, -1) # T * N x dim
        query_h = self.linear_ht(x) # T * N x dim
        query_h = query_h.unsqueeze(2) # T * N x dim x 1
        logits = torch.matmul(attended_visual_features, query_h).squeeze(2) # T * N x M

        # print('logits',logits.shape, T, N)

        pano_stop_distribution = CustomFixedCategorical(
            logits=torch.cat([logits, self.stop_linear(x)], dim=1)
        )

        catted_features = torch.cat(
            [
                attended_visual_features,
                x.unsqueeze(1).repeat(1, attended_visual_features.size(1), 1),
            ],
            dim=2,
        )

        # ===========================
        #     Distance Prediction
        # ===========================

        if self.wypt_cfg.continuous_distance:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable1 = (
                self.wypt_cfg.max_distance_prediction
                - self.wypt_cfg.min_distance_prediction
            ) * distance_variable1 + self.wypt_cfg.min_distance_prediction

            distance_variable2 = (
                self.wypt_cfg.max_distance_var - self.wypt_cfg.min_distance_var
            ) * self.distance_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_distance_var
        else:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable2 = None

        # ===========================
        #      Offset Prediction
        # ===========================

        if self.wypt_cfg.continuous_offset:
            offset_variable1 = self.offset_scale * self.offset_linear(
                catted_features
            ).squeeze(2)
            offset_variable2 = (
                self.wypt_cfg.max_offset_var - self.wypt_cfg.min_offset_var
            ) * self.offset_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_offset_var
        else:
            offset_variable1 = self.offset_linear(catted_features).squeeze(2)
            offset_variable2 = None

        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        )

        # return hidden_states, e_t, q_t

    def forward_single(self, OBS, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [T * batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [T * batch_size x RGB_ENCODER.output_size]
        
        """
        # rnn_hidden_states should be batch_second like [layers, batch, hidden_dim]
        # rnn_hidden_states = rnn_hidden_states.permute(1, 0, 2)
        observations = {}
        for k, v in OBS.items():
            observations[k] = v
        #     print(k, v.shape)
        # print('pre action', prev_actions.shape, 'masks', masks.shape)
        if 'rgb' in observations:
            T, N, M, _, _, _= observations['rgb'].shape
        else:
            T, N, M, _, _, _= observations['rgb_features'].shape
        # compass = observations['compass']
        # compass_embedding = torch.stack([torch.sin(compass),torch.cos(compass)], -1).view(T * N, 2, 1).expand(T * N, 2, self.compass_embedding_size//2).reshape(T * N, -1)
        # gps_embedding = observations['gps'].view(T * N, 2, 1).expand(T * N, 2, self.gps_embedding_size//2).reshape(T * N, -1)
        # device = compass.device

        if 'rgb' in observations:
            observations['rgb'] = wrap_helper_pano(observations['rgb'])
        if 'depth' in observations:
            observations['depth'] = wrap_helper_pano(observations['depth'])
        if 'rgb_features' in observations:
            observations['rgb_features'] = wrap_helper_pano(observations['rgb_features'])
        if 'depth_features' in observations:
            observations['depth_features'] = wrap_helper_pano(observations['depth_features'])
        

        depth_embedding = self.depth_encoder(observations) # T * N * M x dim x h x w
        rgb_embedding = self.rgb_encoder(observations) # T * N * M x dim x h x w
        # print(observations['depth_pano'].shape, depth_embedding.shape)
        rgb_embedding = torch.flatten(rgb_embedding, 2) # T * N * M x dim x K
        depth_embedding = torch.flatten(depth_embedding, 2) # T * N * M x dim x K
        device = rgb_embedding.device
        # rgb_embedding = self.rgb_linear(rgb_embedding) # T * N * M x dim
        # depth_embedding = self.depth_linear(depth_embedding) # T * N * M x dim
        
        # rgb_embedding = rgb_embedding.view(T * N, M, -1).permute(0, 2, 1) # T * N x dim x M
        # depth_embedding = depth_embedding.view(T * N, M, -1).permute(0, 2, 1) # T * N x dim x M

        
        
        

        if 'instruction' not in observations:
        # if instruction_embedding is None:
            instruction_embedding = torch.zeros(T * N, 1, self.instruction_encoder.output_size, device=device)
            vln_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 0)
            
        else:
            observations['instruction'] = wrap_helper(observations['instruction'])
            instruction_embedding = self.instruction_encoder(observations) # T * N x length x dim 
            instruction_embedding = instruction_embedding.permute(0, 2, 1)
            _, l, _ = instruction_embedding.shape
            vln_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 0)
            # print(instruction_embedding.shape)
            
            
            # pass
        
        if 'imagegoal' not in observations:
        # if imagegoal_embedding is None:
            imagegoal_embedding = torch.zeros(T * N, 16, self.target_token_size, device=device)
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 1)
            # imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        else:
            observations['imagegoal'] = wrap_helper(observations['imagegoal'])
            imagegoal_embedding = self.imagegoal_encoder(observations) # T * N x dim x h x w
            imagegoal_embedding = torch.flatten(imagegoal_embedding, 2).permute(0,2,1) # T * N x 16 x dim
            _, l, _ = imagegoal_embedding.shape
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 1)
            # imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        
        if 'spectrogram' not in observations:
        # if audio_embedding is None:
            audio_embedding = torch.zeros(T * N, M, self.audio_encoder.output_size, device=device)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            # audio_embedding = audio_task_embedding + audio_embedding
        else:
            observations['spectrogram'] = wrap_helper(observations['spectrogram'])
            audio_embedding = self.audio_encoder(observations) # T * N * M x dim 
            audio_embedding = audio_embedding.view(T * N, M, -1)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            # audio_embedding = audio_task_embedding + audio_embedding

        
        if 'objectgoal' in observations:
            objectgoal_embedding = self.object_embedding(observations['objectgoal'].long()).view(T * N, 1, -1)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
            # objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding
        else:
            objectgoal_embedding = torch.zeros(T * N, 1, self.object_embedding.embedding_dim, device=device)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
            # objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding



        t_mask = torch.cat([
            (instruction_embedding == 0.0).all(-1),
            (imagegoal_embedding == 0.0).all(-1),
            (audio_embedding == 0.0).all(-1),
            (objectgoal_embedding == 0.0).all(-1)
        ], dim=1
        ) # T * N x L

        TN, L = t_mask.size()
        attn_mask = t_mask.unsqueeze(1).repeat_interleave(L, dim=1) # T * N x L x L
        t_mask = t_mask.permute(1,0).unsqueeze(2) # L x T * N x 1
        # t_mask = t_mask | t_mask.permute(0,2,1)
        
        # print(t_mask.shape)
        # for _ in t_mask:
        #     print(_)

        instruction_embedding = instruction_embedding + vln_task_embedding
        
        imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        
        audio_embedding = audio_task_embedding + audio_embedding
        
        objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding

        target_ins = torch.cat([instruction_embedding, imagegoal_embedding, audio_embedding, objectgoal_embedding], 1) #   T * N x length x dim

        target_ins = target_ins.permute(1, 0, 2) # length x T * N x dim

        # print('ins', target_ins.size(), masks.size())

        encoded_target = self.target_tf(target_ins, mask=attn_mask) # length x T * N x dim
        # encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim
        encoded_target_avg = (encoded_target * t_mask).sum(0) / t_mask.sum(0)

        _, _, K = rgb_embedding.shape # T * N * M x dim x K

        encoded_target_avg_M = encoded_target_avg.repeat_interleave(M).view(T * N, -1, M).permute(0,2, 1).reshape(T * N * M, -1)

        # print(encoded_target_avg_M.unsqueeze(0).shape, rgb_embedding.permute(2, 0, 1).shape)
        rgb_attn_embedding, _ = self.rgb_spatial_attn(encoded_target_avg_M.unsqueeze(0), rgb_embedding.permute(2, 0, 1), rgb_embedding.permute(2, 0, 1)) # 1 x T * N * M x dim
        rgb_attn_embedding = rgb_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim
        # print(rgb_attn_embedding.shape)
        # rgb_attn_embedding = self.rgb_linear(rgb_attn_embedding) # M x T * N x dim

        depth_attn_embedding, _ = self.depth_spatial_attn(encoded_target_avg_M.unsqueeze(0), depth_embedding.permute(2, 0, 1), depth_embedding.permute(2, 0, 1)) # 1 x T * N * M x dim
        depth_attn_embedding = depth_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # depth_attn_embedding = self.depth_linear(depth_attn_embedding) # M x T * N x dim

        attended_rgb, _ = self.rgb_attn(encoded_target_avg.unsqueeze(0), rgb_attn_embedding, rgb_attn_embedding)
        attended_rgb = attended_rgb.squeeze(0) # T * N x dim

        attended_depth, _ = self.depth_attn(encoded_target_avg.unsqueeze(0), depth_attn_embedding, depth_attn_embedding)
        attended_depth = attended_depth.squeeze(0) # T * N x dim
        # print(rgb_embedding.shape, depth_embedding.shape)
        # attended_rgb = torch.relu(self.rgb_linear(rgb_embedding))
        # attended_depth = torch.relu(self.depth_linear(depth_embedding))


        attended_audio, _ = self.audio_attn(encoded_target_avg.unsqueeze(0), audio_embedding.permute(1,0,2), audio_embedding.permute(1,0,2))
        attended_audio = attended_audio.squeeze(0) # T * N x dim
        
        # msi_input = torch.stack([attended_rgb, attended_depth, attended_audio], 0) # 3 x T * N x dim

        

        # o_t = self.msi(msi_input) # 3 x T * N x dim
        # o_t = o_t.permute(1,2,0).reshape(T * N, -1) # T * N x dim
        o_t = torch.cat([attended_rgb, attended_depth, attended_audio], -1) # T * N x 3 * dim
        
        
        # prev_actions = self.prev_action_embedding(
        #     ((prev_actions.float() + 1) * masks).long().view(-1)
        # ) # T * N x dim
        prev_actions = (
            torch.cat(
                [
                    self._map_pano_to_heading_features(prev_actions["pano"]),
                    self.offset_to_continuous(prev_actions["offset"]),
                    self.distance_to_continuous(prev_actions["distance"]),
                ],
                dim=1,
            ).float()
            * masks
        ) # T * N x dim

        # print(o_t.shape, prev_actions.shape)
        # e_t = self.linear_et(torch.cat([o_t, prev_actions, compass_embedding, gps_embedding], -1)) # T * N x dim
        e_t = self.linear_et(torch.cat([o_t, prev_actions], -1)) # T * N x dim
        e_t = e_t.view(T, N, -1)

        encoded_target = encoded_target.permute(1, 0, 2) # length x T * N x dim
        

        #################### following uses memory cache ##########################

        e_t_pre = OBS['pre_e_t'] # T x N x dim, e_t cache
        q_t_pre = OBS['pre_q_t'] # T x N x dim, q_t cache
        T, N, _ = e_t_pre.shape
        T = T + 1 # the new e_t
        e_t = torch.cat([e_t_pre, e_t], 0) # T x N x dim
        

        position_embedding = self.position_embedding(T).expand(N, -1, -1).permute(1, 0, 2) # T x N x dim
        e_t_e = e_t + position_embedding

        idx_c = torch.linspace(0, T - 1, T, dtype=torch.long, device=device).repeat(T, 1)
        idx_r = idx_c.permute(1, 0)
        src_mask = idx_r < idx_c

        e_t_tilde = self.ehe(e_t_e, src_mask) # T x N x dim

        q_t, _ = self.mha(e_t_tilde[-1].view(1, N, -1), encoded_target, encoded_target) # 1 x 1 * N x dim

        q_t = q_t.view(1, N, -1) # 1 x N x dim
        q_t = torch.cat([q_t_pre, q_t], 0) # T x N x dim

        

        tgt_mask = src_mask
        hidden_states = self.mtp(q_t, e_t_tilde, tgt_mask, src_mask) # T x N x dim

        attended_visual_features = torch.cat(
            [
                rgb_attn_embedding.permute(1,0,2),
                depth_attn_embedding.permute(1,0,2),
                observations['angle_features'].squeeze(0)
            ],
            dim=2,
        ) # 1 * N x M x dim

        x = hidden_states[-1]
        query_h = self.linear_ht(x).view(N, -1) # 1 * N x dim
        query_h = query_h.unsqueeze(2) # 1 * N x dim x 1
        logits_pano = torch.matmul(attended_visual_features, query_h).squeeze(2) # 1 * N x M
        logits = torch.cat([logits_pano, self.stop_linear(x)], dim=1)
        # print('logits',logits)
        pano_stop_distribution = CustomFixedCategorical(
            logits=logits
        )

        catted_features = torch.cat(
            [
                attended_visual_features,
                x.unsqueeze(1).repeat(1, attended_visual_features.size(1), 1),
            ],
            dim=2,
        )

        # ===========================
        #     Distance Prediction
        # ===========================

        if self.wypt_cfg.continuous_distance:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable1 = (
                self.wypt_cfg.max_distance_prediction
                - self.wypt_cfg.min_distance_prediction
            ) * distance_variable1 + self.wypt_cfg.min_distance_prediction

            distance_variable2 = (
                self.wypt_cfg.max_distance_var - self.wypt_cfg.min_distance_var
            ) * self.distance_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_distance_var
        else:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable2 = None

        # ===========================
        #      Offset Prediction
        # ===========================

        if self.wypt_cfg.continuous_offset:
            offset_variable1 = self.offset_scale * self.offset_linear(
                catted_features
            ).squeeze(2)
            offset_variable2 = (
                self.wypt_cfg.max_offset_var - self.wypt_cfg.min_offset_var
            ) * self.offset_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_offset_var
        else:
            offset_variable1 = self.offset_linear(catted_features).squeeze(2)
            offset_variable2 = None

        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        )
        # return hidden_states, e_t, q_t
    
    def forward(self, OBS, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [T * batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [T * batch_size x RGB_ENCODER.output_size]
        
        """
        # if 
        if 'pre_e_t' in OBS and 'pre_q_t' in OBS:
            return self.forward_single(OBS, prev_actions, masks) 

        return self.forward_seq(OBS, prev_actions, masks) 

class WaypointTF(WaypointPredictionNet):
    def __init__(self, observation_space: Space, model_config: Config):
        super().__init__(observation_space, model_config)
        self.compass_embedding_size = 128
        self.gps_embedding_size = 128
        self.object_num = 21
        self.target_token_size = model_config.RGB_ENCODER.output_size
        self.pe_max_length = 505
        self.pe_dim = self.target_token_size

        device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # self.linear_et = nn.Sequential(
        #     nn.Linear(model_config.RGB_ENCODER.output_size + PREV_ACTION_DIM, self.target_token_size),
        #     nn.ReLU(True)
        # )

        # visual_tf_layer = TFEL(self.target_token_size, nhead=1)
        # self.visual_tf = nn.TransformerEncoder(visual_tf_layer, 2)

        # main_tf_layer = TFEL(self.target_token_size, nhead=1)

        # self.main_tf = nn.TransformerEncoder(main_tf_layer, 2)

        # self.pe_dim = self.target_token_size
        # self.pe_max_length = 505

        # self.pe = {}
        

        ###
        self.audio_encoder = AudioCNN([65, 32, 2], model_config.RGB_ENCODER.output_size)

        # Init the Imagegoal encoder
        self.imagegoal_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=True,
        )
        self.linear_ig = nn.Sequential(
            nn.Linear(self.imagegoal_encoder.output_shape[0], self.target_token_size),
            nn.ReLU()
        )
        # ImageGoalEncoder(
        #     observation_space, self.target_token_size, device,
        #     spatial_output=True
        # )
        


        # Init the object category embedding
        self.object_embedding = nn.Embedding(self.object_num + 1, self.target_token_size)

        # Init the task embedding for task tokens
        self.task_embedding = nn.Embedding(4, self.target_token_size)

        self.rgb_spatial_attn = nn.ModuleDict({
            'vln': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0]),
            'van': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0]),
            'ign': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0]),
            'ogn': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0])
            }
        )
        self.depth_spatial_attn = nn.ModuleDict({
            'vln': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0]),
            'van': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0]),
            'ign': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0]),
            'ogn': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0]),
        })
        

        self.rgb_spatial_linear = nn.Sequential(
            nn.Linear(self.rgb_encoder.output_shape[0] * 16, self.target_token_size),
            nn.ReLU()
        )
        self.depth_spatial_linear = nn.Sequential(
            nn.Linear(self.depth_encoder.output_shape[0] * 16, self.target_token_size),
            nn.ReLU()
        )


        self.modal_type_embedding = nn.Embedding(3, self.target_token_size)

        self.pe = {}

        self.rgb_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )
        self.depth_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )
        self.audio_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )

        self.rgb_attn = nn.ModuleDict({
            'vln': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'van': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ign': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ogn': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        }) 
        self.depth_attn = nn.ModuleDict({
            'vln': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'van': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ign': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ogn': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        }) 
        # self.audio_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        self.audio_attn = nn.ModuleDict({
            'vln': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'van': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ign': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE),
            'ogn': nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        }) 
        

        target_tf_layer = TFEL(self.target_token_size, nhead=4)
        # self.target_tf = nn.TransformerEncoder(target_tf_layer, 2)
        self.target_tf = nn.ModuleDict({
            'vln' : nn.TransformerEncoder(target_tf_layer, 2),
            'van' : nn.TransformerEncoder(target_tf_layer, 2),
            'ign' : nn.TransformerEncoder(target_tf_layer, 2),
            'ogn' : nn.TransformerEncoder(target_tf_layer, 2)
        })

        msi_layer = TFEL(self.target_token_size, nhead=1)
        self.msi = nn.TransformerEncoder(msi_layer, 2)

        self.linear_et = nn.Linear(self.target_token_size * 3 + PREV_ACTION_DIM + GPS_FEATURE_SIZE, self.target_token_size)

        ehe_layer = TFEL(self.target_token_size, nhead=4)
        self.ehe = nn.TransformerEncoder(ehe_layer, 2)

        self.mha = nn.MultiheadAttention(self.target_token_size, 4)

        mtp_layer = TFDL(self.target_token_size, nhead=4)
        self.mtp_share = nn.TransformerDecoder(mtp_layer, 1)

        mtp_layer = TFEL(self.target_token_size, nhead=4)

        self.mtps = nn.ModuleDict({
            'vln' : nn.TransformerEncoder(mtp_layer, 2),
            'van' : nn.TransformerEncoder(mtp_layer, 2),
            'ign' : nn.TransformerEncoder(mtp_layer, 2),
            'ogn' : nn.TransformerEncoder(mtp_layer, 2)
        })

        self.linear_h_fuse = nn.Sequential(
            nn.Linear(
                self.target_token_size * 3,
                self.target_token_size
            ),
            nn.ReLU(True)
        )


        final_feature_size = model_config.RGB_ENCODER.output_size + model_config.DEPTH_ENCODER.output_size + ANGLE_FEATURE_SIZE
        self.linear_ht = nn.Sequential(
            nn.Linear(
                self.target_token_size,
                final_feature_size,
            ),
            nn.ReLU(True),
        )
        self.stop_linear = nn.Linear(self.target_token_size, 1)

        

        self.pe = {}

        ###

        self._init_pe()
        self.train()
        

    def _init_pe(self):
        self.embed = torch.zeros(self.pe_max_length, self.pe_dim).float().cuda(self.device)
        for p in range(self.pe_max_length):
            for i in range(self.pe_dim):
                if i % 2 == 0:
                    self.embed[p, i] = math.sin(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
                else:
                    self.embed[p, i] = math.cos(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
        

    def position_embedding(self, length):
        return self.embed[:length].detach()

    def forward(
        self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
    ) -> Tuple[
        CustomFixedCategorical,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Returns:
            pano_stop_distribution: [B, p+1] with range [0, inf]
            offsets: [B, p] with range: [-offset_scale, offset_scale]
            distances: [B, p] with range: [min_distance_prediction, max_distance_prediction]
            offsets_vars: [B, p] with range: [min_offset_var, max_offset_var]
            distances_vars: [B, p] with range: [min_distance_var, max_distance_var]
            x: [B, 512]
            rnn_states: [B, 512]
        """
        task = None
        if 'instruction' in observations:
            task = 'vln'
        elif 'spectrogram' in observations:
            task = 'van'
        elif 'imagegoal' in observations:
            task = 'ign'
        elif 'objectgoal' in observations:
            task = 'ogn'

        # assert "rgb" in observations
        # assert "depth" in observations
        # assert "instruction" in observations
        # assert "rgb_history" in observations
        # assert "depth_history" in observations
        assert "angle_features" in observations

        device = self.device
        # ===========================
        #  Single Modality Encoding
        # ===========================
        T, N, M, _, _, _= observations['rgb_features'].shape
        # T, N, M, _, _, _= observations['rgb'].shape
        # print('T',T, 'device', device)
        if 'instruction' not in observations:
        # if instruction_embedding is None:
            instruction_embedding = torch.zeros(T * N, 1, self.instruction_encoder.output_size, device=device)
            vln_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 0).detach()
        else:
            observations['instruction'] = wrap_helper(observations['instruction'])
            instruction_embedding = self.instruction_encoder(observations) # T * N x length x dim 
            instruction_embedding = instruction_embedding.permute(0, 2, 1)
            _, l, _ = instruction_embedding.shape
            vln_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 0)
            # print(instruction_embedding.shape)
            # pass
        
        if 'imagegoal' not in observations:
        # if imagegoal_embedding is None:
            imagegoal_embedding = torch.zeros(T * N, 16, self.target_token_size, device=device)
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 1).detach()
        else:
            observations['imagegoal'] = wrap_helper(observations['imagegoal'])
            self.imagegoal_encoder.eval()
            imagegoal_embedding = self.imagegoal_encoder({'rgb': observations['imagegoal']}) # T * N x dim x h x w
            imagegoal_embedding = torch.flatten(imagegoal_embedding, 2).permute(0,2,1) # T * N x 16 x dim
            # print(imagegoal_embedding.shape)
            imagegoal_embedding = self.linear_ig(imagegoal_embedding)
            _, l, _ = imagegoal_embedding.shape
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 1)
            
        
        if 'spectrogram' not in observations:
        # if audio_embedding is None:
            audio_embedding = torch.zeros(T * N, M, self.audio_encoder.output_size, device=device)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2).detach()
            
        else:
            observations['spectrogram'] = wrap_helper_pano(observations['spectrogram'])
            audio_embedding = self.audio_encoder(observations) # T * N * M x dim 
            audio_embedding = audio_embedding.view(T * N, M, -1)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            

        
        if 'objectgoal' not in observations:
            objectgoal_embedding = torch.zeros(T * N, 1, self.object_embedding.embedding_dim, device=device)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3).detach()
        else:
            objectgoal_embedding = self.object_embedding(observations['objectgoal'].long()).view(T * N, 1, -1)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
            
        t_mask = torch.cat([
            (instruction_embedding == 0.0).all(-1),
            (imagegoal_embedding == 0.0).all(-1),
            (audio_embedding == 0.0).all(-1),
            (objectgoal_embedding == 0.0).all(-1)
        ], dim=1
        ) # T * N x L

        TN, L = t_mask.size()
        attn_mask = t_mask.unsqueeze(1).repeat_interleave(L, dim=1) # T * N x L x L
        t_mask = t_mask.permute(1,0).unsqueeze(2) # L x T * N x 1

        # plus task embeddings
        instruction_embedding = instruction_embedding + vln_task_embedding
        
        imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        
        audio_embedding = audio_task_embedding + audio_embedding
        
        objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding

        # concat target tokens
        target_ins = torch.cat([instruction_embedding, imagegoal_embedding, audio_embedding, objectgoal_embedding], 1) #   T * N x length x dim

        target_ins = target_ins.permute(1, 0, 2) # length x T * N x dim

        # encoded_target = self.target_tf(target_ins, mask=attn_mask) # length x T * N x dim
        # # encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim
        # encoded_target_avg = (encoded_target * t_mask).sum(0) / t_mask.sum(0)
        encoded_target = self.target_tf[task](target_ins) # length x T * N x dim
        encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim

        
        rgb_features = observations['rgb_features']
        rgb_features = wrap_helper_pano(rgb_features) # T * N * M x H x W x D
        rgb_embedding = self.rgb_encoder({"rgb_features": rgb_features}) # T * N * M x D x H x W 
        rgb_embedding = torch.flatten(
            rgb_embedding.view(T * N * M, *rgb_embedding.shape[1:]), 2
        )

        depth_features = observations['depth_features']
        depth_features = wrap_helper_pano(depth_features)
        depth_embedding = self.depth_encoder({"depth_features": depth_features})
        depth_embedding = torch.flatten(
            depth_embedding.view(T * N * M, *depth_embedding.shape[1:]),
            2,
        )
        # print(rgb_embedding.shape, depth_embedding.shape)


        if len(prev_actions["pano"].shape) == 1:
            for k in prev_actions:
                prev_actions[k] = prev_actions[k].unsqueeze(1)

        prev_actions = (
            torch.cat(
                [
                    self._map_pano_to_heading_features(prev_actions["pano"]),
                    self.offset_to_continuous(prev_actions["offset"]),
                    self.distance_to_continuous(prev_actions["distance"]),
                ],
                dim=1,
            ).float()
            * masks
        )

        # compute visual spatial attention
        encoded_target_avg_M = encoded_target_avg.repeat_interleave(M).view(T * N, -1, M).permute(0,2, 1).reshape(T * N * M, -1)

        rgb_attn_embedding, _ = self.rgb_spatial_attn[task](encoded_target_avg_M.unsqueeze(0), rgb_embedding.permute(2, 0, 1), rgb_embedding.permute(2, 0, 1)) # T * N * M x dim
        rgb_attn_embedding = rgb_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim


        depth_attn_embedding, _ = self.depth_spatial_attn[task](encoded_target_avg_M.unsqueeze(0), depth_embedding.permute(2, 0, 1), depth_embedding.permute(2, 0, 1)) # T * N * M x dim
        depth_attn_embedding = depth_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim
        # print(rgb_attn_embedding.shape, depth_attn_embedding.shape)
        
        # rgb_attn_embedding = self.rgb_spatial_linear(rgb_embedding.view(T * N * M, -1)) # T * N * M x dim
        # rgb_attn_embedding = rgb_attn_embedding.view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # depth_attn_embedding = self.depth_spatial_linear(depth_embedding.view(T * N * M, -1)) # T * N * M x dim
        # depth_attn_embedding = depth_attn_embedding.view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # print(rgb_attn_embedding.shape, depth_attn_embedding.shape)


        # compute panoramic sensory attention
        rgb_attn_embedding_with_orien = torch.cat([rgb_attn_embedding, observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)
        depth_attn_embedding_with_orien = torch.cat([depth_attn_embedding, observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)
        audio_embedding_with_orien = torch.cat([audio_embedding.permute(1, 0, 2), observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)

        attended_rgb, _ = self.rgb_attn[task](encoded_target_avg.unsqueeze(0), rgb_attn_embedding_with_orien, rgb_attn_embedding_with_orien)
        attended_rgb = attended_rgb.squeeze(0) # T * N x dim

        attended_depth, _ = self.depth_attn[task](encoded_target_avg.unsqueeze(0), depth_attn_embedding_with_orien, depth_attn_embedding_with_orien)
        attended_depth = attended_depth.squeeze(0) # T * N x dim

        attended_audio, _ = self.audio_attn[task](encoded_target_avg.unsqueeze(0), audio_embedding_with_orien, audio_embedding_with_orien)
        attended_audio = attended_audio.squeeze(0) # T * N x dim

        # attended_rgb = self.rgb_pano_linear(rgb_attn_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))

        # attended_depth = self.depth_pano_linear(depth_attn_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))

        # attended_audio = self.audio_pano_linear(audio_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))


        # construct o_t token
        o_t = torch.cat([attended_rgb, attended_depth, attended_audio], -1) # T * N x 3 * dim

        gps = observations['globalgps'].view(T * N, -1)

        e_t = self.linear_et(torch.cat([o_t, prev_actions, gps], -1)) # T * N x dim

        if 'pre_e_t' in observations:
            pre_e_t = observations['pre_e_t']
            e_t = torch.cat([pre_e_t, e_t.view(1, N, -1)], 0) # T x N x dim
        else:
            e_t = e_t.view(T, N, -1)

        t, _, _ = e_t.size()
        position_embedding = self.position_embedding(t).expand(N, -1, -1).permute(1, 0, 2) # T x N x dim
        e_t_e = e_t + position_embedding

        idx_c = torch.linspace(0, t - 1, t, dtype=torch.long, device=self.device).repeat(t, 1)
        idx_r = idx_c.permute(1, 0)
        src_mask = idx_r < idx_c
        
        e_t_tilde = self.ehe(e_t_e, src_mask) # T x N x dim

        encoded_target = encoded_target.permute(1, 0, 2) # length x T * N x dim
        # print('e_t_tilde', e_t_tilde.shape)
        # print('encoded_target',encoded_target.shape)
        # print('========')
        

        if 'pre_q_t' in observations:
            q_t, _ = self.mha(e_t_tilde[-1].view(1, T * N, -1), encoded_target, encoded_target) # 1 x T * N x dim
            pre_q_t = observations['pre_q_t']
            q_t = torch.cat([pre_q_t, q_t.view(1, N, -1)], 0) # T x N x dim
        else:
            q_t, _ = self.mha(e_t_tilde.view(1, T * N, -1), encoded_target, encoded_target) # 1 x T * N x dim
            q_t = q_t.view(T, N, -1) # T x N x dim
        
        q_t_e = q_t + position_embedding

        tgt_mask = src_mask
        hidden_states = self.mtp_share(q_t_e, e_t_tilde, tgt_mask, src_mask) # T x N x dim
        
        
        hidden_states = self.mtps[task](hidden_states, src_mask)

        # hidden_states = self.linear_h_fuse(torch.cat([q_t_e, hidden_states, e_t_tilde], -1))
        # hidden_states = self.linear_h_fuse(torch.cat([q_t_e, e_t_tilde], -1))

        # print(q_t_e.shape, e_t_tilde.shape)

        attended_visual_features = torch.cat(
            [
                rgb_attn_embedding.permute(1,0,2),
                depth_attn_embedding.permute(1,0,2),
                observations['angle_features'].view(T * N, M, -1)
            ],
            dim=2,
        ) # T * N x M x dim
        
        if 'pre_e_t' in observations:
            x = hidden_states[-1].view(T * N, -1) # T * N x dim
        else:
            x = hidden_states.view(T * N, -1) # T * N x dim
        query_h = self.linear_ht(x) # T * N x dim
        query_h = query_h.unsqueeze(2) # T * N x dim x 1
        logits = torch.matmul(attended_visual_features, query_h).squeeze(2) # T * N x M
        
        # if not 'pre_e_t' in observations:
        #     print('logits',logits, self.stop_linear(x))
        try:
            pano_stop_distribution = CustomFixedCategorical(
                logits=torch.cat([logits, self.stop_linear(x)], dim=1)
            )
        except:
            
            assert False, 'error'
        
            
        catted_features = torch.cat(
            [
                attended_visual_features,
                x.unsqueeze(1).repeat(1, attended_visual_features.size(1), 1),
            ],
            dim=2,
        )

        # ===========================
        #     Distance Prediction
        # ===========================

        if self.wypt_cfg.continuous_distance:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable1 = (
                self.wypt_cfg.max_distance_prediction
                - self.wypt_cfg.min_distance_prediction
            ) * distance_variable1 + self.wypt_cfg.min_distance_prediction

            distance_variable2 = (
                self.wypt_cfg.max_distance_var - self.wypt_cfg.min_distance_var
            ) * self.distance_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_distance_var
        else:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable2 = None

        # ===========================
        #      Offset Prediction
        # ===========================

        if self.wypt_cfg.continuous_offset:
            offset_variable1 = self.offset_scale * self.offset_linear(
                catted_features
            ).squeeze(2)
            offset_variable2 = (
                self.wypt_cfg.max_offset_var - self.wypt_cfg.min_offset_var
            ) * self.offset_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_offset_var
        else:
            offset_variable1 = self.offset_linear(catted_features).squeeze(2)
            offset_variable2 = None

        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        )

class BackBone(WaypointPredictionNet):
    def __init__(self, observation_space: Space, model_config: Config):
        super().__init__(observation_space, model_config)
        self.compass_embedding_size = 128
        self.gps_embedding_size = 128
        self.object_num = 21
        self.target_token_size = model_config.RGB_ENCODER.output_size
        self.pe_max_length = 505
        self.pe_dim = self.target_token_size

        device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # self.linear_et = nn.Sequential(
        #     nn.Linear(model_config.RGB_ENCODER.output_size + PREV_ACTION_DIM, self.target_token_size),
        #     nn.ReLU(True)
        # )

        # visual_tf_layer = TFEL(self.target_token_size, nhead=1)
        # self.visual_tf = nn.TransformerEncoder(visual_tf_layer, 2)

        # main_tf_layer = TFEL(self.target_token_size, nhead=1)

        # self.main_tf = nn.TransformerEncoder(main_tf_layer, 2)

        # self.pe_dim = self.target_token_size
        # self.pe_max_length = 505

        # self.pe = {}
        

        ###
        self.audio_encoder = AudioCNN([65, 32, 2], model_config.RGB_ENCODER.output_size)

        # Init the Imagegoal encoder
        self.imagegoal_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=True,
        )
        self.linear_ig = nn.Sequential(
            nn.Linear(self.imagegoal_encoder.output_shape[0], self.target_token_size),
            nn.ReLU()
        )
        # ImageGoalEncoder(
        #     observation_space, self.target_token_size, device,
        #     spatial_output=True
        # )
        


        # Init the object category embedding
        self.object_embedding = nn.Embedding(self.object_num + 1, self.target_token_size)

        # Init the task embedding for task tokens
        self.task_embedding = nn.Embedding(4, self.target_token_size)

        self.rgb_spatial_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.rgb_encoder.output_shape[0], vdim=self.rgb_encoder.output_shape[0])
        self.depth_spatial_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.depth_encoder.output_shape[0], vdim=self.depth_encoder.output_shape[0])

        self.rgb_spatial_linear = nn.Sequential(
            nn.Linear(self.rgb_encoder.output_shape[0] * 16, self.target_token_size),
            nn.ReLU()
        )
        self.depth_spatial_linear = nn.Sequential(
            nn.Linear(self.depth_encoder.output_shape[0] * 16, self.target_token_size),
            nn.ReLU()
        )


        self.modal_type_embedding = nn.Embedding(3, self.target_token_size)

        self.pe = {}

        self.rgb_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )
        self.depth_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )
        self.audio_pano_linear = nn.Sequential(
            nn.Linear((self.target_token_size + ANGLE_FEATURE_SIZE) * 12, self.target_token_size),
            nn.ReLU()
        )

        self.rgb_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        self.depth_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        self.audio_attn = nn.MultiheadAttention(self.target_token_size, 1, kdim=self.target_token_size + ANGLE_FEATURE_SIZE, vdim=self.target_token_size + ANGLE_FEATURE_SIZE)
        

        target_tf_layer = TFEL(self.target_token_size, nhead=1)
        self.target_tf = nn.TransformerEncoder(target_tf_layer, 2)

        msi_layer = TFEL(self.target_token_size, nhead=1)
        self.msi = nn.TransformerEncoder(msi_layer, 2)

        self.linear_et = nn.Linear(self.target_token_size * 3 + PREV_ACTION_DIM + GPS_FEATURE_SIZE, self.target_token_size)

        ehe_layer = TFEL(self.target_token_size, nhead=4)
        self.ehe = nn.TransformerEncoder(ehe_layer, 2)

        self.mha = nn.MultiheadAttention(self.target_token_size, 4)

        mtp_layer = TFDL(self.target_token_size, nhead=4)
        self.mtp = nn.TransformerDecoder(mtp_layer, 1)
        

        self.pe = {}

        ###

        self._init_pe()
        self.train()     

    def _init_pe(self):
        self.embed = torch.zeros(self.pe_max_length, self.pe_dim).float().cuda(self.device)
        for p in range(self.pe_max_length):
            for i in range(self.pe_dim):
                if i % 2 == 0:
                    self.embed[p, i] = math.sin(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
                else:
                    self.embed[p, i] = math.cos(1.0 * p / 10000 ** (1.0 * i / self.pe_dim))
        

    def position_embedding(self, length):
        return self.embed[:length].detach()

    def forward(
        self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
    ) -> Tuple[
        CustomFixedCategorical,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Returns:
            pano_stop_distribution: [B, p+1] with range [0, inf]
            offsets: [B, p] with range: [-offset_scale, offset_scale]
            distances: [B, p] with range: [min_distance_prediction, max_distance_prediction]
            offsets_vars: [B, p] with range: [min_offset_var, max_offset_var]
            distances_vars: [B, p] with range: [min_distance_var, max_distance_var]
            x: [B, 512]
            rnn_states: [B, 512]
        """
        # assert "rgb" in observations
        # assert "depth" in observations
        # assert "instruction" in observations
        # assert "rgb_history" in observations
        # assert "depth_history" in observations
        assert "angle_features" in observations

        device = self.device
        # ===========================
        #  Single Modality Encoding
        # ===========================
        T, N, M, _, _, _= observations['rgb_features'].shape
        # T, N, M, _, _, _= observations['rgb'].shape
        # print('T',T, 'device', device)
        if 'instruction' not in observations:
        # if instruction_embedding is None:
            instruction_embedding = torch.zeros(T * N, 1, self.instruction_encoder.output_size, device=device)
            vln_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 0)
        else:
            observations['instruction'] = wrap_helper(observations['instruction'])
            instruction_embedding = self.instruction_encoder(observations) # T * N x length x dim 
            instruction_embedding = instruction_embedding.permute(0, 2, 1)
            _, l, _ = instruction_embedding.shape
            vln_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 0)
            # print(instruction_embedding.shape)
            # pass
        
        if 'imagegoal' not in observations:
        # if imagegoal_embedding is None:
            imagegoal_embedding = torch.zeros(T * N, 16, self.target_token_size, device=device)
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 1)
        else:
            observations['imagegoal'] = wrap_helper(observations['imagegoal'])
            imagegoal_embedding = self.imagegoal_encoder({'rgb': observations['imagegoal']}) # T * N x dim x h x w
            imagegoal_embedding = torch.flatten(imagegoal_embedding, 2).permute(0,2,1) # T * N x 16 x dim
            # print(imagegoal_embedding.shape)
            imagegoal_embedding = self.linear_ig(imagegoal_embedding)
            _, l, _ = imagegoal_embedding.shape
            imagegoal_task_embedding = self.task_embedding(torch.ones(T * N, l, device=device).long() * 1)
            
        
        if 'spectrogram' not in observations:
        # if audio_embedding is None:
            audio_embedding = torch.zeros(T * N, M, self.audio_encoder.output_size, device=device)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            
        else:
            observations['spectrogram'] = wrap_helper_pano(observations['spectrogram'])
            audio_embedding = self.audio_encoder(observations) # T * N * M x dim 
            audio_embedding = audio_embedding.view(T * N, M, -1)
            audio_task_embedding = self.task_embedding(torch.ones(T * N, M, device=device).long() * 2)
            

        
        if 'objectgoal' not in observations:
            objectgoal_embedding = torch.zeros(T * N, 1, self.object_embedding.embedding_dim, device=device)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
        else:
            objectgoal_embedding = self.object_embedding(observations['objectgoal'].long()).view(T * N, 1, -1)
            objectgoal_task_embedding = self.task_embedding(torch.ones(T * N, 1, device=device).long() * 3)
            
        t_mask = torch.cat([
            (instruction_embedding == 0.0).all(-1),
            (imagegoal_embedding == 0.0).all(-1),
            (audio_embedding == 0.0).all(-1),
            (objectgoal_embedding == 0.0).all(-1)
        ], dim=1
        ) # T * N x L

        TN, L = t_mask.size()
        attn_mask = t_mask.unsqueeze(1).repeat_interleave(L, dim=1) # T * N x L x L
        t_mask = t_mask.permute(1,0).unsqueeze(2) # L x T * N x 1

        # plus task embeddings
        instruction_embedding = instruction_embedding + vln_task_embedding
        
        imagegoal_embedding = imagegoal_embedding + imagegoal_task_embedding
        
        audio_embedding = audio_task_embedding + audio_embedding
        
        objectgoal_embedding = objectgoal_embedding + objectgoal_task_embedding

        # concat target tokens
        target_ins = torch.cat([instruction_embedding, imagegoal_embedding, audio_embedding, objectgoal_embedding], 1) #   T * N x length x dim

        target_ins = target_ins.permute(1, 0, 2) # length x T * N x dim

        # encoded_target = self.target_tf(target_ins, mask=attn_mask) # length x T * N x dim
        # # encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim
        # encoded_target_avg = (encoded_target * t_mask).sum(0) / t_mask.sum(0)
        encoded_target = self.target_tf(target_ins) # length x T * N x dim
        encoded_target_avg = torch.mean(encoded_target, 0) # T * N x dim

        
        rgb_features = observations['rgb_features']
        rgb_features = wrap_helper_pano(rgb_features) # T * N * M x H x W x D
        rgb_embedding = self.rgb_encoder({"rgb_features": rgb_features}) # T * N * M x D x H x W 
        rgb_embedding = torch.flatten(
            rgb_embedding.view(T * N * M, *rgb_embedding.shape[1:]), 2
        )

        depth_features = observations['depth_features']
        depth_features = wrap_helper_pano(depth_features)
        depth_embedding = self.depth_encoder({"depth_features": depth_features})
        depth_embedding = torch.flatten(
            depth_embedding.view(T * N * M, *depth_embedding.shape[1:]),
            2,
        )
        # print(rgb_embedding.shape, depth_embedding.shape)


        if len(prev_actions["pano"].shape) == 1:
            for k in prev_actions:
                prev_actions[k] = prev_actions[k].unsqueeze(1)

        prev_actions = (
            torch.cat(
                [
                    self._map_pano_to_heading_features(prev_actions["pano"]),
                    self.offset_to_continuous(prev_actions["offset"]),
                    self.distance_to_continuous(prev_actions["distance"]),
                ],
                dim=1,
            ).float()
            * masks
        )

        # compute visual spatial attention
        encoded_target_avg_M = encoded_target_avg.repeat_interleave(M).view(T * N, -1, M).permute(0,2, 1).reshape(T * N * M, -1)

        rgb_attn_embedding, _ = self.rgb_spatial_attn(encoded_target_avg_M.unsqueeze(0), rgb_embedding.permute(2, 0, 1), rgb_embedding.permute(2, 0, 1)) # T * N * M x dim
        rgb_attn_embedding = rgb_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim


        depth_attn_embedding, _ = self.depth_spatial_attn(encoded_target_avg_M.unsqueeze(0), depth_embedding.permute(2, 0, 1), depth_embedding.permute(2, 0, 1)) # T * N * M x dim
        depth_attn_embedding = depth_attn_embedding.squeeze(0).view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim
        # print(rgb_attn_embedding.shape, depth_attn_embedding.shape)
        
        # rgb_attn_embedding = self.rgb_spatial_linear(rgb_embedding.view(T * N * M, -1)) # T * N * M x dim
        # rgb_attn_embedding = rgb_attn_embedding.view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # depth_attn_embedding = self.depth_spatial_linear(depth_embedding.view(T * N * M, -1)) # T * N * M x dim
        # depth_attn_embedding = depth_attn_embedding.view(T * N, M, -1).permute(1, 0, 2) # M x T * N x dim

        # print(rgb_attn_embedding.shape, depth_attn_embedding.shape)


        # compute panoramic sensory attention
        rgb_attn_embedding_with_orien = torch.cat([rgb_attn_embedding, observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)
        depth_attn_embedding_with_orien = torch.cat([depth_attn_embedding, observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)
        audio_embedding_with_orien = torch.cat([audio_embedding.permute(1, 0, 2), observations['angle_features'].view(T * N, M, -1).permute(1, 0, 2)], -1)

        attended_rgb, _ = self.rgb_attn(encoded_target_avg.unsqueeze(0), rgb_attn_embedding_with_orien, rgb_attn_embedding_with_orien)
        attended_rgb = attended_rgb.squeeze(0) # T * N x dim

        attended_depth, _ = self.depth_attn(encoded_target_avg.unsqueeze(0), depth_attn_embedding_with_orien, depth_attn_embedding_with_orien)
        attended_depth = attended_depth.squeeze(0) # T * N x dim

        attended_audio, _ = self.audio_attn(encoded_target_avg.unsqueeze(0), audio_embedding_with_orien, audio_embedding_with_orien)
        attended_audio = attended_audio.squeeze(0) # T * N x dim

        # attended_rgb = self.rgb_pano_linear(rgb_attn_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))

        # attended_depth = self.depth_pano_linear(depth_attn_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))

        # attended_audio = self.audio_pano_linear(audio_embedding_with_orien.permute(1, 0, 2).reshape(T * N, -1))


        # construct o_t token
        o_t = torch.cat([attended_rgb, attended_depth, attended_audio], -1) # T * N x 3 * dim

        gps = observations['globalgps'].view(T * N, -1)

        e_t = self.linear_et(torch.cat([o_t, prev_actions, gps], -1)) # T * N x dim

        if 'pre_e_t' in observations:
            pre_e_t = observations['pre_e_t']
            e_t = torch.cat([pre_e_t, e_t.view(1, N, -1)], 0) # T x N x dim
        else:
            e_t = e_t.view(T, N, -1)

        t, _, _ = e_t.size()
        position_embedding = self.position_embedding(t).expand(N, -1, -1).permute(1, 0, 2) # T x N x dim
        e_t_e = e_t + position_embedding

        idx_c = torch.linspace(0, t - 1, t, dtype=torch.long, device=self.device).repeat(t, 1)
        idx_r = idx_c.permute(1, 0)
        src_mask = idx_r < idx_c
        
        e_t_tilde = self.ehe(e_t_e, src_mask) # T x N x dim

        encoded_target = encoded_target.permute(1, 0, 2) # length x T * N x dim


        attended_visual_features = torch.cat(
            [
                rgb_attn_embedding.permute(1,0,2),
                depth_attn_embedding.permute(1,0,2),
                observations['angle_features'].view(T * N, M, -1)
            ],
            dim=2,
        ) # T * N x M x dim
        

        if 'pre_q_t' in observations:
            q_t, _ = self.mha(e_t_tilde[-1].view(1, T * N, -1), encoded_target, encoded_target) # 1 x T * N x dim
            pre_q_t = observations['pre_q_t']
            q_t = torch.cat([pre_q_t, q_t.view(1, N, -1)], 0) # T x N x dim
        else:
            q_t, _ = self.mha(e_t_tilde.view(1, T * N, -1), encoded_target, encoded_target) # 1 x T * N x dim
            q_t = q_t.view(T, N, -1) # T x N x dim
        
        q_t_e = q_t + position_embedding

        tgt_mask = src_mask
        hidden_states = self.mtp(q_t_e, e_t_tilde, tgt_mask, src_mask) # T x N x dim
        
        return hidden_states, attended_visual_features, e_t, q_t, src_mask
        
class MThead(nn.Module):
    def __init__(self, model_config: Config):
        super(MThead, self).__init__()
        self.model_config = model_config
        self.wypt_cfg = model_config.WAYPOINT
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        self._num_panos = self.model_config.num_panos


        self.target_token_size = model_config.RGB_ENCODER.output_size

        mtp_layer = TFEL(self.target_token_size, nhead=4)
        self.mtps = nn.TransformerEncoder(mtp_layer, 2)

        final_feature_size = model_config.RGB_ENCODER.output_size + model_config.DEPTH_ENCODER.output_size + ANGLE_FEATURE_SIZE
        self.linear_ht = nn.Sequential(
            nn.Linear(
                self.target_token_size,
                final_feature_size,
            ),
            nn.ReLU(True),
        )
        self.stop_linear = nn.Linear(self.target_token_size, 1)

        in_dim = self._hidden_size + final_feature_size
        self._init_distance_linear(in_dim, final_feature_size)
        self._init_offset_linear(in_dim, final_feature_size)

        self.train()

    def _init_distance_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the distance output to be either discrete or
        continuous. If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_distance:
            self.distance_linear = nn.Sequential(
                nn.Linear(in_dim, 1), nn.Sigmoid()
            )
            self.distance_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.distance_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_distances
            )

    def _init_offset_linear(
        self, in_dim: int, final_feature_size: int
    ) -> None:
        """Initialize the offset output to be either discrete or continuous.
        If continuous, the distribution is TruncatedNormal.
        """
        if self.wypt_cfg.continuous_offset:
            self.offset_linear = nn.Sequential(
                nn.Linear(in_dim, 1),
                TemperatureTanh(temperature=self.wypt_cfg.offset_temperature),
            )
            self.offset_scale = np.pi / self._num_panos
            self.offset_var_linear = nn.Sequential(
                nn.Linear(self._hidden_size + final_feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.offset_linear = nn.Linear(
                in_dim, self.wypt_cfg.discrete_offsets
            )

    def forward(self, args):
        hidden_states, (attended_visual_features, src_mask, single_output) = args
        T, N, _ = hidden_states.shape
        hidden_states = self.mtps(hidden_states, src_mask) # T x N x dim
        if single_output:
            x = hidden_states[-1].view(N, -1) # T * N x dim
        else:
            x = hidden_states.view(T * N, -1) # T * N x dim
        query_h = self.linear_ht(x) # T * N x dim
        query_h = query_h.unsqueeze(2) # T * N x dim x 1
        logits = torch.matmul(attended_visual_features, query_h).squeeze(2) # T * N x M
        
        # print('logits',logits, self.stop_linear(x))
        try:
            pano_stop_distribution = CustomFixedCategorical(
                logits=torch.cat([logits, self.stop_linear(x)], dim=1)
            )
        except:
            
            assert False, 'error'
        
            
        catted_features = torch.cat(
            [
                attended_visual_features,
                x.unsqueeze(1).repeat(1, attended_visual_features.size(1), 1),
            ],
            dim=2,
        )

        # ===========================
        #     Distance Prediction
        # ===========================

        if self.wypt_cfg.continuous_distance:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable1 = (
                self.wypt_cfg.max_distance_prediction
                - self.wypt_cfg.min_distance_prediction
            ) * distance_variable1 + self.wypt_cfg.min_distance_prediction

            distance_variable2 = (
                self.wypt_cfg.max_distance_var - self.wypt_cfg.min_distance_var
            ) * self.distance_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_distance_var
        else:
            distance_variable1 = self.distance_linear(catted_features)
            distance_variable1 = distance_variable1.squeeze(2)
            distance_variable2 = None

        # ===========================
        #      Offset Prediction
        # ===========================

        if self.wypt_cfg.continuous_offset:
            offset_variable1 = self.offset_scale * self.offset_linear(
                catted_features
            ).squeeze(2)
            offset_variable2 = (
                self.wypt_cfg.max_offset_var - self.wypt_cfg.min_offset_var
            ) * self.offset_var_linear(catted_features).squeeze(
                2
            ) + self.wypt_cfg.min_offset_var
        else:
            offset_variable1 = self.offset_linear(catted_features).squeeze(2)
            offset_variable2 = None

        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x
        )

class RotoGradNav(RotoGrad):

    def forward(self, x: Any) -> Sequence[Any]:
        """Forwards the input `x` through the backbone and all heads, returning a list with all the task predictions.
        It can be thought as something similar to:

        .. code-block:: python

            preds = []
            z = backbone(x)
            for R_i, head in zip(rotations, heads):
                z_i = rotate(R_i, z)
                preds.append(head(z_i))
            return preds

        """
        observations, prev_actions, masks = x
        preds = []
        hidden_states, attended_visual_features, e_t, q_t, src_mask = self.backbone(observations, prev_actions, masks)

        rep = hidden_states # T x N x dim

        if self.training:
            self.rep = rep

        task_id = None
        if 'instruction' in observations:
            task_id = 0
        elif 'spectrogram' in observations:
            task_id = 1
        elif 'imagegoal' in observations:
            task_id = 2
        elif 'objectgoal' in observations:
            task_id = 3

        head = self.heads[task_id]
        rep_i = rep.detach().clone()
        rep_i.requires_grad = True
        rep_i.register_hook(self._hook(task_id)) # z

        if 'pre_e_t' in observations:
            single_output = True
        else:
            single_output = False

        out_i = head((rep_i, attended_visual_features.detach().clone(), src_mask, single_output))

        (pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x
         ) = out_i
        
        return (pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        )

    def backward(self, loss, task_id, **kwargs):
        assert self.training, 'Backward should only be called when training'
        if self.iteration_counter == 0 or self.iteration_counter == self.burn_in_period:
            self.initial_losses[task_id] = loss.item()

        self.iteration_counter += 1

        loss.backward(**kwargs) # backward the loss for all heads, since the representation is detached for the heads, the following code backward the gradient for the representation

        # self.rep.backward(self.original_grads[task_id])
        self.rep.backward(self._rep_grad(task_id))

    def _rep_grad(self, task_id):
        grad_norms = []
        for i, grad in enumerate(self.original_grads):
            if not i == task_id and grad is None:
                grad = torch.zeros_like(self.original_grads[task_id])
            
            grad_norms.append(torch.norm(grad, keepdim=True).clamp_min(1e-15))

        # multi task grads
        grad_norms = torch.stack(grad_norms, 0)
        world_size = distrib.get_world_size()
        process_num = world_size / self.num_tasks
        distrib.all_reduce(grad_norms)
        grad_norms = grad_norms / process_num


        if self.initial_grads is None or self.counter % self.burn_in_period == 0:
            self.initial_grads = grad_norms
        self.counter += 1

        conv_ratios = [x / y for x, y, in zip(grad_norms, self.initial_grads)]
        alphas = [x / torch.clamp(sum(conv_ratios), 1e-15) for x in conv_ratios]

        weighted_sum_norms = sum([a * g for a, g in zip(alphas, grad_norms)])
        # grads = [g / n * weighted_sum_norms for g, n in zip(self.original_grads, grad_norms)]
        grads = self.original_grads[task_id] / grad_norms[task_id] * weighted_sum_norms
        # return sum(grads)
        return grads * world_size
    

    def _rep_grad_bk(self, task_id):
        old_grads = self.original_grads  # these grads are already rotated, we have to recover the originals, the grads of z.
        # with torch.no_grad():
        #     grads = [rotate(g, R) for g, R in zip(grads, self.rotation)]
        #
        grads = self.grads # the grads of r.

        assert distrib.is_initialized(), 'must be used for distributed multitask training'
        ### The following code is only for calc the gradient of the rotation matrix
        if distrib.is_initialized():
            world_size = distrib.get_world_size()
            process_num = world_size / self.num_tasks
            old_grads_list = []
            for i, g in enumerate(old_grads): # sum the old grads of all process
                if i == task_id:
                    old_grads_list.append(g.clone())
                else:
                    old_grads_list.append(torch.zeros_like(old_grads[task_id]))

            old_grads = torch.stack(old_grads_list, 0)
            distrib.all_reduce(old_grads)
            old_grads /= process_num

            grads_list = []
            for i, g in enumerate(grads): # sum the old grads of all process
                if i == task_id:
                    grads_list.append(g.clone())
                else:
                    grads_list.append(torch.zeros_like(grads[task_id]))

            grads = torch.stack(grads_list, 0)
            distrib.all_reduce(grads)
            grads /= process_num

            print(old_grads.shape, grads.shape)
        
        # Compute the reference vector
        mean_grad = sum([g for g in old_grads]).detach().clone() / len(grads)
        mean_norm = mean_grad.norm(p=2)
        old_grads2 = [g * divide(mean_norm, g.norm(p=2)) for g in old_grads] 
        mean_grad = sum([g for g in old_grads2]).detach().clone() / len(grads) # the reference vector
        
        for i, grad in enumerate(grads):
            R = self.rotation[i]
            shape = mean_grad.shape
            mean_grad = mean_grad.reshape(-1, shape[-1])
            mean_grad_R = rotate(mean_grad, R, self.latent_size).reshape(*shape)
            loss_rotograd = mean_grad_R - grad
            loss_rotograd = loss_rotograd.reshape(-1, self.latent_size)
            loss_rotograd = torch.einsum('bi,bi->b', loss_rotograd, loss_rotograd)
            loss_rotograd.mean().backward()

        ### The following code is for grad magnitude norm
        grad_norms = [torch.norm(g, keepdim=True).clamp_min(1e-15) for g in old_grads]

        if self.initial_grads is None or self.counter % self.burn_in_period == 0:
            self.initial_grads = grad_norms
        self.counter += 1

        conv_ratios = [x / y for x, y, in zip(grad_norms, self.initial_grads)]
        alphas = [x / torch.clamp(sum(conv_ratios), 1e-15) for x in conv_ratios]

        weighted_sum_norms = sum([a * g for a, g in zip(alphas, grad_norms)])
        grads = [g / n * weighted_sum_norms for g, n in zip(old_grads, grad_norms)]

        return sum(grads)
        

class NavNet(nn.Module):
    def __init__(self, observation_space: Space, model_config: Config):
        super(NavNet, self).__init__()
        self.backbone = BackBone(observation_space, model_config)
        self.heads = nn.ModuleList([
             MThead(model_config), # 0 : vln
             MThead(model_config), # 1 : van
             MThead(model_config), # 2 : ign
             MThead(model_config), # 3 : ogn
        ])
        self.model = RotoGradNav(self.backbone, [h for h in self.heads], self.backbone.target_token_size)
    
    @property
    def rgb_encoder(self):
        return self.backbone.rgb_encoder
    
    @property
    def depth_encoder(self):
        return self.backbone.depth_encoder

    @property
    def output_size(self):
        return self.backbone._hidden_size

    @property
    def distance_to_continuous(self):
        return self.backbone.distance_to_continuous
    
    @property
    def offset_to_continuous(self):
        return self.backbone.offset_to_continuous

    def forward(self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor):
        
        res = self.model([observations, prev_actions, masks])
        (pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
         ) = res
        return (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        )

@baseline_registry.register_policy
class WaypointPolicyTF(WaypointPolicy, Policy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        Policy.__init__(
            self,
            WaypointTF(
                observation_space=observation_space,
                model_config=model_config,
            ),
            1,  # ignore the action dimension
        )

        self._config = model_config
        self.wypt_cfg = model_config.WAYPOINT
        self._offset_limit = np.pi / self._config.num_panos
    
    def act(
        self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[
        Tensor,
        List[Dict[str, Any]],
        Dict[str, Tensor],
        Dict[str, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        # try:
        output = self.net(
            observations,
            prev_actions,
            masks,
        )
        # except:
        #     for key in observations:
        #         print('key', observations[key].shape, torch.isnan(observations[key]).any())
        #     print('rnn', torch.isnan(rnn_states).any())
        #     print('pre_action', prev_actions)
        #     traceback.print_exc()
            
        (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        ) = output

        pano_stop = (
            pano_stop_distribution.mode()
            if deterministic
            else pano_stop_distribution.sample()
        )
        stop = (pano_stop == self._config.num_panos).to(torch.uint8)
        pano = pano_stop % self._config.num_panos
        # print('pano',pano)
        # print('distance_variable1', distance_variable1)
        # print('distance_variable2', distance_variable2)
        # print('offset_variable1', offset_variable1)
        # print('offset_variable2', offset_variable2)
        distance_distribution = self._create_distance_distribution(
            distance_variable1, distance_variable2, pano
        )
        offset_distribution = self._create_offset_distribution(
            offset_variable1, offset_variable2, pano
        )

        (
            distance,
            action_distance,
            distance_log_probs,
            dist_var,
            dist_mode,
        ) = self.get_distance_prediction(distance_distribution, deterministic)
        (
            offset,
            action_offset,
            offset_log_probs,
            ofst_var,
            ofst_mode,
        ) = self.get_offset_prediction(offset_distribution, deterministic)

        # print(action_distance, distance_log_probs, action_offset, offset_log_probs)

        actions = []
        radians_per_pano = 2 * np.pi / self._config.num_panos
        theta = (pano * radians_per_pano + action_offset) % (2 * np.pi)
        for i in range(pano_stop.shape[0]):
            if stop[i]:
                actions.append({"action": "STOP"})
            else:
                actions.append(
                    {
                        "action": {
                            "action": "GO_TOWARD_POINT",
                            "action_args": {
                                "r": action_distance[i].item(),
                                "theta": theta[i].item(),
                            },
                        }
                    }
                )

        action_log_probs = pano_stop_distribution.log_probs(pano_stop)

        # only include distance and offset log probs if action != STOP
        pano_mask = (pano_stop != self._config.num_panos).to(
            action_log_probs.dtype
        )
        if self.wypt_cfg.predict_distance:
            action_log_probs = (
                action_log_probs
                + pano_mask
                * self.wypt_cfg.predict_distance
                * distance_log_probs
            )
        if self.wypt_cfg.predict_offset:
            action_log_probs = (
                action_log_probs
                + pano_mask * self.wypt_cfg.predict_offset * offset_log_probs
            )

        value = self.critic(x)
        action_elements = {
            "pano": pano_stop,
            "offset": offset,
            "distance": distance,
        }
        variances = {"distance": dist_var, "offset": ofst_var}
        modes = {"offset": ofst_mode, "distance": dist_mode}
        rnn_states_out = x.unsqueeze(1)

        return (
            value,
            actions,
            action_elements,
            modes,
            variances,
            action_log_probs,
            rnn_states_out,
            pano_stop_distribution,
            e_t,
            q_t
        )

    def get_value(
        self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
    ) -> Tensor:
        output = self.net(
            observations,
            prev_actions,
            masks,
        )
        hidden_state = output[5]
        return self.critic(hidden_state)

    
    def evaluate_actions(
        self,
        observations: Dict[str, Tensor],
        prev_actions: Dict[str, Tensor],
        masks: Tensor,
        action_components: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Tensor]:
        # print('start eval actions')
        # try:
        output = self.net(
            observations,
            # rnn_states,
            prev_actions,
            masks,
        )
        # except:
        #     for key in observations:
        #         print('key', observations[key].shape, torch.isnan(observations[key]).any())
        #     print('rnn', torch.isnan(rnn_states).any())
        #     print('pre_action', prev_actions)
        #     traceback.print_exc()
        # print(action_components)
        (
            pano_stop_distribution,
            offset_variable1,
            offset_variable2,
            distance_variable1,
            distance_variable2,
            x,
            e_t,
            q_t
        ) = output

        value = self.critic(x)
        pano_log_probs = pano_stop_distribution.log_probs(
            action_components["pano"]
        )
        # print('log_probs',pano_log_probs.view(-1))
        # if pano_log_probs.min() < -13:
        #     # if pano_log_probs.shape[0] == 2:
        #     probs = pano_stop_distribution.safe_prob(
        #         action_components["pano"].squeeze(-1)
        #     )
        #     print('logits', pano_stop_distribution.logits)
        #     print('log_probs', pano_log_probs)
        #     print('action', action_components["pano"])
        #     print('probs', probs)
        #     print('=====================')
        #     # print('logits shape', pano_stop_distribution.logits.shape)
        #     # print('log_probs shape', pano_log_probs.shape)
        #     # print('action shape', action_components["pano"].shape)

        idx = (
            action_components["pano"].to(torch.int64) % self._config.num_panos
        )

        # print('pano_log_probs',pano_log_probs.shape, 'idx', idx.shape)
        # print('distance_variable1',distance_variable1.shape, 'distance_variable2', distance_variable2.shape)

        distance_distribution = self._create_distance_distribution(
            distance_variable1, distance_variable2, idx
        )
        offset_distribution = self._create_offset_distribution(
            offset_variable1, offset_variable2, idx
        )

        # only include distance and offset log probs if the action included them
        pano_mask = (action_components["pano"] != self._config.num_panos).to(
            pano_log_probs.dtype
        )
        d_mask = pano_mask * self.wypt_cfg.predict_distance
        o_mask = pano_mask * self.wypt_cfg.predict_offset

        distance_log_probs = d_mask * distance_distribution.log_prob(
            action_components["distance"]
        )
        offset_log_probs = o_mask * offset_distribution.log_prob(
            action_components["offset"]
        )
        

        action_log_probs = (
            pano_log_probs + distance_log_probs + offset_log_probs
        )
        entropy = {
            "pano": pano_stop_distribution.entropy(),
            "offset": (o_mask * offset_distribution.entropy()).squeeze(1),
            "distance": (d_mask * distance_distribution.entropy()).squeeze(1),
        }
        rnn_states_out = x.unsqueeze(1) # T x N x 1 x dim
        return (
            value,
            action_log_probs,
            entropy,
            rnn_states_out,
        )
