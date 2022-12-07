from turtle import forward
from typing import Dict, Tuple

from habitat import Config
import numpy as np
import torch
from torch import Tensor, device, nn
from gym import Space

from vienna.models.encoders import resnet_encoders
from collections import defaultdict
import os

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn




class CLEnc(nn.Module):
    def __init__(self, model_config: Config, feat_size, trainable=True):
        super().__init__()
        self.word_embedding = nn.Embedding(
            num_embeddings=model_config.INSTRUCTION_ENCODER.vocab_size,
            embedding_dim=model_config.INSTRUCTION_ENCODER.embedding_size,
            padding_idx=0,
        )

        self.feat_size = feat_size
        self.feat_enc = nn.Sequential(
            nn.Linear(self.feat_size, self.feat_size),
            nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size),
            nn.ReLU(),
        )

        self.attn = SoftDotAttention(model_config.INSTRUCTION_ENCODER.embedding_size, self.feat_size)

        self.linear = nn.Linear(self.feat_size, model_config.INSTRUCTION_ENCODER.embedding_size)


        self.models = [self.feat_enc, self.attn, self.linear]
        for m in self.models:
            for param in m.parameters():
                param.requires_grad_(trainable)
        
    
    def forward(self, rgb_features):
        '''
            rgb_features: ... x feature_size
        '''
        return self.feat_enc(rgb_features)
    

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path, map_location='cpu')
        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            
        all_tuple = [("wb", self.word_embedding),
                     ("feat_enc", self.feat_enc),
                     ("attn", self.attn),
                     ("linear", self.linear)
                     ]
        for param in all_tuple:
            recover_state(*param)
        
        
class CLEncAgent:
    def __init__(self, model_config: Config):
        cnn_type = model_config.RGB_ENCODER.cnn_type
        self.rgb_encoder = getattr(resnet_encoders, cnn_type)(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            spatial_output=True,
            single_spatial_filter=False,
        )
        self.clenc = CLEnc(model_config, self.rgb_encoder.output_shape[0])
        self.cl_optimizer = torch.optim.Adam(self.clenc.parameters(), lr=1e-4)


    def to(self, device):
        self.rgb_encoder.to(device)
        self.clenc.to(device)

    def get_similarity(self, a, b):
        '''
            a: n x dim
            b: n x dim
            return : n
        '''
        a_norm = a / torch.sqrt((a * a).sum(-1, True))
        b_norm = b / torch.sqrt((b * b).sum(-1, True))
        return (a_norm*b_norm).sum(-1)

    def forward(self, observations):
        assert "rgb" in observations

        rgb_obs = observations['rgb']
        rgb_size = rgb_obs.size()
        bs, M, _, _, _ = rgb_size 
        rgb_obs = rgb_obs.view(rgb_size[0] * rgb_size[1], *rgb_size[2:]) # bs * num_pano x W x H x D
        feats = self.rgb_encoder({"rgb" : rgb_obs}) # bs * num_pano x D x W x H
        feats_size = feats.size()
        feats = feats.view(bs, M, *feats_size[1:]) # bs x M x D x W x H
        return feats
        
    def rollout(self, observations):
        '''
            rgb_obs: batch x T x num_panos x W x H x D
            neg_views: batch x neg_num x W x H x D
            neg_vocabs: batch x neg_num x W x H x D

            rgb_features: [T x num_panos x D x W x H] * batch
            neg_views_feats: [neg_num x D x W x H] * batch
            neg_vocab: [neg_num] * batch
            queries: [word_num] * batch
            masks: batch x T (do not need it anymore)
            
        '''
        self.logs = {}
        # assert "rgb" in observations
        assert "rgb_features" in observations
        # assert "neg_views" in observations
        assert "neg_views_feats" in observations
        assert "neg_vocabs" in observations
        # assert "instruction" in observations
        assert "queries" in observations

        # rgb_obs = observations['rgb']
        # rgb_size = rgb_obs.size()
        # bs, T, M, _, _, _ = rgb_size
        # rgb_obs = rgb_obs.view(rgb_size[0] * rgb_size[1] * rgb_size[2], *rgb_size[3:]) # bs * T * num_pano x W x H x D
        
        # rgb_embedding = self.rgb_encoder({"rgb" : rgb_obs}) # bs * T * num_pano x D x W x H
        # rgb_embedding = rgb_embedding.view(*rgb_size[:2], *rgb_embedding.shape[1:]) # bs x T x D x W x H
        # rgb_embedding = torch.flatten(rgb_embedding, 3) # bs * T * num_pano x D x W * H
        # rgb_embedding = rgb_embedding.permute(0, 2, 1).view(bs, -1, self.feat_size) # bs x T * num_pano * W * H x D
        # rgb_embedding = self.feat_enc(rgb_embedding) 

        # neg_views = observations["neg_views"] 
        # neg_views_size = neg_views.size()
        # _, neg_num, _, _, _ = neg_views_size
        # neg_views = neg_views.view(neg_views_size[0] * neg_views_size[1], *neg_views_size[2:])

        # neg_views_embedding = self.rgb_encoder({"rgb" : neg_views}) # bs * neg_num x D x W x H
        # neg_views_embedding = torch.flatten(neg_views_embedding, 2) # bs * neg_num x D x W * H
        # neg_views_embedding = neg_views_embedding.permute(0, 2, 1).view(bs, -1, self.feat_size) # bs x neg_num * W * H x D
        
        rgb_embedding = observations["rgb_features"]
        queries = observations["queries"]

        neg_views_feats = observations["neg_views_feats"]
        neg_vocabs = observations["neg_vocabs"] 
        

        positive_sample_cnt = 0
        negative_loss = 0.
        positive_loss = 0.
        negative_loss_vocab_vocab = 0.
        negative_loss_vocab_view = 0.
        negative_loss_view_vocab = 0.
        negative_loss_view_view = 0.
        entropy = 0.
        for i, vocabs_tokens in enumerate(queries): # batch

            vocabs_feat = self.clenc.word_embedding(vocabs_tokens) # vocab_num x embdim

            vis_feat = rgb_embedding[i]
            vis_feat = vis_feat.permute(0,1,3,4,2)
            _,_,_,_,D = vis_feat.size()
            vis_feat = vis_feat.reshape(-1, D)

            vis_feat = self.clenc.feat_enc(vis_feat)
            vis_feat = vis_feat.unsqueeze(0) # 1 x max_length x feature_size


            nega_view_feat = neg_views_feats[i]
            nega_view_feat = self.clenc.feat_enc(nega_view_feat)
            nega_view_feat = self.clenc.linear(nega_view_feat)

            
            # negative samples
            nega_vocab_feat = self.clenc.word_embedding(neg_vocabs[i]) # neg_num x D

            neg_num_vocab, _ = nega_vocab_feat.size()
            neg_num_views, _ = nega_view_feat.size()
            

            for vocab_feat in vocabs_feat: # enumerate the positive tokens
                vocab_feat = vocab_feat.unsqueeze(0) # 1 x embdim
                _, probs = self.clenc.attn(vocab_feat, vis_feat, None, False, True) # 1 x feature_size
                c = torch.distributions.Categorical(probs)
                entropy += c.entropy().sum()
                _, idx = probs.max(1)
                idx = idx.detach()
                chosen_vis_feat = vis_feat[0, idx]
                chosen_vis_feat_linear = self.clenc.linear(chosen_vis_feat) # 1 x embdim
                
                positive_loss += - self.get_similarity(chosen_vis_feat_linear, vocab_feat).mean()

                ################## negative loss for view ################
                chosen_vis_feat_linear_repeat = chosen_vis_feat_linear.repeat(neg_num_vocab,1)
                # negative view to vocab                 
                negative_loss_view_vocab += self.get_similarity(chosen_vis_feat_linear_repeat, nega_vocab_feat).mean()

                chosen_vis_feat_linear_repeat = chosen_vis_feat_linear.repeat(neg_num_views,1)
                
                # negative view to view
                negative_loss_view_view += self.get_similarity(chosen_vis_feat_linear_repeat, nega_view_feat).mean()

                ################## negative loss for vocab ################
                vocab_feat_repeat = vocab_feat.repeat(neg_num_views, 1)
                # negative vocab to view
                negative_loss_vocab_view += self.get_similarity(vocab_feat_repeat, nega_view_feat).mean()

                vocab_feat_repeat = vocab_feat.repeat(neg_num_vocab, 1)

                # negative vocab to vocab
                negative_loss_vocab_vocab += self.get_similarity(vocab_feat_repeat, nega_vocab_feat).mean()
                

                positive_sample_cnt += 1


        negative_loss_vocab_vocab /= positive_sample_cnt
        negative_loss_vocab_view /= positive_sample_cnt
        negative_loss_view_vocab /= positive_sample_cnt
        negative_loss_view_view /= positive_sample_cnt

        negative_loss = (negative_loss_vocab_vocab + negative_loss_vocab_view + negative_loss_view_vocab + negative_loss_view_view) / 4
        positive_loss /= positive_sample_cnt

        self.logs['negative_vocab_vocab'] = negative_loss_vocab_vocab.item()
        self.logs['negative_vocab_view'] = negative_loss_vocab_view.item()
        self.logs['negative_view_vocab'] = negative_loss_view_vocab.item()
        self.logs['negative_view_view'] = negative_loss_view_view.item()
        self.logs['positive_loss'] = positive_loss.item()
        self.logs['negative_loss'] = negative_loss.item()
        return negative_loss, positive_loss

    
    def train(self, observations):
        self.logs = {}

        negative_loss, positive_loss = self.rollout(observations)
        
        total_loss = negative_loss + positive_loss

        self.cl_optimizer.zero_grad()
        
        total_loss.backward()

        self.cl_optimizer.step()


    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }
        all_tuple = [("wb", self.clenc.word_embedding),
                     ("feat_enc", self.clenc.feat_enc),
                     ("attn", self.clenc.attn),
                     ("linear", self.clenc.linear)
                     ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            
        all_tuple = [("wb", self.clenc.word_embedding),
                     ("feat_enc", self.clenc.feat_enc),
                     ("attn", self.clenc.attn),
                     ("linear", self.clenc.linear)
                     ]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1