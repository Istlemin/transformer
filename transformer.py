import copy
from torch import nn
import torch

import numpy as np
from torch.nn.modules.transformer import _get_clones

class TransformerEncodingLayer(nn.Module):
    def __init__(self,device,embed_dim, num_heads=8, hidden_layer_size=512, dropout=0.0):
        super(TransformerEncodingLayer, self).__init__()
        self.device = device
        self.mha = nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.layer_norm1(X+self.dropout(self.mha(X,X,X)[0]))
        X = self.layer_norm2(X+self.dropout(self.ff(X)))
        return X

class TransformerDecodingLayer(nn.Module):
    def __init__(self,device,embed_dim,phrase_length,num_heads=4,hidden_layer_size=512, dropout=0.0):
        super(TransformerDecodingLayer, self).__init__()
        self.device = device
        self.mha1 = nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout)
        self.mha2 = nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.look_ahead_mask = torch.tensor([[i<j for j in range(phrase_length)] for i in range(phrase_length)],device=device)

    def forward(self, X, Y):
        Y = self.layer_norm1(Y+self.dropout(self.mha1(Y,Y,Y,attn_mask=self.look_ahead_mask)[0]))
        Y = self.layer_norm2(Y+self.dropout(self.mha2(Y,X,X)[0]))
        Y = self.layer_norm3(Y+self.dropout(self.ff(Y)))
        return Y

def get_angles(pos, i, embed_dim):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / embed_dim)
  return pos * angle_rates

def positional_encoding(phrase_length, embed_dim):
  angle_rads = get_angles(np.arange(phrase_length)[:, np.newaxis],
                          np.arange(embed_dim)[np.newaxis, :],
                          embed_dim)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  return angle_rads

class Transformer(nn.Module):
    def __init__(self,device, num_input_tokens, num_output_tokens, phrase_length, embed_dim=256, encoder_layers=6, decoder_layers=6, num_heads=8):
        super(Transformer, self).__init__()
        self.device = device
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens
        self.phrase_length = phrase_length
        self.embed_dim = embed_dim
        
        self.look_ahead_mask = torch.tensor([[i<j for j in range(phrase_length)] for i in range(phrase_length)],device=device)

        self.encoding_layers = nn.ModuleList([TransformerEncodingLayer(device,self.embed_dim,num_heads=num_heads).to(device) for _ in range(encoder_layers)])
        self.decoding_layers = nn.ModuleList([TransformerDecodingLayer(device,self.embed_dim,phrase_length,num_heads=num_heads) for i in range(decoder_layers)])

        self.input_embedder = nn.Embedding(num_input_tokens, self.embed_dim)
        self.target_embedder = nn.Embedding(num_output_tokens, self.embed_dim)
        self.debedder = nn.Linear(self.embed_dim, num_output_tokens)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.pos_encoding = torch.tensor(positional_encoding(phrase_length,self.embed_dim),device=device).float().reshape((phrase_length,1,self.embed_dim))
        
    def forward(self, X,Y):
        
        X = self.input_embedder(X)*self.embed_dim**0.5 + self.pos_encoding
        Y = self.target_embedder(Y)*self.embed_dim**0.5 + self.pos_encoding
        
        for l in self.encoding_layers:
            X = l(X)
        
        for l in self.decoding_layers:
            Y = l(X,Y)

        return self.debedder(Y)
    
    def loss(self, out, Y):
        return self.cross_entropy(out[:-1].reshape((-1, self.num_output_tokens)), Y[1:].flatten())