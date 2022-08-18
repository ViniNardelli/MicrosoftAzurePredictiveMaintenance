import torch.nn as nn
from torch import triu, ones

"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser
 and Illia Polosukhin. 2017.
"""


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size=18, num_layers=3, dropout=0):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=9, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 3)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (triu(ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output
