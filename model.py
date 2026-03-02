import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000)/d))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class VectorQuantizer(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.emb = nn.Embedding(n, d)
        self.emb.weight.data.uniform_(-1/n,1/n)

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        dist = (
            flat.pow(2).sum(1,True)
            -2*flat@self.emb.weight.t()
            +self.emb.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        quant = self.emb(idx).view_as(x)
        loss = F.mse_loss(quant.detach(), x)
        return x + (quant-x).detach(), idx.view(x.shape[:2]), loss

class VocalSoundTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inp = nn.Linear(config.n_mels, config.hidden_dim)
        self.cls = nn.Embedding(config.num_classes, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim)
        enc = nn.TransformerEncoderLayer(
            config.hidden_dim, config.num_heads,
            config.ff_dim, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc, config.num_layers)
        self.vqs = nn.ModuleList([
            VectorQuantizer(config.codebook_size, config.hidden_dim)
            for _ in range(config.num_codebooks)
        ])
        self.out = nn.Linear(config.hidden_dim, config.n_mels)

    def forward(self, mel, label):
        x = self.inp(mel.transpose(1,2))
        x += self.cls(label).unsqueeze(1)
        x = self.pos(x)
        x = self.tr(x)

        vq_loss = 0
        residual = x
        codes = []
        for vq in self.vqs:
            q, c, l = vq(residual)
            residual = residual - q
            vq_loss += l
            codes.append(c)

        x = self.tr(x)
        return self.out(x).transpose(1,2), vq_loss
