import torch 
import math 

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class CombinedEmbedding(torch.nn.Module):
    def __init__(self, d_model, vocab, dropout, max_len=5000):
        super(CombinedEmbedding, self).__init__()
        self.embedding = Embeddings(d_model, vocab)
        self.pe = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        x = self.pe(self.embedding(x))
        return x

class TransformerEncoderLayerWithMask(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayerWithMask, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.layer_norm = torch.nn.LayerNorm(d_model)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        x = self.encoder_layer(src = x, src_mask=mask)
        x = self.layer_norm(x)
        return x

class FinalFCLayer(torch.nn.Module):
    def __init__(self, d_model, vocab):
        super(LMLossFn, self).__init__()
        self.fc = torch.nn.Linear(d_model, vocab)
        self.d_model = d_model
        
    def forward(self, x):
        x = x.view(-1, self.d_model)
        x = self.fc(x)
        return x

def _transformer(N, D, H, vocab_size, dropout, max_len=5000):
    layers = [CombinedEmbedding(D, vocab_size, dropout, max_len)]

    for _ in range(N):
        layers.append(
            TransformerEncoderLayerWithMask(
                D, H, dim_feedforward=D*4, dropout=dropout, activation='relu'
                )
        )

    model = torch.nn.Sequential(*layers)

def gpt2_tiny():
    return _transformer(6, 512, 8, 50257, 0.1)

def gpt2_small():
    return _transformer(12, 768, 12, 50257, 0.1)

def gpt2_medium():
    return _transformer(24, 1024, 16, 50257, 0.1)

def gpt2_large():
    return _transformer(36, 1280, 20, 50257, 0.1)

def gpt2_xl():
    return _transformer(48, 1600, 25, 50257, 0.1)