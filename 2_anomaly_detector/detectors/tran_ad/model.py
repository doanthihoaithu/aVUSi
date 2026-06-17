# Adapted from https://github.com/imperial-qore/TranAD/tree/main
# Original License: BSD 3-Clause License
#
# Modifications:
# - Added statements to extract per-dimension contribution scores (DCM)
# - Added statements to save anomaly scores, dimension contributions, and related outputs to files

import math

from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
import torch

from detectors.tran_ad.dlultils import PositionalEncoding, TransformerEncoderLayer, TransformerDecoderLayer


class TranAD(nn.Module):
	def __init__(self, feats, n_window, device):
		super(TranAD, self).__init__()
		self.name = 'tran_ad'
		self.lr = 1e-3
		self.batch = 128
		self.device = device
		self.n_feats = feats
		self.n_window = n_window
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1,)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

class OmniAnomaly(nn.Module):
	def __init__(self, feats, n_window, device):
		super(OmniAnomaly, self).__init__()
		self.name = 'omni_anomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.batch = 128
		self.n_window = n_window
		self.device = device
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden),
			nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden),
			nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		# hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float32, device=self.device) if hidden is not None else hidden
		# out, hidden = self.lstm(x.view(1,1, -1), hidden)
		hidden = torch.rand(2, self.n_hidden, dtype=torch.float32, device=self.device)
		for i in range(0, self.n_window-1):
			out, hidden = self.lstm(x[:,i,:], hidden)
		# out, hidden = self.lstm(x, hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		# return x.view(-1), mu.view(-1), logvar.view(-1), hidden
		return x.view(-1, self.n_feats), mu.reshape(-1), logvar.reshape(-1), hidden

# ## MTAD_GAT Model (ICDM 20)
# class MTAD_GAT(nn.Module):
# 	def __init__(self, feats, device):
# 		super(MTAD_GAT, self).__init__()
# 		self.name = 'mtad_gat'
# 		self.device = device
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = feats
# 		self.n_hidden = feats * feats
# 		self.g = dgl.graph((torch.tensor(list(range(1, feats+1)), device=device), torch.tensor([0]*feats, device=device)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(feats, 1, feats)
# 		self.time_gat = GATConv(feats, 1, feats)
# 		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)
#
# 	def forward(self, data, hidden):
# 		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float32, device=self.device) if hidden is not None else hidden
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.cat((torch.zeros(1, self.n_feats, device=self.device), data))
# 		feat_r = self.feature_gat(self.g, data_r)
# 		data_t = torch.cat((torch.zeros(1, self.n_feats, device=self.device), data.t()))
# 		time_r = self.time_gat(self.g, data_t)
# 		data = torch.cat((torch.zeros(1, self.n_feats, device=self.device), data))
# 		data = data.view(self.n_window+1, self.n_feats, 1)
# 		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
# 		x, h = self.gru(x, hidden)
# 		return x.view(-1,self.n_feats), h

## GDN Model (AAAI 21)
# class GDN(nn.Module):
# 	def __init__(self, feats, device):
# 		super(GDN, self).__init__()
# 		self.name = 'gdn'
# 		self.device = device
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = 5
# 		self.n_hidden = 16
# 		self.n = self.n_window * self.n_feats
# 		src_ids = np.repeat(np.array(list(range(feats))), feats)
# 		dst_ids = np.array(list(range(feats))*feats)
# 		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(1, 1, feats)
# 		self.attention = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
# 		)
# 		self.fcn = nn.Sequential(
# 			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
# 		)
#
# 	def forward(self, data):
# 		# Bahdanau style attention
# 		data = data.view(-1)
# 		att_score = self.attention(data).view(self.n_window, 1)
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.matmul(data.permute(1, 0), att_score)
# 		# GAT convolution on complete graph
# 		feat_r = self.feature_gat(self.g, data_r)
# 		feat_r = feat_r.view(self.n_feats, self.n_feats)
# 		# Pass through a FCN
# 		x = self.fcn(feat_r)
# 		return x.permute(1, 0)
