# Adapted from https://github.com/ML4ITS/mtad-gat-pytorch/tree/main
# Original License: MIT License
#
# Modifications:
# - Added statements to extract per-dimension contribution scores (DCM)
# - Added statements to save anomaly scores, dimension contributions, and related outputs to files

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .modules import FeatureAttentionLayer, TemporalAttentionLayer, ConvLayer, GRULayer, Forecasting_Model


class MTAD_GAT(nn.Module):

    def __init__(self, customParameters: DictConfig):
        super(MTAD_GAT, self).__init__()
        self.n_feats = customParameters['n_feats']
        self.n_window = customParameters['n_window']
        self.out_dim = customParameters['n_feats']
        self.kernel_size = customParameters.get('kernel_size', 5)
        self.dropout = customParameters.get('dropout', 0.2)
        self.alpha = customParameters.get('alpha', 0.2)
        self.embed_dim = customParameters.get('embed_dim', None)
        self.use_gatv2 = customParameters.get('use_gatv2', True)
        self.use_bias = customParameters.get('use_bias', True)

        self.gru_n_layers = customParameters.get('gru_n_layers', 1)
        self.gru_hid_dim = customParameters.get('gru_hid_dim', 64)

        self.forecast_n_layers = customParameters.get('forecast_n_layers', 1)
        self.forecast_hid_dim = customParameters.get('forecast_hid_dim', 64)

        self.conv = ConvLayer(n_features=self.n_feats, kernel_size=self.kernel_size)

        self.feature_gat = FeatureAttentionLayer(n_features=self.n_feats,
                                                 window_size=self.n_window,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 embed_dim=self.embed_dim,
                                                 use_gatv2=self.use_gatv2,
                                                 use_bias=self.use_bias)
        self.temporal_gat = TemporalAttentionLayer(n_features=self.n_feats,
                                                 window_size=self.n_window,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 embed_dim=self.embed_dim,
                                                 use_gatv2=self.use_gatv2,
                                                 use_bias=self.use_bias)

        self.gru = GRULayer(in_dim=3 * self.n_feats,
                            hid_dim=self.gru_hid_dim,
                            n_layers=self.gru_n_layers,
                            dropout=self.dropout)

        self.forecating_model = Forecasting_Model(in_dim=self.gru_hid_dim,
                                                  hid_dim=self.forecast_hid_dim,
                                                  n_layers=self.forecast_n_layers,
                                                  dropout=self.dropout,
                                                  out_dim=self.out_dim)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.conv(x)
        feat_r = self.feature_gat(x)
        time_r = self.temporal_gat(x)
        x = torch.cat((x, feat_r, time_r), dim=2) # (b, n, 3k)
        _, h_end = self.gru(x)
        x = self.forecating_model(h_end)
        return x

