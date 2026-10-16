import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================================
# 1. Evidential Regression Head
# Corresponds to: III-B Evidential Regression Head
# ==========================================================
class EvidentialHead(nn.Module):
    """
    Evidential regression head for Normal-Inverse-Gamma distribution.

    Output:
        gamma: expected mean of BP prediction
        nu: virtual observation count
        alpha: shape parameter
        beta: scale parameter
    """

    def __init__(self, in_features, out_features=2):
        super(EvidentialHead, self).__init__()
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features * 4)

    def forward(self, x):
        out = self.fc(x).view(-1, self.out_features, 4)

        gamma = out[:, :, 0]
        nu = F.softplus(out[:, :, 1]) + 1e-6
        alpha = F.softplus(out[:, :, 2]) + 1.0 + 1e-6
        beta = F.softplus(out[:, :, 3]) + 1e-6

        return gamma, nu, alpha, beta


# ==========================================================
# 2. KAN-based FiLM modulation
# Corresponds to: III-A-2 KFM-LSTM Module
# ==========================================================
class KANLinear(nn.Module):
    """
    Single-layer KAN mapping with linear base term and Fourier basis.

    Implements:
        Phi(G) = W_base * G + b_base + sum_k C_k * phi_k(G)
        phi_k(G) = sin(kG) + cos(kG)
    """

    def __init__(self, in_features, out_features, grid_size=5):
        super(KANLinear, self).__init__()
        self.grid_size = grid_size

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_bias = nn.Parameter(torch.Tensor(out_features))
        self.coef = nn.Parameter(torch.Tensor(out_features, in_features, grid_size * 2))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.zeros_(self.base_bias)
        nn.init.normal_(self.coef, mean=0.0, std=0.01)

    def forward(self, x):
        base = F.linear(x, self.base_weight, self.base_bias)

        k = torch.arange(1, self.grid_size + 1, device=x.device).float()
        phase = x.unsqueeze(-1) * k
        basis = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        out = base + torch.einsum('big,oig->bo', basis, self.coef)
        return out


class KANFiLMLayer(nn.Module):
    """
    FiLM modulation using KAN-generated scale and shift coefficients.

    Implements:
        s1 = Phi_scale(G)
        s2 = Phi_shift(G)
        X_out = s1 * X + s2
    """

    def __init__(self, guide_dim, channels):
        super(KANFiLMLayer, self).__init__()
        self.scale = KANLinear(guide_dim, channels)
        self.shift = KANLinear(guide_dim, channels)

        # Identity initialization strategy described in the paper.
        nn.init.constant_(self.scale.base_bias, 1.0)
        nn.init.constant_(self.shift.base_bias, 0.0)

    def forward(self, x, guide):
        """
        Args:
            x: [B, C, L]
            guide: [B, guide_dim]
        """
        scale = self.scale(guide).unsqueeze(2)
        shift = self.shift(guide).unsqueeze(2)
        return x * scale + shift


# ==========================================================
# 3. CNN Module
# Corresponds to: III-A-1 CNN Module
# ==========================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    Residual block with three convolutional layers, as described in the paper.
    """

    def __init__(self, inplanes, planes, shortcut=True):
        super(BasicBlock, self).__init__()
        self.downsample = None
        self.shortcut = shortcut

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm1d(planes)

        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.shortcut:
            out += identity

        return self.relu(out)


cfg = {'A': [32, 32, 'M', 64, 64, 'M']}


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []

    for v in cfg['A']:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(
                in_channels,
                v,
                kernel_size=3,
                stride=1,
                padding=1
            )

            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)


class CNNModule(nn.Module):
    """
    CNN module described in III-A-1.

    Structure:
        convolutional block:
            4 conv layers + 2 max pooling layers
        residual blocks:
            2 residual blocks, each with 3 conv layers
        final max pooling
    """

    def __init__(self, in_channels=3):
        super(CNNModule, self).__init__()

        self.conv_block = make_layers(cfg, in_channels=in_channels, batch_norm=True)

        self.res_block1 = BasicBlock(64, 128)
        self.res_block2 = BasicBlock(128, 128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool(x)
        return x


# ==========================================================
# 4. Attention modules used in FMSTE
# ==========================================================
class TemporalAttention(nn.Module):
    """
    Temporal attention used in XLSTM module.
    """

    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.Q = nn.Parameter(torch.rand(feature_dim, 1))

    def forward(self, x):
        scores = torch.matmul(x, self.Q)
        scores = F.silu(scores)
        attn_weights = F.softmax(scores, dim=1)
        return x * attn_weights


class SpatialAttention(nn.Module):
    """
    Spatial attention used in Fusion module.
    """

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.Q = nn.Parameter(torch.rand(1, 1))

    def forward(self, x):
        """
        Args:
            x: [B, D, 1]
        """
        scores = torch.matmul(x, self.Q)
        scores = F.silu(scores)
        attn_weights = F.softmax(scores, dim=1)
        return x * attn_weights


class CrossAttention(nn.Module):
    """
    Cross attention used in Fusion module.
    """

    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.Q = nn.Parameter(torch.rand(feature_dim, 1))

    def forward(self, x):
        """
        Args:
            x: [B, 2, D]
        """
        scores = torch.matmul(x, self.Q)
        scores = F.silu(scores)
        attn_weights = F.softmax(scores, dim=1)
        return x * attn_weights


# ==========================================================
# 5. KFM-LSTM Module
# Corresponds to: III-A-2 KFM-LSTM Module
# ==========================================================
class KFMLSTM(nn.Module):
    """
    KFM-LSTM module:
        1. FiLM modulation using guidance feature G
        2. Smooth absolute value transformation S(x) = x * tanh(x)
        3. Hourglass LSTM: 128 -> 64 -> 128
    """

    def __init__(self, guide_dim=39, channels=128):
        super(KFMLSTM, self).__init__()

        self.film = KANFiLMLayer(guide_dim=guide_dim, channels=channels)

        self.lstm1 = nn.LSTM(
            input_size=channels,
            hidden_size=64,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=channels,
            batch_first=True
        )

    def forward(self, x, guide):
        """
        Args:
            x: [B, 128, L]
            guide: [B, guide_dim]
        """
        x = self.film(x, guide)

        # Smooth absolute value transformation: S(x) = x * tanh(x)
        x = x * torch.tanh(x)

        # LSTM expects [B, T, C]
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)

        # Use the final hidden state as the output representation
        return h_n.squeeze(0)


# ==========================================================
# 6. XLSTM Module
# Corresponds to: III-A-3 XLSTM Module
# ==========================================================
class XLSTMModule(nn.Module):
    """
    XLSTM module:
        BiLSTM -> Temporal Attention -> LSTM -> LSTM
    """

    def __init__(self, in_channels=3):
        super(XLSTMModule, self).__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.temp_attention = TemporalAttention(feature_dim=128)

        self.lstm_up = nn.LSTM(
            input_size=128,
            hidden_size=256,
            batch_first=True
        )

        self.lstm_down = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, L]
        """
        # [B, L, C]
        x = x.permute(0, 2, 1)

        x, _ = self.bilstm(x)
        x = self.temp_attention(x)
        x, _ = self.lstm_up(x)
        x, _ = self.lstm_down(x)

        # Use the hidden state at the final time step
        return x[:, -1, :]


# ==========================================================
# 7. Fusion Module
# Corresponds to: III-A-4 Fusion Module
# ==========================================================
class FusionModule(nn.Module):
    """
    Fusion module:
        Spatial attention -> LayerNorm -> Cross attention
        Guidance feature encoding -> element-wise modulation + residual
    """

    def __init__(self, feature_dim=128, meta_dim=39, out_dim=256):
        super(FusionModule, self).__init__()

        self.spatial_attention = SpatialAttention()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.cross_attention = CrossAttention(feature_dim)

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, local_feat, temporal_feat, guide):
        """
        Args:
            local_feat: [B, 128], output of KFM-LSTM
            temporal_feat: [B, 128], output of XLSTM
            guide: [B, 39], demographic + additional features
        """
        # Spatial attention
        v1 = self.spatial_attention(local_feat.unsqueeze(2)).squeeze(2)
        v2 = self.spatial_attention(temporal_feat.unsqueeze(2)).squeeze(2)

        # Layer normalization
        v1 = self.layer_norm(v1)
        v2 = self.layer_norm(v2)

        # Stack two feature vectors: [B, 2, 128]
        v3 = torch.stack([v1, v2], dim=1)

        # Cross attention
        v_att = self.cross_attention(v3)

        # Concatenate: [B, 256]
        v4 = torch.cat([v_att[:, 0, :], v_att[:, 1, :]], dim=1)

        # Guidance feature encoding and modulation
        guide_encoded = torch.tanh(self.meta_encoder(guide))
        out = v4 * guide_encoded + v4

        return out


# ==========================================================
# 8. FMSTE Backbone
# Corresponds to: III-A FMSTE
# ==========================================================
class FMSTE(nn.Module):
    """
    Feature-Modulated Spatio-Temporal Encoder.

    Components:
        CNNModule
        KFMLSTM
        XLSTMModule
        FusionModule
    """

    def __init__(self, in_channels=3, demo_dim=5, extra_dim=34):
        super(FMSTE, self).__init__()

        guide_dim = demo_dim + extra_dim

        self.cnn = CNNModule(in_channels=in_channels)

        self.kfm_lstm = KFMLSTM(
            guide_dim=guide_dim,
            channels=128
        )

        self.xlstm = XLSTMModule(in_channels=in_channels)

        self.fusion = FusionModule(
            feature_dim=128,
            meta_dim=guide_dim,
            out_dim=256
        )

    def forward(self, x, demos, extra_feats):
        """
        Args:
            x: [B, C, L], PPG / VPG / APG signals
            demos: [B, 5], demographic features
            extra_feats: [B, 34], selected additional PPG features
        """
        # Guidance feature G: demographic + additional features
        guide = torch.cat([demos, extra_feats], dim=1)

        # Local spatial feature extraction
        local_feat = self.cnn(x)

        # KFM-LSTM: FiLM modulation + smooth abs + hourglass LSTM
        local_feat = self.kfm_lstm(local_feat, guide)

        # Temporal feature extraction
        temporal_feat = self.xlstm(x)

        # Feature fusion
        fused_feat = self.fusion(local_feat, temporal_feat, guide)

        return fused_feat


# ==========================================================
# 9. SERNet
# Corresponds to: III METHODOLOGY
# ==========================================================
class SERNet(nn.Module):
    """
    SERNet: Siamese Evidential Regression Network.

    Structure:
        Shared FMSTE encoder
        Absolute evidential regression head
        Differential evidential regression head
        Confidence-driven gating mechanism
    """

    def __init__(self, num_classes=2, in_channels=3, demo_dim=5, extra_dim=34):
        super(SERNet, self).__init__()

        self.encoder = FMSTE(
            in_channels=in_channels,
            demo_dim=demo_dim,
            extra_dim=extra_dim
        )

        feat_dim = 256

        # Absolute evidential regression head
        self.abs_regressor_pre = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.abs_evidential = EvidentialHead(128, num_classes)

        # Differential evidential regression head
        self.diff_extractor = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.delta_evidential = EvidentialHead(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        x,
        demos,
        extra_feats,
        x_ref=None,
        demos_ref=None,
        extra_feats_ref=None,
        label_ref=None
    ):
        """
        Args:
            x: current PPG input, [B, C, L]
            demos: current demographic features, [B, 5]
            extra_feats: current additional features, [B, 34]

            x_ref: reference PPG input, [B, C, L]
            demos_ref: reference demographic features, [B, 5]
            extra_feats_ref: reference additional features, [B, 34]
            label_ref: reference BP value, [B, 2]
        """
        # Shared encoder extracts current features
        feat_curr = self.encoder(x, demos, extra_feats)

        # Absolute evidential prediction
        gamma_abs, nu_abs, alpha_abs, beta_abs = self.abs_evidential(
            self.abs_regressor_pre(feat_curr)
        )

        # If no reference input is given, only use absolute path
        if x_ref is None:
            return gamma_abs, None, (gamma_abs, nu_abs, alpha_abs, beta_abs)

        # Shared encoder extracts reference features
        feat_ref = self.encoder(x_ref, demos_ref, extra_feats_ref)

        # Siamese differential feature:
        # [F_curr - F_ref, F_curr * F_ref]
        diff_input = torch.cat(
            [feat_curr - feat_ref, feat_curr * feat_ref],
            dim=1
        )

        diff_emb = self.diff_extractor(diff_input)

        # Differential evidential prediction
        gamma_delta, nu_delta, alpha_delta, beta_delta = self.delta_evidential(diff_emb)

        # Uncertainty calculation:
        # u = beta / (nu * (alpha - 1))
        u_abs = beta_abs / (nu_abs * (alpha_abs - 1) + 1e-6)
        u_delta = beta_delta / (nu_delta * (alpha_delta - 1) + 1e-6)

        # Confidence: c_j = exp(-u_j)
        c_abs = torch.exp(-u_abs.mean(dim=1, keepdim=True))
        c_delta = torch.exp(-u_delta.mean(dim=1, keepdim=True))

        # Gating coefficient:
        # g = c_delta / (c_abs + c_delta + eps)
        g = c_delta / (c_abs + c_delta + 1e-6)

        # Confidence-driven final prediction:
        # y_hat = (1 - g) * gamma_abs + g * (refer_bp + gamma_delta)
        final_pred = (1 - g) * gamma_abs + g * (label_ref + gamma_delta)

        return final_pred, g, (
            gamma_abs, nu_abs, alpha_abs, beta_abs,
            gamma_delta, nu_delta, alpha_delta, beta_delta
        )
