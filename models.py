from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers*(2 if self.lstm.bidirectional else 1),
                         x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers*(2 if self.lstm.bidirectional else 1),
                         x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1])

# STT (Spatio-Temporal Transformer) Definition
class STT(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=16, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                    batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(dim_feedforward, num_classes)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.classifier(x.mean(dim=1))

# Medformer-style Transformer (simplified)
class Medformer(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_linear = nn.Linear(embed_dim, 1)  # attention weights
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        attn = torch.softmax(self.attn_linear(x), dim=1)  # shape (B, T, 1)
        pooled = (attn * x).sum(dim=1)
        # return self.classifier(x.mean(dim=1))
        return self.classifier(pooled)

# Focal Loss implementation for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    def forward(self, input, target):
        BCE_loss = self.ce(input, target)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() if self.reduction=='mean' else F_loss

class Chomp1d(nn.Module):
    """Removes the extra padding at the end to ensure causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch  = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation  = 2 ** i
            padding   = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_channels: list[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        input_dim: number of features per timestep
        num_channels: list of output channels for each TemporalBlock
        """
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)           # → (batch, input_dim, seq_len)
        y = self.tcn(x)                 # → (batch, num_channels[-1], seq_len)
        out = y[:, :, -1]               # take the last timestep
        return self.classifier(out)     # → (batch, num_classes)
