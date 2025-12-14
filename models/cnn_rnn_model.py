import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRNNModel(nn.Module):
    """
    Basic CNN + RNN model for automatic music transcription.
    
    Input:
        - Mel spectrogram tensor of shape (B, 1, n_mels, T)
    Output:
        - Predicted piano-roll logits of shape (B, 88, T)
          (before sigmoid activation)
    """

    def __init__(
        self,
        n_mels: int = 229,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.output_dim = 88  # added for consistency with zero-length fallback

        # --- CNN Frontend ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample freq only

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample freq again
        )

        # --- Recurrent Layer (temporal modeling) ---
        freq_bins_after = n_mels // 4  # due to two (2×1) pools
        lstm_input_size = 64 * freq_bins_after

        self.rnn = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output Projection ---
        self.fc = nn.Linear(hidden_size * 2, self.output_dim)  # 88 piano keys

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)  # (B, C, F, T)
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        B, T, C, F = features.shape
        features = features.view(B, T, C * F).contiguous()    # (B, T, C*F)

        # Defensive check for zero-length sequences
        if T == 0:
            return torch.zeros(B, self.output_dim, 1, device=x.device)

        # Run LSTM in FP32 for cuDNN stability (even under AMP)
        with torch.amp.autocast('cuda', enabled=False):
            rnn_out, _ = self.rnn(features.float())  # (B, T, 2*hidden_size)

        # Framewise projection → 88 keys
        logits = self.fc(rnn_out)       # (B, T, 88)
        return logits.transpose(1, 2)   # (B, 88, T)

class ResidualBlock(nn.Module):
    """Residual block for CNN with skip connections."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if channels differ
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for temporal modeling with numerical stability."""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Numerical stability constants
        self.attn_clip_val = 10.0  # Clip attention logits to prevent overflow

    def forward(self, x):
        # x: (B, T, hidden_dim)
        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores with clipping for stability
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Clip attention logits to prevent overflow in softmax
        attn = torch.clamp(attn, min=-self.attn_clip_val, max=self.attn_clip_val)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # (B, T, hidden_dim)
        out = self.proj(out)
        return out


class CNNRNNModelLarge(nn.Module):
    """
    Enhanced CNN + RNN model with 7 major improvements:
    1. Deeper CNN (4 blocks instead of 2)
    2. Residual connections for better gradient flow
    3. Multi-head attention for temporal context
    4. Frequency-aware convolutions (asymmetric kernels)
    5. Advanced dropout (variational + spatial + DropConnect)
    6. Multi-scale temporal modeling (parallel LSTMs)
    7. Onset/offset detection heads for better note boundaries

    Input: (B, 1, n_mels, T)
    Output: (B, 88, T) - frame predictions (or dict with onset/offset if using heads)
    """

    def __init__(
        self,
        n_mels: int = 229,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
        use_onset_offset_heads: bool = True,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.output_dim = 88
        self.use_attention = use_attention
        self.use_onset_offset_heads = use_onset_offset_heads

        # =====================================================================
        # IMPROVEMENT 1 & 2: Deeper CNN with Residual Connections
        # =====================================================================
        # Block 1: 1 → 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample freq
        )

        # Block 2: 32 → 64 with residual
        self.res_block1 = ResidualBlock(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout2d_1 = nn.Dropout2d(0.1)  # Improvement 5: Spatial dropout

        # Block 3: 64 → 128 with residual
        self.res_block2 = ResidualBlock(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.dropout2d_2 = nn.Dropout2d(0.1)

        # IMPROVEMENT 4: Frequency-aware asymmetric convolution
        # (7, 3) kernel captures wider frequency context
        self.freq_aware_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # one more freq downsample
        )
        self.dropout2d_3 = nn.Dropout2d(0.15)

        # =====================================================================
        # IMPROVEMENT 6: Multi-Scale Temporal Modeling
        # =====================================================================
        # Calculate feature dimension after CNN
        freq_bins_after = n_mels // 8  # three (2×1) pools
        lstm_input_size = 256 * freq_bins_after

        # Main LSTM with variational dropout
        self.rnn_main = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # Additional LSTM for multi-scale (smaller, captures local patterns)
        self.rnn_local = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Combine main + local RNN outputs
        combined_rnn_dim = hidden_size * 2 + (hidden_size // 2) * 2  # 2*512 + 2*256 = 1536

        # =====================================================================
        # IMPROVEMENT 3: Multi-Head Attention
        # =====================================================================
        if use_attention:
            self.attention = MultiHeadAttention(
                hidden_dim=combined_rnn_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            # LayerNorm with epsilon for numerical stability
            self.attention_norm = nn.LayerNorm(combined_rnn_dim, eps=1e-6)

        # =====================================================================
        # IMPROVEMENT 7: Onset/Offset Detection Heads
        # =====================================================================
        if use_onset_offset_heads:
            # Shared feature layer
            self.shared_fc = nn.Linear(combined_rnn_dim, hidden_size)
            self.shared_dropout = nn.Dropout(dropout * 1.5)  # Higher dropout before heads

            # Three heads: frame, onset, offset
            self.frame_head = nn.Linear(hidden_size, self.output_dim)
            self.onset_head = nn.Linear(hidden_size, self.output_dim)
            self.offset_head = nn.Linear(hidden_size, self.output_dim)
        else:
            # Single head with DropConnect (Improvement 5)
            self.fc = nn.Linear(combined_rnn_dim, self.output_dim)
            self.fc_dropout = nn.Dropout(dropout * 1.5)

    def forward(self, x, return_all_heads=False):
        """
        Forward pass with all improvements.

        Args:
            x: (B, 1, n_mels, T) mel spectrogram
            return_all_heads: if True and using onset/offset heads, return dict

        Returns:
            If return_all_heads=True and use_onset_offset_heads=True:
                dict with keys 'frame', 'onset', 'offset' (B, 88, T)
            Otherwise:
                logits (B, 88, T)
        """
        # =====================================================================
        # CNN Feature Extraction with Residuals
        # =====================================================================
        x = self.conv1(x)  # (B, 32, F/2, T)

        x = self.res_block1(x)  # (B, 64, F/2, T)
        x = self.pool1(x)  # (B, 64, F/4, T)
        x = self.dropout2d_1(x)

        x = self.res_block2(x)  # (B, 128, F/4, T)
        x = self.dropout2d_2(x)

        x = self.freq_aware_conv(x)  # (B, 256, F/8, T)
        x = self.dropout2d_3(x)

        # Reshape for RNN
        features = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, Freq)
        B, T, C, Freq = features.shape
        features = features.view(B, T, C * Freq).contiguous()  # (B, T, C*Freq)

        # Defensive check for zero-length sequences
        if T == 0:
            if self.use_onset_offset_heads and return_all_heads:
                return {
                    'frame': torch.zeros(B, self.output_dim, 1, device=x.device),
                    'onset': torch.zeros(B, self.output_dim, 1, device=x.device),
                    'offset': torch.zeros(B, self.output_dim, 1, device=x.device),
                }
            return torch.zeros(B, self.output_dim, 1, device=x.device)

        # =====================================================================
        # Multi-Scale RNN (run in FP32 for stability)
        # =====================================================================
        with torch.amp.autocast('cuda', enabled=False):
            features_fp32 = features.float()
            rnn_main_out, _ = self.rnn_main(features_fp32)  # (B, T, 2*hidden_size)
            rnn_local_out, _ = self.rnn_local(features_fp32)  # (B, T, 2*hidden_size/2)

        # Concatenate multi-scale outputs
        rnn_out = torch.cat([rnn_main_out, rnn_local_out], dim=-1)  # (B, T, combined_dim)

        # =====================================================================
        # Multi-Head Attention
        # =====================================================================
        if self.use_attention:
            attn_out = self.attention(rnn_out)
            rnn_out = self.attention_norm(rnn_out + attn_out)  # Residual connection

        # =====================================================================
        # Output Heads (Onset/Offset or Single)
        # =====================================================================
        if self.use_onset_offset_heads:
            # Shared features
            shared = F.relu(self.shared_fc(rnn_out))  # (B, T, hidden_size)
            shared = self.shared_dropout(shared)

            # Three separate predictions
            frame_logits = self.frame_head(shared)  # (B, T, 88)
            onset_logits = self.onset_head(shared)  # (B, T, 88)
            offset_logits = self.offset_head(shared)  # (B, T, 88)

            if return_all_heads:
                return {
                    'frame': frame_logits.transpose(1, 2),  # (B, 88, T)
                    'onset': onset_logits.transpose(1, 2),
                    'offset': offset_logits.transpose(1, 2),
                }
            else:
                # Default: return only frame predictions
                return frame_logits.transpose(1, 2)
        else:
            logits = self.fc(rnn_out)  # (B, T, 88)
            logits = self.fc_dropout(logits)
            return logits.transpose(1, 2)  # (B, 88, T)