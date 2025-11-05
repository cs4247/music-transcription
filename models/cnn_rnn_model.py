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