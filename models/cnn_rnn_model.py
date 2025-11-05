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

        # CNN Frontend
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (n_mels // 2, T // 2)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample frequency only
        )

        # Recurrent Layer (temporal modeling)
        # Flatten frequency dimension → time sequence
        # Input to LSTM = channels * freq_bins_after_cnn
        freq_bins_after = n_mels // 4  # due to 2x2 and 2x1 pooling
        lstm_input_size = 64 * freq_bins_after

        self.rnn = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        # Output Projection (frame-wise classification)
        self.fc = nn.Linear(hidden_size * 2, 88)  # 88 piano keys

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        """
        B, _, _, T = x.shape

        # CNN
        features = self.cnn(x)  # (B, C, F', T')
        B, C, F, T_new = features.shape

        # Prepare for RNN: move time to dimension 1 and flatten frequency axis
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, T', C, F')
        features = features.view(B, T_new, C * F)  # (B, T', lstm_input_size)

        # RNN
        rnn_out, _ = self.rnn(features)  # (B, T', 2*hidden_size)

        # Output
        logits = self.fc(rnn_out)  # (B, T', 88)
        logits = logits.permute(0, 2, 1)  # (B, 88, T')
        return logits