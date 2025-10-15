import torch
import torch.nn as nn
from .cnn_rnn_model import CNNRNNModel
import torch.nn.functional as F

class TranscriptionModel(nn.Module):
    """
    High-level wrapper for automatic music transcription models.
    Handles:
      - model selection (currently CNN+RNN)
      - forward pass
      - loss computation
      - prediction thresholding (optional)
    """

    def __init__(
        self,
        model_type: str = "cnn_rnn",
        n_mels: int = 229,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.device = device

        if self.model_type in ["cnn_rnn", "cnn+rnn"]:
            self.model = CNNRNNModel(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Binary cross-entropy loss (note-wise activation prediction)
        self.criterion = nn.BCEWithLogitsLoss()

        self.to(device)

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            logits: (B, 88, T')
        """
        return self.model(x)

    # -------------------------------------------------------------------------
    def compute_loss(self, logits, targets):
        # Align time if needed
        if logits.shape[-1] != targets.shape[-1]:
            # upsample logits along time to target length
            logits = F.interpolate(logits, size=targets.shape[-1],
                                mode="linear", align_corners=False)
        return self.criterion(logits, targets)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        """
        Inference method for note activation probabilities.
        Args:
            x: (B, 1, n_mels, T)
            threshold: float, decision threshold for activation
        Returns:
            piano_roll: binary tensor (B, 88, T')
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        piano_roll = (probs > threshold).float()
        return piano_roll