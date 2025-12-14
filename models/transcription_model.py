import torch
import torch.nn as nn
from .cnn_rnn_model import CNNRNNModel, CNNRNNModelLarge
import torch.nn.functional as F

class TranscriptionModel(nn.Module):
    """
    High-level wrapper for automatic music transcription models.
    Handles:
      - model selection (CNN+RNN or CNN+RNN Large)
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
        use_attention: bool = True,
        use_onset_offset_heads: bool = True,
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.device = device
        self.use_onset_offset_heads = use_onset_offset_heads

        if self.model_type in ["cnn_rnn", "cnn+rnn"]:
            self.model = CNNRNNModel(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif self.model_type in ["cnn_rnn_large", "large"]:
            self.model = CNNRNNModelLarge(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_attention=use_attention,
                use_onset_offset_heads=use_onset_offset_heads,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Binary cross-entropy loss (note-wise activation prediction)
        self.criterion = nn.BCEWithLogitsLoss()

        self.to(device)

    def forward(self, x, return_all_heads=False):
        """
        Forward pass through the model.
        Args:
            x: (B, 1, n_mels, T)
            return_all_heads: For large model with onset/offset, return all predictions
        Returns:
            logits: (B, 88, T') or dict with 'frame', 'onset', 'offset' keys
        """
        if self.model_type in ["cnn_rnn_large", "large"] and self.use_onset_offset_heads:
            return self.model(x, return_all_heads=return_all_heads)
        else:
            return self.model(x)

    def compute_loss(self, logits, targets, lengths=None):
        """
        Compute BCE loss between logits and ground truth piano-roll.
        If sequence lengths are provided, padded time frames are masked out.

        For large model with onset/offset heads, computes combined loss.

        Args:
            logits: (B, 88, T') or dict with 'frame', 'onset', 'offset' keys
            targets: (B, 88, T')
            lengths: (B,) or None — number of valid time steps per sample
        """
        # Handle multi-head outputs from large model
        if isinstance(logits, dict):
            return self._compute_multi_head_loss(logits, targets, lengths)

        # Standard single-head loss
        # Align time if needed
        if logits.shape[-1] != targets.shape[-1]:
            logits = F.interpolate(logits, size=targets.shape[-1],
                                mode="linear", align_corners=False)

        # --- If no masking, just return regular BCE ---
        if lengths is None:
            return self.criterion(logits, targets)
        lengths = lengths.to(logits.device)

        # --- Apply mask ---
        B, P, T = logits.shape
        device = logits.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)  # shape (B, 1, T)

        # BCE without reduction
        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # shape (B, 88, T)

        # Zero out masked frames and average
        masked_loss = loss_per_elem * mask
        denom = mask.sum() * P  # total valid elements
        return masked_loss.sum() / denom.clamp_min(1)

    def _compute_multi_head_loss(self, logits_dict, targets, lengths=None):
        """
        Compute combined loss for onset/offset detection heads.

        Args:
            logits_dict: dict with 'frame', 'onset', 'offset' keys
            targets: (B, 88, T') ground truth piano-roll
            lengths: (B,) or None
        Returns:
            Combined weighted loss
        """
        # Compute onset/offset ground truth from frame targets
        # Onset: transition from 0 to 1
        # Offset: transition from 1 to 0
        onset_targets = torch.zeros_like(targets)
        offset_targets = torch.zeros_like(targets)

        if targets.shape[-1] > 1:
            # Onset: diff > 0 (note starts)
            onset_targets[:, :, 1:] = torch.clamp(targets[:, :, 1:] - targets[:, :, :-1], min=0)
            # Offset: diff < 0 (note ends)
            offset_targets[:, :, :-1] = torch.clamp(targets[:, :, :-1] - targets[:, :, 1:], min=0)

        # Compute individual losses
        frame_loss = self._compute_single_loss(logits_dict['frame'], targets, lengths)
        onset_loss = self._compute_single_loss(logits_dict['onset'], onset_targets, lengths)
        offset_loss = self._compute_single_loss(logits_dict['offset'], offset_targets, lengths)

        # Weighted combination (frame is most important, onset/offset help with boundaries)
        total_loss = 0.5 * frame_loss + 0.25 * onset_loss + 0.25 * offset_loss
        return total_loss

    def _compute_single_loss(self, logits, targets, lengths=None):
        """Helper to compute loss for a single prediction head."""
        # Align time if needed
        if logits.shape[-1] != targets.shape[-1]:
            logits = F.interpolate(logits, size=targets.shape[-1],
                                mode="linear", align_corners=False)

        if lengths is None:
            return self.criterion(logits, targets)

        lengths = lengths.to(logits.device)
        B, P, T = logits.shape
        device = logits.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)

        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        masked_loss = loss_per_elem * mask
        denom = mask.sum() * P
        return masked_loss.sum() / denom.clamp_min(1)

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