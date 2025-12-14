import torch
import torch.nn as nn
from .remi_tokenizer import REMITokenizer

try:
    # Hugging Face imports (optional). If not installed, init will raise an informative error.
    from transformers import AutoFeatureExtractor, AutoModel
    _HF_AVAILABLE = True
except Exception:
    AutoFeatureExtractor = None
    AutoModel = None
    _HF_AVAILABLE = False


class ASTModel(nn.Module):
    """
    Audio Spectrogram Transformer encoder + Transformer decoder for REMI token generation.

    - Uses a pretrained AST encoder from Hugging Face (specified by `pretrained_model_name`).
    - Initially freezes encoder weights by default to reduce compute.
    - Transformer decoder generates REMI tokens autoregressively (teacher forcing during training).

    Input to forward():
      - waveforms: Tensor[B, L] or list of 1D Tensors (raw audio in float, range [-1,1])
      - sampling_rate: int (e.g., 16000)
      - targets (optional): LongTensor[B, T] of token ids for teacher forcing

    Returns:
      - If targets provided -> logits: Tensor[B, T, vocab_size]
      - Else -> generated token ids: Tensor[B, gen_len]
    """

    def __init__(
        self,
        pretrained_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        use_mock_encoder: bool = False,
        freeze_encoder: bool = True,
        remi_vocab_size: int = 512,
        decoder_layers: int = 4,
        decoder_dim: int = 384,
        decoder_heads: int = 6,
        dropout: float = 0.2,
        max_output_len: int = 1024,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.pretrained_model_name = pretrained_model_name
        self.freeze_encoder = freeze_encoder
        self.remi_vocab_size = remi_vocab_size
        self.decoder_dim = decoder_dim
        self.max_output_len = max_output_len
        self.use_mock_encoder = use_mock_encoder

        if self.use_mock_encoder:
            # Build a tiny mock feature extractor + encoder for unit tests (no HF download)
            class _MockFeatureExtractor:
                def __init__(self, hidden_size=64):
                    self.hidden_size = hidden_size

                def __call__(self, waveforms, sampling_rate=None, return_tensors="pt", padding=True):
                    # Return a dummy tensor shaped (B, S, hidden)
                    import numpy as _np
                    if isinstance(waveforms, list):
                        B = len(waveforms)
                        max_len = max([w.shape[0] if hasattr(w, 'shape') else len(w) for w in waveforms])
                    else:
                        waveforms = [waveforms]
                        B = 1
                        max_len = waveforms[0].shape[0]
                    S = max(1, max_len // 160)  # coarse time dimension
                    return {"input_values": torch.randn(B, S, self.hidden_size)}

            class _MockEncoder(nn.Module):
                def __init__(self, hidden_size=64):
                    super().__init__()
                    self.config = type("C", (), {"hidden_size": hidden_size})

                def forward(self, **kwargs):
                    x = kwargs.get("input_values")
                    # assume x is (B, S, H)
                    return type("O", (), {"last_hidden_state": x})

            self.feature_extractor = _MockFeatureExtractor(hidden_size=decoder_dim)
            self.encoder = _MockEncoder(hidden_size=decoder_dim)
        else:
            if not _HF_AVAILABLE:
                raise ImportError(
                    "The ASTModel requires `transformers` to be installed. "
                )

            # Feature extractor converts raw audio waveforms to log-mel patches expected by AST
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)

            # Pretrained AST encoder (we'll freeze its weights by default)
            self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # encoder hidden size (from model config)
        enc_hidden = getattr(self.encoder.config, "hidden_size", None)
        if enc_hidden is None:
            # fallback: try common attributes
            enc_hidden = getattr(self.encoder.config, "embed_dim", decoder_dim)

        # Freeze encoder if requested
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Project encoder features to decoder dimensionality
        self.enc_to_dec = nn.Linear(enc_hidden, decoder_dim)

        # Decoder token embeddings + positional embeddings
        self.token_emb = nn.Embedding(remi_vocab_size, decoder_dim)
        self.pos_emb = nn.Embedding(max_output_len, decoder_dim)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Final projection to REMI vocabulary
        self.output_fc = nn.Linear(decoder_dim, remi_vocab_size)

        # initialization helpers
        self._reset_parameters()

        self.to(device)

    def _reset_parameters(self):
        # small init for newly added heads
        nn.init.normal_(self.enc_to_dec.weight, mean=0.0, std=0.02)
        if self.enc_to_dec.bias is not None:
            nn.init.zeros_(self.enc_to_dec.bias)
        nn.init.normal_(self.output_fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_fc.bias)

    def _generate_square_subsequent_mask(self, sz: int):
        # PyTorch transformer expects float mask with -inf on illegal positions
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        return mask

    def forward(self, waveforms, sampling_rate: int = 16000, targets: torch.LongTensor = None, generate_max_len: int = 256):
        """
        Forward pass.

        Args:
            waveforms: Tensor[B, L] or List[Tensor[B_i]] of raw audio float32 in [-1,1]
            sampling_rate: sampling rate used by the feature extractor
            targets: (optional) LongTensor[B, T] for teacher forcing
            generate_max_len: when generating, maximum tokens to produce
        """
        device = next(self.parameters()).device

        # Ensure waveforms are 1-D numpy arrays (the HF AST feature extractor uses
        # torchaudio/kaldi fbank under the hood and expects 1-D arrays). Users may
        # pass torch tensors or numpy arrays; convert torch tensors to numpy here.
        import numpy as _np

        if isinstance(waveforms, torch.Tensor):
            # tensor of shape (B, L) -> list of 1-D numpy arrays
            waveforms = [w.cpu().numpy().astype(_np.float32) for w in waveforms]
        elif isinstance(waveforms, (list, tuple)):
            converted = []
            for w in waveforms:
                if isinstance(w, torch.Tensor):
                    converted.append(w.cpu().numpy().astype(_np.float32))
                else:
                    # ensure numpy array dtype
                    try:
                        arr = _np.asarray(w, dtype=_np.float32)
                    except Exception:
                        arr = _np.array(w).astype(_np.float32)
                    converted.append(arr)
            waveforms = converted

        # Use feature_extractor to prepare inputs for encoder. It accepts lists or tensors.
        # It returns tensors on CPU by default; move them to device.
        # Ensure each waveform has a minimal length compatible with Kaldi fbank
        # (default frame_length in HF AST feature extractor is 25 ms -> 0.025 * sr samples)
        try:
            min_length = int(0.025 * float(sampling_rate)) if sampling_rate is not None else 400
        except Exception:
            min_length = 400
        for i, w in enumerate(waveforms):
            # ensure flat numpy array
            try:
                arr = _np.asarray(w, dtype=_np.float32).squeeze()
            except Exception:
                arr = _np.array(w, dtype=_np.float32).squeeze()
            if arr.ndim != 1:
                arr = arr.flatten()
            if arr.size < min_length:
                pad = _np.zeros(min_length - arr.size, dtype=_np.float32)
                arr = _np.concatenate([arr, pad])
            waveforms[i] = arr

        inputs = self.feature_extractor(waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        # Move all tensors to model device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # Run encoder. Because encoder is large and often frozen, run under no_grad when frozen.
        if self.freeze_encoder:
            with torch.no_grad():
                enc_outputs = self.encoder(**inputs)
        else:
            enc_outputs = self.encoder(**inputs)

        # Hugging Face models often put sequence features in last_hidden_state
        enc_feats = getattr(enc_outputs, "last_hidden_state", None)
        if enc_feats is None:
            # try other common keys
            enc_feats = enc_outputs[0]

        # Project to decoder dim
        # enc_feats shape: (B, S, enc_hidden)
        dec_memory = self.enc_to_dec(enc_feats)  # (B, S, D)

        # Transformer decoder expects memory shape (S, B, D)
        memory = dec_memory.permute(1, 0, 2).contiguous()

        if targets is None:
            # Greedy autoregressive generation (CPU/GPU-friendly, not optimized)
            return self.generate(memory, sos_id=0, max_len=generate_max_len)

        # Teacher forcing path: produce logits for all time steps in targets
        # If targets is a piano-roll batch (B, 88, T) convert to token sequences
        if torch.is_tensor(targets) and targets.dim() == 3:
            # targets assumed (B, 88, T) or (B, T, 88)
            B = targets.size(0)
            tokenizer = REMITokenizer(vocab_size=self.remi_vocab_size)
            max_len = min(self.max_output_len, generate_max_len)
            seqs = []
            for i in range(B):
                pr = targets[i].detach().cpu()
                # encoder expects (88, T) or (T,88)  tokenizer will handle transposes
                seq = tokenizer.encode_from_pianoroll(pr, max_len=max_len)
                seqs.append(seq)
            targets = torch.as_tensor(seqs, dtype=torch.long, device=device)

        # If targets is a list of piano-rolls (numpy/torch), convert to token sequences first
        if not torch.is_tensor(targets) and isinstance(targets, (list, tuple)):
            # inspect first element to decide if it's piano-roll-like
            first = targets[0]
            is_pr = False
            try:
                import numpy as _np
                if isinstance(first, torch.Tensor) and first.dim() == 2 and (first.size(0) == 88 or first.size(1) == 88):
                    is_pr = True
                elif isinstance(first, (_np.ndarray, list)):
                    arr = _np.asarray(first)
                    if arr.ndim == 2 and (arr.shape[0] == 88 or arr.shape[1] == 88):
                        is_pr = True
            except Exception:
                is_pr = False

            if is_pr:
                tokenizer = REMITokenizer(vocab_size=self.remi_vocab_size)
                max_len = min(self.max_output_len, generate_max_len)
                seqs = []
                for pr in targets:
                    if isinstance(pr, torch.Tensor):
                        p = pr.detach().cpu()
                    else:
                        import numpy as _np
                        p = torch.as_tensor(_np.asarray(pr), dtype=torch.float32)
                    seq = tokenizer.encode_from_pianoroll(p, max_len=max_len)
                    seqs.append(seq)
                targets = torch.as_tensor(seqs, dtype=torch.long, device=device)
            else:
                targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        else:
            targets = targets.to(device)

        # If there's an extra trailing singleton dim (e.g. shape (B, T, 1)), squeeze it
        if targets.dim() > 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)

        if targets.dim() != 2:
            raise ValueError(f"Expected targets of shape (B, T) after normalization, got {tuple(targets.shape)}")

        B, T = targets.shape
        positions = torch.arange(T, device=targets.device).unsqueeze(0).expand(B, T)
        tgt_emb = self.token_emb(targets) + self.pos_emb(positions)

        # Permute to shape (T, B, D) expected by nn.TransformerDecoder
        tgt_emb = tgt_emb.permute(1, 0, 2).contiguous()

        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)

        # Decoder returns (T, B, D)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        dec_out = dec_out.permute(1, 0, 2).contiguous()  # (B, T, D)
        logits = self.output_fc(dec_out)  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        memory,
        sos_id: int = 0,
        max_len: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        mask_sos: bool = True,
        repetition_penalty: float = 0.0,
    ):
        """Autoregressive generation from the decoder using provided encoder memory.

        Backwards-compatible defaults produce the previous greedy behavior
        (do_sample=False, mask_sos=False, repetition_penalty=0.0).

        Args:
            memory: (S, B, D) encoder memory
            sos_id: start token id
            max_len: max tokens to generate
            do_sample: whether to sample from the softmax (vs argmax)
            temperature: softmax temperature when sampling
            top_k: if >0, restrict sampling to top_k logits
            mask_sos: if True, forbid emitting sos token after step 0
            repetition_penalty: float >=0. Subtracts penalty*count[token] from logits

        Returns:
            Tensor[B, L] of generated token ids
        """
        device = memory.device
        S, B, D = memory.shape
        vocab_size = self.remi_vocab_size

        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        # counts per batch item for repetition penalty
        if repetition_penalty and repetition_penalty > 0.0:
            counts = torch.zeros((B, vocab_size), dtype=torch.long, device=device)
            # initialize with sos count
            counts.scatter_add_(1, generated, torch.ones_like(generated, dtype=torch.long))
        else:
            counts = None

        def top_k_logits(logits, k: int):
            if k <= 0:
                return logits
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(logits < min_values, torch.full_like(logits, float("-1e9")), logits)

        for step in range(max_len):
            positions = torch.arange(generated.size(1), device=device).unsqueeze(0).expand(B, -1)
            tgt_emb = self.token_emb(generated) + self.pos_emb(positions)
            tgt = tgt_emb.permute(1, 0, 2).contiguous()  # (T, B, D)
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(device)
            dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # (T, B, D)
            last = dec_out[-1]  # (B, D)
            logits = self.output_fc(last)  # (B, V)

            # Optionally forbid producing sos after the first position
            if mask_sos and step > 0:
                if 0 <= sos_id < logits.size(-1):
                    logits[:, sos_id] = float("-1e9")

            # Apply repetition penalty (simple count-based subtraction)
            if counts is not None:
                # subtract penalty * counts (counts is integer tensor)
                logits = logits - repetition_penalty * counts.float()

            if do_sample:
                # sampling path: apply temperature and top_k filtering
                sample_logits = logits / max(1e-8, float(temperature))
                if top_k > 0:
                    sample_logits = top_k_logits(sample_logits, top_k)
                probs = torch.softmax(sample_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # greedy argmax
                next_tokens = logits.argmax(dim=-1, keepdim=True)

            # update counts if used
            if counts is not None:
                counts.scatter_add_(1, next_tokens, torch.ones_like(next_tokens, dtype=torch.long))

            generated = torch.cat([generated, next_tokens], dim=1)

        return generated[:, 1:]
