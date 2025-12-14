import torch
from typing import List

class REMITokenizer:
    """Simple REMI-like tokenizer for testing and small-scale experiments.

    Vocabulary layout (small, deterministic):
      - 0: <sos>
      - 1: <eos>
      - 2: <pad>
      - 10-97: NOTE_ON_0 .. NOTE_ON_87  (pitch 0..87)
      - 110-197: NOTE_OFF_0 .. NOTE_OFF_87
      - 210-242: VELOCITY_0 .. VELOCITY_32 (coarse velocities)
      - 300-399: TIME_SHIFT_1 .. TIME_SHIFT_100 (time shift in frames)

    This is intentionally small and not a full REMI implementation. It
    provides encode/decode helpers to go between piano-rolls and token sequences
    so the transformer decoder can be trained/tested end-to-end.
    """

    def __init__(self, vocab_size: int = 512, max_time_shift: int = 100):
        self.vocab_size = vocab_size
        self.sos = 0
        self.eos = 1
        self.pad = 2

        self.note_on_base = 10
        self.note_off_base = 110
        self.velocity_base = 210
        self.time_shift_base = 300

        self.max_time_shift = max_time_shift

    def note_on_id(self, pitch: int) -> int:
        return self._safe_id(self.note_on_base + int(pitch))

    def note_off_id(self, pitch: int) -> int:
        return self._safe_id(self.note_off_base + int(pitch))

    def velocity_id(self, vel_idx: int) -> int:
        return self._safe_id(self.velocity_base + int(vel_idx))

    def time_shift_id(self, frames: int) -> int:
        frames = max(1, min(self.max_time_shift, int(frames)))
        return self._safe_id(self.time_shift_base + (frames - 1))

    def _safe_id(self, idx: int) -> int:
        """Ensure returned token id fits within vocab size; if not, return pad id.

        This prevents IndexError when the tokenizer's internal base offsets exceed
        the provided vocab_size (useful for tests with small vocabularies).
        """
        if idx < 0 or idx >= self.vocab_size:
            return self.pad
        return int(idx)

    def encode_from_pianoroll(self, piano_roll: torch.Tensor, frame_rate: int = 100, max_len: int = 256) -> List[int]:
        """Encode a piano-roll (88 x T) into a token sequence.

        Naive algorithm:
          - iterate frames t=0..T-1
          - if a note turns on at t (was 0 at t-1), emit NOTE_ON(pitch) + VELOCITY(default)
          - if a note turns off at t (was 1 at t-1), emit NOTE_OFF(pitch)
          - after processing frame, emit TIME_SHIFT_1

        Returns token id list (including <sos> and <eos>) truncated/padded to max_len.
        """
        # piano_roll: Tensor[88, T] or [T,88]
        pr = piano_roll
        if pr.dim() == 2 and pr.shape[0] != 88 and pr.shape[1] == 88:
            pr = pr.t()
        pr = pr.clone().float()
        if pr.shape[0] == 88:
            pr = pr.t()
        # now pr is (T, 88)
        T, P = pr.shape
        assert P == 88, "piano_roll must have 88 pitches"

        seq = [self.sos]
        prev = torch.zeros(88, dtype=torch.float32)
        t = 0
        while t < T:
            frame = pr[t]
            # note on
            ons = (frame > 0.5) & (prev <= 0.5)
            offs = (frame <= 0.5) & (prev > 0.5)
            for p in torch.nonzero(ons).squeeze(-1).tolist():
                seq.append(self.note_on_id(p))
                # coarse velocity index (single default)
                seq.append(self.velocity_id(0))
            for p in torch.nonzero(offs).squeeze(-1).tolist():
                seq.append(self.note_off_id(p))

            # Count consecutive frames with no changes to merge time shifts
            num_frames = 1
            has_events = (ons.any() or offs.any())

            # If no events in this frame and we're not at the last frame,
            # look ahead to merge consecutive silent frames
            if not has_events and t + 1 < T:
                while t + num_frames < T and num_frames < self.max_time_shift:
                    next_frame = pr[t + num_frames]
                    next_ons = (next_frame > 0.5) & (frame <= 0.5)
                    next_offs = (next_frame <= 0.5) & (frame > 0.5)
                    if next_ons.any() or next_offs.any():
                        break
                    num_frames += 1
                    frame = next_frame

            # Emit time shift
            seq.append(self.time_shift_id(num_frames))
            prev = frame
            t += num_frames

            if len(seq) >= max_len - 1:
                break

        seq.append(self.eos)
        # pad or trim
        if len(seq) < max_len:
            seq = seq + [self.pad] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def decode_to_pianoroll(self, tokens: List[int], max_T: int = 1024) -> torch.Tensor:
        """Decode token sequence into a piano-roll (88, T) using simple semantics.

        Note: TIME_SHIFT tokens advance time by N frames.
        """
        T = max_T
        pr = torch.zeros(88, T, dtype=torch.float32)
        t = 0
        active = set()
        i = 0
        while i < len(tokens) and t < T:
            tok = tokens[i]
            if tok == self.sos:
                i += 1
                continue
            if tok == self.eos:
                break
            if self.note_on_base <= tok < self.note_off_base:
                pitch = tok - self.note_on_base
                active.add(pitch)
                # default velocity
                pr[pitch, t] = 1.0
                i += 1
                continue
            if self.note_off_base <= tok < self.velocity_base:
                pitch = tok - self.note_off_base
                if pitch in active:
                    active.remove(pitch)
                i += 1
                continue
            if self.time_shift_base <= tok < self.time_shift_base + self.max_time_shift:
                frames = (tok - self.time_shift_base) + 1
                # keep active notes for frames
                for ft in range(frames):
                    for p in active:
                        if t < T:
                            pr[p, t] = 1.0
                    t += 1
                i += 1
                continue
            # velocity or unknown token: skip
            i += 1

        return pr[:, :t]
