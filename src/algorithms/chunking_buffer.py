import torch


class ChunkingBuffer:
    def __init__(self, chunk_size: int, action_size: int, exp_weight: float = 0.9):
        self.chunk_size = chunk_size
        self.action_size = action_size
        self.exp_weight = exp_weight
        self.buffer = []
        self.start_t = 0

    def reset(self):
        self.buffer.clear()
        self.start_t = 0

    def add(self, actions: torch.Tensor, t: int):
        self._evict_stale(t)
        self.buffer.append((t, actions))

    def _evict_stale(self, t: int):
        self.buffer = [
            (added_at, chunk)
            for added_at, chunk in self.buffer
            if t - added_at < self.chunk_size
        ]

    def get_action(self, t: int) -> torch.Tensor:
        candidates = []
        weights = []

        for i, (added_at, chunk) in enumerate(self.buffer):
            action_idx = t - added_at
            if 0 <= action_idx < self.chunk_size:
                candidates.append(chunk[action_idx])
                weights.append(self.exp_weight ** (len(self.buffer) - i - 1))

        candidates = torch.stack(candidates)
        weights = torch.tensor(
            weights, dtype=candidates.dtype, device=candidates.device
        )
        weights = weights / weights.sum()

        return (candidates * weights.unsqueeze(1)).sum(dim=0)
