import pytest
import torch
from algorithms.chunking_buffer import ChunkingBuffer

CHUNK_SIZE = 5
ACTION_SIZE = 6


@pytest.fixture
def buf():
    return ChunkingBuffer(chunk_size=CHUNK_SIZE, action_size=ACTION_SIZE)


def _chunk(fill: float) -> torch.Tensor:
    return torch.full((CHUNK_SIZE, ACTION_SIZE), fill)


def test_single_chunk_returns_correct_action(buf):
    chunk = _chunk(1.0)
    buf.add(chunk, t=0)

    action = buf.get_action(t=2)

    assert action.shape == (ACTION_SIZE,)
    assert torch.allclose(action, chunk[2])


def test_reset_clears_buffer(buf):
    buf.add(_chunk(1.0), t=0)
    buf.reset()

    assert len(buf.buffer) == 0


def test_eviction_removes_stale_chunks(buf):
    buf.add(_chunk(1.0), t=0)
    buf.add(_chunk(2.0), t=CHUNK_SIZE)  # t=0 chunk is now stale

    assert len(buf.buffer) == 1
    assert buf.buffer[0][0] == CHUNK_SIZE


def test_weighted_average_weights_recent_higher(buf):
    # two overlapping chunks; the later one should have higher weight
    chunk_old = _chunk(0.0)
    chunk_new = _chunk(1.0)
    buf.add(chunk_old, t=0)
    buf.add(chunk_new, t=1)

    # at t=1, chunk_old provides action[1]=0.0, chunk_new provides action[0]=1.0
    # recent chunk gets weight exp_weight^0 = 1.0, old gets exp_weight^1 = 0.9
    action = buf.get_action(t=1)
    w_old = 0.9
    w_new = 1.0
    expected = (0.0 * w_old + 1.0 * w_new) / (w_old + w_new)

    assert action.shape == (ACTION_SIZE,)
    assert torch.allclose(action, torch.full((ACTION_SIZE,), expected), atol=1e-5)


def test_get_action_single_candidate_no_weighting_needed(buf):
    chunk = torch.arange(CHUNK_SIZE * ACTION_SIZE, dtype=torch.float).view(
        CHUNK_SIZE, ACTION_SIZE
    )
    buf.add(chunk, t=0)

    for t in range(CHUNK_SIZE):
        action = buf.get_action(t)
        assert torch.allclose(action, chunk[t])
