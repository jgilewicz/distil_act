import pytest
import torch
from algorithms.act_policy import ACT, EncoderCVAE

B = 2
K = 2
ACTION_DIM = 6
JOINT_DIM = 6
EMBED_DIM = 256
LATENT_DIM = 128
CHUNK = 10


@pytest.fixture(scope="module")
def act():
    return ACT(
        action_dim=ACTION_DIM,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        joint_dim=JOINT_DIM,
        action_query_len=CHUNK,
    )


@pytest.fixture(scope="module")
def batch():
    return {
        "images": torch.rand(B, K, 3, 224, 224),
        "qpos": torch.rand(B, JOINT_DIM),
        "actions": torch.rand(B, CHUNK, ACTION_DIM),
    }


def test_training_forward_shapes(act, batch):
    pred, mu, logvar = act(batch["images"], batch["qpos"], batch["actions"])

    assert pred.shape == batch["actions"].shape
    assert mu.shape == (B, LATENT_DIM)
    assert logvar.shape == (B, LATENT_DIM)


def test_inference_forward_shape(act, batch):
    with torch.no_grad():
        out = act(batch["images"], batch["qpos"])

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, CHUNK, ACTION_DIM)


def test_inference_returns_tensor_not_tuple(act, batch):
    with torch.no_grad():
        out = act(batch["images"], batch["qpos"])

    assert not isinstance(out, tuple)


def test_encoder_cvae_shapes():
    cvae = EncoderCVAE(
        embed_dim=EMBED_DIM, latent_dim=LATENT_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM
    )
    actions = torch.rand(B, CHUNK, ACTION_DIM)
    joints = torch.rand(B, JOINT_DIM)

    z, mu, logvar = cvae(actions, joints)

    assert z.shape == (B, LATENT_DIM)
    assert mu.shape == (B, LATENT_DIM)
    assert logvar.shape == (B, LATENT_DIM)
