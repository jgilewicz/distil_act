from algorithms.embedding import ImageEmbedding
import torch
import torch.nn as nn


class EncoderCVAE(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 128,
        joint_dim: int = 6,
        action_dim: int = 6,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        z_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.z_encoder = nn.TransformerEncoder(z_encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.action_projection = nn.Linear(action_dim, embed_dim)
        self.joints_projection = nn.Linear(joint_dim, embed_dim)

        self.z_mu = nn.Linear(embed_dim, latent_dim)
        self.z_logvar = nn.Linear(embed_dim, latent_dim)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        actions: torch.Tensor,
        joints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = actions.shape[0]

        actions_tokens = self.action_projection(actions)
        joints_tokens = self.joints_projection(joints).unsqueeze(1)

        cls_token = self.cls_token.expand(B, -1, -1)
        encoder_input = torch.cat([cls_token, joints_tokens, actions_tokens], dim=1)
        encoded = self.z_encoder(encoder_input)

        z = encoded[:, 0, :]
        mu = self.z_mu(z)
        logvar = self.z_logvar(z)
        z = self.reparametrize(mu, logvar)

        return z, mu, logvar


class ACT(nn.Module):
    def __init__(
        self,
        action_dim: int = 6,
        embed_dim: int = 256,
        latent_dim: int = 128,
        joint_dim: int = 6,
        action_query_len: int = 50,
        nhead: int = 8,
        num_layers: int = 4,
        num_cameras: int = 2,
        teacher_latent_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.image_embedding = ImageEmbedding(
            embed_dim=embed_dim, num_cameras=num_cameras
        )
        self.joints_projection = nn.Linear(joint_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.encoder_cvae = EncoderCVAE(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            joint_dim=joint_dim,
            action_dim=action_dim,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.z_projection = nn.Linear(latent_dim, embed_dim)

        self.action_head = nn.Linear(embed_dim, action_dim)
        self.action_queries = nn.Embedding(action_query_len, embed_dim)

        if teacher_latent_dim is not None:
            self.latent_projection = nn.Linear(self.latent_dim, teacher_latent_dim)

    def forward(
        self,
        images: torch.Tensor,
        joints: torch.Tensor,
        actions: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        B = images.shape[0]

        image_tokens = self.image_embedding(images)
        joints_token = self.joints_projection(joints).unsqueeze(1)

        if actions is not None:
            z, mu, logvar = self.encoder_cvae(actions, joints)
        else:
            z = torch.zeros(B, self.latent_dim, device=images.device)
            mu = logvar = None

        z_token = self.z_projection(z).unsqueeze(1)

        encoder_input = torch.cat([image_tokens, joints_token, z_token], dim=1)
        encoded = self.encoder(encoder_input)

        action_queries = self.action_queries(
            torch.arange(self.action_queries.num_embeddings, device=images.device)
        )
        action_queries = action_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(action_queries, encoded)
        pred_actions = self.action_head(decoded)

        if actions is not None:
            return pred_actions, mu, logvar

        return pred_actions
