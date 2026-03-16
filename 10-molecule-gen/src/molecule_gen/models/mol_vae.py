"""Molecular Variational Autoencoder (VAE) for SMILES-based molecule generation.

Implements a sequence-to-sequence VAE architecture that learns a continuous latent
representation of molecular structures encoded as SMILES strings. The latent space
enables smooth interpolation between molecules and property-conditioned generation
for targeted drug design.

Architecture:
    Encoder: GRU-based recurrent encoder maps variable-length SMILES token
             sequences to a fixed-dimensional latent vector via reparameterization.
    Decoder: GRU-based autoregressive decoder reconstructs SMILES token sequences
             from latent vectors, optionally conditioned on molecular properties.

Loss:
    ELBO = Reconstruction (cross-entropy) + beta * KL divergence
    Beta-VAE annealing prevents posterior collapse during early training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """Configuration for the Molecular VAE architecture."""

    vocab_size: int = 45
    embedding_dim: int = 64
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    latent_dim: int = 128
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    max_sequence_length: int = 120
    dropout: float = 0.1
    num_property_conditions: int = 0  # 0 = unconditioned generation
    teacher_forcing_ratio: float = 0.9


class MolecularEncoder(nn.Module):
    """GRU-based encoder that maps SMILES token sequences to latent distributions.

    Processes tokenized SMILES through an embedding layer and bidirectional GRU,
    then projects the final hidden state to mean (mu) and log-variance (logvar)
    vectors that parameterize the approximate posterior q(z|x).
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Bidirectional doubles the hidden size
        encoder_output_dim = config.encoder_hidden_dim * 2

        self.fc_mu = nn.Linear(encoder_output_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, config.latent_dim)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode a batch of tokenized SMILES to latent distribution parameters.

        Args:
            x: Token indices of shape (batch, seq_len).
            lengths: True lengths of each sequence for packing (optional).

        Returns:
            Tuple of (mu, logvar), each of shape (batch, latent_dim).
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(embedded)

        # hidden: (num_layers * 2, batch, hidden_dim) -- bidirectional
        # Concatenate the final forward and backward hidden states
        hidden_fwd = hidden[-2]  # last layer forward
        hidden_bwd = hidden[-1]  # last layer backward
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=-1)  # (batch, hidden*2)

        mu = self.fc_mu(hidden_cat)
        logvar = self.fc_logvar(hidden_cat)

        return mu, logvar


class MolecularDecoder(nn.Module):
    """GRU-based autoregressive decoder that reconstructs SMILES from latent vectors.

    Supports optional property conditioning by concatenating property values
    to the latent vector before decoding. Uses teacher forcing during training
    and greedy/sampling-based decoding during generation.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        latent_input_dim = config.latent_dim + config.num_property_conditions

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )
        # Project latent vector to initial decoder hidden state
        self.latent_to_hidden = nn.Linear(
            latent_input_dim, config.decoder_hidden_dim * config.num_decoder_layers
        )
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.num_decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_decoder_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(config.decoder_hidden_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        z: Tensor,
        target: Tensor | None = None,
        properties: Tensor | None = None,
    ) -> Tensor:
        """Decode latent vectors to SMILES token logits.

        Args:
            z: Latent vectors of shape (batch, latent_dim).
            target: Ground-truth token indices for teacher forcing (batch, seq_len).
                    If None, performs autoregressive generation.
            properties: Optional property values for conditioned generation
                        of shape (batch, num_properties).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size = z.size(0)

        # Concatenate property conditions if provided
        if properties is not None:
            z = torch.cat([z, properties], dim=-1)

        # Initialize hidden state from latent vector
        hidden = self.latent_to_hidden(z)  # (batch, hidden * num_layers)
        hidden = hidden.view(
            batch_size, self.config.num_decoder_layers, self.config.decoder_hidden_dim
        )
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden)

        if target is not None:
            # Teacher forcing: feed ground-truth tokens as input
            seq_len = target.size(1)
            embedded = self.dropout(self.embedding(target))  # (batch, seq_len, embed)
            output, _ = self.gru(embedded, hidden)
            logits = self.output_projection(output)  # (batch, seq_len, vocab_size)
        else:
            # Autoregressive generation
            logits_list: list[Tensor] = []
            # Start token (index 1 = SOS)
            current_input = torch.ones(batch_size, 1, dtype=torch.long, device=z.device)

            for _ in range(self.config.max_sequence_length):
                embedded = self.dropout(self.embedding(current_input))
                output, hidden = self.gru(embedded, hidden)
                step_logits = self.output_projection(output)  # (batch, 1, vocab)
                logits_list.append(step_logits)

                # Greedy selection for next input
                current_input = step_logits.argmax(dim=-1)  # (batch, 1)

            logits = torch.cat(logits_list, dim=1)

        return logits


class MolecularVAE(nn.Module):
    """Full Variational Autoencoder for molecular SMILES generation.

    Combines the encoder and decoder with the reparameterization trick to enable
    gradient-based optimization of the Evidence Lower Bound (ELBO). Supports
    beta-VAE annealing to balance reconstruction quality and latent space
    regularity, preventing posterior collapse in early training.

    Key capabilities:
        - Encode existing molecules to latent representations
        - Decode latent vectors to SMILES strings
        - Interpolate between molecules in latent space
        - Condition generation on desired molecular properties
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MolecularEncoder(config)
        self.decoder = MolecularDecoder(config)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Apply the reparameterization trick: z = mu + std * epsilon.

        Enables backpropagation through the stochastic sampling step by
        expressing z as a deterministic function of the distribution
        parameters and an independent noise source.

        Args:
            mu: Mean of the approximate posterior, shape (batch, latent_dim).
            logvar: Log-variance of the approximate posterior, shape (batch, latent_dim).

        Returns:
            Sampled latent vector z of shape (batch, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + std * epsilon
        return mu

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        properties: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass: encode, reparameterize, decode.

        Args:
            x: Input token indices of shape (batch, seq_len).
            lengths: True sequence lengths for packing.
            properties: Optional property conditions of shape (batch, num_properties).

        Returns:
            Tuple of (logits, mu, logvar):
                logits: Reconstructed token logits (batch, seq_len, vocab_size).
                mu: Posterior mean (batch, latent_dim).
                logvar: Posterior log-variance (batch, latent_dim).
        """
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, target=x, properties=properties)

        return logits, mu, logvar

    def encode(self, x: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode input SMILES tokens to latent distribution parameters.

        Args:
            x: Token indices of shape (batch, seq_len).
            lengths: True sequence lengths.

        Returns:
            Tuple of (mu, logvar), each (batch, latent_dim).
        """
        return self.encoder(x, lengths)

    def decode(
        self,
        z: Tensor,
        properties: Tensor | None = None,
    ) -> Tensor:
        """Decode latent vectors to SMILES token logits (autoregressive).

        Args:
            z: Latent vectors of shape (batch, latent_dim).
            properties: Optional property conditions (batch, num_properties).

        Returns:
            Token logits of shape (batch, max_seq_len, vocab_size).
        """
        return self.decoder(z, target=None, properties=properties)

    def sample(
        self,
        num_samples: int,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        properties: Tensor | None = None,
    ) -> Tensor:
        """Sample novel molecules by decoding random latent vectors.

        Draws z ~ N(0, I) and decodes to SMILES token sequences.

        Args:
            num_samples: Number of molecules to generate.
            device: Target device for tensor allocation.
            temperature: Sampling temperature; higher = more diverse, lower = more conservative.
            properties: Optional property conditions (num_samples, num_properties).

        Returns:
            Token logits of shape (num_samples, max_seq_len, vocab_size).
        """
        z = torch.randn(num_samples, self.config.latent_dim, device=device) * temperature
        return self.decode(z, properties=properties)

    def interpolate(
        self,
        z_start: Tensor,
        z_end: Tensor,
        num_steps: int = 10,
        properties: Tensor | None = None,
    ) -> Tensor:
        """Interpolate linearly between two points in latent space.

        Enables smooth "molecular morphing" between two molecules by
        generating intermediate structures along the geodesic in latent space.

        Args:
            z_start: Starting latent vector (1, latent_dim).
            z_end: Ending latent vector (1, latent_dim).
            num_steps: Number of interpolation steps including endpoints.
            properties: Optional property conditions (num_steps, num_properties).

        Returns:
            Token logits for all interpolation steps (num_steps, max_seq_len, vocab_size).
        """
        alphas = torch.linspace(0.0, 1.0, num_steps, device=z_start.device)
        z_interp = torch.stack(
            [(1 - a) * z_start.squeeze(0) + a * z_end.squeeze(0) for a in alphas]
        )
        return self.decode(z_interp, properties=properties)


def vae_loss(
    logits: Tensor,
    targets: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    pad_idx: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the VAE ELBO loss = Reconstruction + beta * KL divergence.

    The reconstruction loss uses token-level cross-entropy, ignoring padding.
    KL divergence regularizes the posterior q(z|x) toward the standard normal
    prior p(z) = N(0, I).

    Beta-VAE annealing (Bowman et al., 2016) gradually increases beta from 0
    to 1 during training to prevent posterior collapse, where the decoder
    learns to ignore the latent code.

    Args:
        logits: Predicted token logits (batch, seq_len, vocab_size).
        targets: Ground-truth token indices (batch, seq_len).
        mu: Posterior mean (batch, latent_dim).
        logvar: Posterior log-variance (batch, latent_dim).
        beta: KL weight for beta-VAE annealing schedule.
        pad_idx: Padding token index to ignore in cross-entropy.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss).
    """
    # Reconstruction loss: cross-entropy over token predictions
    batch_size, seq_len, vocab_size = logits.shape
    recon_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="mean",
    )

    # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # Closed-form solution for Gaussian distributions:
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def kl_annealing_schedule(
    current_step: int,
    total_annealing_steps: int,
    strategy: str = "linear",
) -> float:
    """Compute beta weight for KL annealing to prevent posterior collapse.

    Gradually increases the KL divergence weight from 0 to 1 over a specified
    number of training steps. This prevents the common failure mode where the
    decoder ignores the latent code early in training.

    Args:
        current_step: Current training step.
        total_annealing_steps: Number of steps over which to anneal beta to 1.0.
        strategy: Annealing schedule type - 'linear' or 'cyclical'.

    Returns:
        Beta value in [0, 1].
    """
    if strategy == "linear":
        return min(1.0, current_step / max(1, total_annealing_steps))
    elif strategy == "cyclical":
        # Cyclical annealing (Fu et al., 2019) - repeating linear ramps
        cycle_length = total_annealing_steps
        position_in_cycle = current_step % cycle_length
        ramp_fraction = 0.5  # proportion of cycle spent ramping
        ramp_steps = int(cycle_length * ramp_fraction)
        if position_in_cycle < ramp_steps:
            return position_in_cycle / max(1, ramp_steps)
        return 1.0
    else:
        raise ValueError(f"Unknown annealing strategy: {strategy}")
