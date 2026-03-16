"""Tests for the Molecular VAE model architecture and loss functions."""

from __future__ import annotations

import pytest
import torch

from molecule_gen.chemistry.smiles_processor import SMILESVocabulary
from molecule_gen.models.mol_vae import (
    MolecularDecoder,
    MolecularEncoder,
    MolecularVAE,
    VAEConfig,
    kl_annealing_schedule,
    vae_loss,
)

from .conftest import ASPIRIN_SMILES


class TestMolecularEncoder:
    """Tests for the GRU-based SMILES encoder."""

    def test_encoder_output_shape(self, vae_config: VAEConfig) -> None:
        """Encoder should produce mu and logvar of shape (batch, latent_dim)."""
        encoder = MolecularEncoder(vae_config)
        batch_size = 4
        seq_len = 30

        x = torch.randint(0, vae_config.vocab_size, (batch_size, seq_len))
        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, vae_config.latent_dim)
        assert logvar.shape == (batch_size, vae_config.latent_dim)

    def test_encoder_with_packed_sequences(self, vae_config: VAEConfig) -> None:
        """Encoder should handle variable-length sequences with packing."""
        encoder = MolecularEncoder(vae_config)
        batch_size = 4
        max_len = 30

        x = torch.randint(0, vae_config.vocab_size, (batch_size, max_len))
        lengths = torch.tensor([30, 25, 20, 15])

        mu, logvar = encoder(x, lengths=lengths)

        assert mu.shape == (batch_size, vae_config.latent_dim)
        assert logvar.shape == (batch_size, vae_config.latent_dim)

    def test_encoder_deterministic_in_eval(self, vae_config: VAEConfig) -> None:
        """Encoder outputs should be deterministic in eval mode."""
        encoder = MolecularEncoder(vae_config)
        encoder.eval()

        x = torch.randint(0, vae_config.vocab_size, (2, 20))

        with torch.no_grad():
            mu1, logvar1 = encoder(x)
            mu2, logvar2 = encoder(x)

        assert torch.allclose(mu1, mu2)
        assert torch.allclose(logvar1, logvar2)


class TestMolecularDecoder:
    """Tests for the GRU-based SMILES decoder."""

    def test_decoder_teacher_forcing(self, vae_config: VAEConfig) -> None:
        """Decoder with teacher forcing should output logits matching target length."""
        decoder = MolecularDecoder(vae_config)
        batch_size = 4
        seq_len = 30

        z = torch.randn(batch_size, vae_config.latent_dim)
        target = torch.randint(0, vae_config.vocab_size, (batch_size, seq_len))

        logits = decoder(z, target=target)

        assert logits.shape == (batch_size, seq_len, vae_config.vocab_size)

    def test_decoder_autoregressive(self, vae_config: VAEConfig) -> None:
        """Decoder without teacher forcing should autoregressively generate max_seq_len tokens."""
        decoder = MolecularDecoder(vae_config)
        batch_size = 2

        z = torch.randn(batch_size, vae_config.latent_dim)
        logits = decoder(z, target=None)

        assert logits.shape == (
            batch_size,
            vae_config.max_sequence_length,
            vae_config.vocab_size,
        )

    def test_decoder_with_property_conditioning(self) -> None:
        """Decoder should accept property conditions concatenated to z."""
        num_props = 3
        config = VAEConfig(
            vocab_size=45,
            embedding_dim=16,
            decoder_hidden_dim=32,
            latent_dim=16,
            num_decoder_layers=1,
            max_sequence_length=20,
            num_property_conditions=num_props,
        )
        decoder = MolecularDecoder(config)
        batch_size = 2

        z = torch.randn(batch_size, config.latent_dim)
        props = torch.randn(batch_size, num_props)

        logits = decoder(z, target=None, properties=props)

        assert logits.shape == (batch_size, config.max_sequence_length, config.vocab_size)


class TestMolecularVAE:
    """Tests for the full VAE model."""

    def test_forward_pass(self, vae_model: MolecularVAE, vae_config: VAEConfig) -> None:
        """Full forward pass should produce logits, mu, and logvar."""
        batch_size = 4
        seq_len = 30

        x = torch.randint(0, vae_config.vocab_size, (batch_size, seq_len))
        vae_model.train()
        logits, mu, logvar = vae_model(x)

        assert logits.shape == (batch_size, seq_len, vae_config.vocab_size)
        assert mu.shape == (batch_size, vae_config.latent_dim)
        assert logvar.shape == (batch_size, vae_config.latent_dim)

    def test_reparameterize_training(self, vae_model: MolecularVAE) -> None:
        """Reparameterization should add noise during training."""
        vae_model.train()
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)

        z1 = vae_model.reparameterize(mu, logvar)
        z2 = vae_model.reparameterize(mu, logvar)

        # With noise, outputs should differ (with very high probability)
        assert not torch.allclose(z1, z2)

    def test_reparameterize_eval(self, vae_model: MolecularVAE) -> None:
        """Reparameterization should return mu directly during eval."""
        vae_model.eval()
        mu = torch.randn(4, 16)
        logvar = torch.zeros(4, 16)

        z = vae_model.reparameterize(mu, logvar)

        assert torch.allclose(z, mu)

    def test_sample_shape(self, vae_model: MolecularVAE, vae_config: VAEConfig) -> None:
        """Sampling should produce logits of shape (num_samples, max_seq_len, vocab)."""
        with torch.no_grad():
            logits = vae_model.sample(num_samples=5, device="cpu")

        assert logits.shape == (
            5,
            vae_config.max_sequence_length,
            vae_config.vocab_size,
        )

    def test_interpolation(self, vae_model: MolecularVAE, vae_config: VAEConfig) -> None:
        """Latent space interpolation should produce correct number of steps."""
        z_start = torch.randn(1, vae_config.latent_dim)
        z_end = torch.randn(1, vae_config.latent_dim)

        with torch.no_grad():
            logits = vae_model.interpolate(z_start, z_end, num_steps=8)

        assert logits.shape[0] == 8
        assert logits.shape[1] == vae_config.max_sequence_length
        assert logits.shape[2] == vae_config.vocab_size

    def test_encode_decode_roundtrip(
        self,
        vae_model: MolecularVAE,
        vocab: SMILESVocabulary,
        vae_config: VAEConfig,
    ) -> None:
        """Encoding and decoding should produce valid output shapes."""
        encoded = vocab.encode_smiles(ASPIRIN_SMILES, max_length=vae_config.max_sequence_length)
        x = torch.tensor([encoded], dtype=torch.long)

        with torch.no_grad():
            mu, logvar = vae_model.encode(x)
            logits = vae_model.decode(mu)

        assert mu.shape == (1, vae_config.latent_dim)
        assert logits.shape[0] == 1
        assert logits.shape[2] == vae_config.vocab_size


class TestVAELoss:
    """Tests for the VAE loss function."""

    def test_loss_components(self) -> None:
        """Loss function should return total, reconstruction, and KL components."""
        batch_size, seq_len, vocab_size = 4, 20, 45

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mu = torch.randn(batch_size, 16)
        logvar = torch.randn(batch_size, 16)

        total, recon, kl = vae_loss(logits, targets, mu, logvar, beta=1.0)

        assert total.ndim == 0  # Scalar
        assert recon.ndim == 0
        assert kl.ndim == 0
        assert total.item() > 0
        assert recon.item() > 0
        assert kl.item() >= 0

    def test_kl_zero_for_standard_normal(self) -> None:
        """KL divergence should be near zero when q(z|x) matches N(0, I)."""
        mu = torch.zeros(100, 16)
        logvar = torch.zeros(100, 16)
        logits = torch.randn(100, 20, 45)
        targets = torch.randint(0, 45, (100, 20))

        _, _, kl = vae_loss(logits, targets, mu, logvar)

        assert kl.item() < 0.01

    def test_beta_zero_removes_kl(self) -> None:
        """With beta=0, total loss should equal reconstruction loss."""
        logits = torch.randn(4, 20, 45)
        targets = torch.randint(0, 45, (4, 20))
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)

        total, recon, kl = vae_loss(logits, targets, mu, logvar, beta=0.0)

        assert torch.allclose(total, recon, atol=1e-6)

    def test_loss_ignores_padding(self) -> None:
        """Cross-entropy should ignore padding tokens (index 0)."""
        logits = torch.randn(2, 10, 45)
        # All padding
        targets_padded = torch.zeros(2, 10, dtype=torch.long)
        # Some real tokens
        targets_real = torch.randint(1, 45, (2, 10))

        mu = torch.zeros(2, 16)
        logvar = torch.zeros(2, 16)

        _, recon_padded, _ = vae_loss(logits, targets_padded, mu, logvar, pad_idx=0)
        _, recon_real, _ = vae_loss(logits, targets_real, mu, logvar, pad_idx=0)

        # Reconstruction loss for all-padding should be 0 (all tokens ignored)
        assert recon_padded.item() == 0.0
        # Reconstruction loss for real tokens should be positive
        assert recon_real.item() > 0.0


class TestKLAnnealing:
    """Tests for KL annealing schedule."""

    def test_linear_starts_at_zero(self) -> None:
        """Linear annealing should start at 0."""
        assert kl_annealing_schedule(0, 1000, "linear") == 0.0

    def test_linear_reaches_one(self) -> None:
        """Linear annealing should reach 1.0 at total_steps."""
        assert kl_annealing_schedule(1000, 1000, "linear") == 1.0

    def test_linear_stays_at_one(self) -> None:
        """Linear annealing should stay at 1.0 after total_steps."""
        assert kl_annealing_schedule(2000, 1000, "linear") == 1.0

    def test_linear_midpoint(self) -> None:
        """Linear annealing should be 0.5 at the midpoint."""
        assert abs(kl_annealing_schedule(500, 1000, "linear") - 0.5) < 1e-6

    def test_cyclical_resets(self) -> None:
        """Cyclical annealing should reset beta at cycle boundaries."""
        beta_at_cycle_start = kl_annealing_schedule(1000, 1000, "cyclical")
        beta_at_new_cycle = kl_annealing_schedule(0, 1000, "cyclical")

        assert beta_at_new_cycle == 0.0

    def test_invalid_strategy(self) -> None:
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown annealing strategy"):
            kl_annealing_schedule(0, 1000, "cosine")
