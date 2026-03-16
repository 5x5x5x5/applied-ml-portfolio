"""Shared test fixtures for ProteinExplorer tests."""

from __future__ import annotations

import pytest

# Real protein sequences for testing


@pytest.fixture
def insulin_b_chain() -> str:
    """Human insulin B chain (30 residues)."""
    return "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


@pytest.fixture
def hemoglobin_alpha() -> str:
    """Human hemoglobin alpha subunit (141 residues, UniProt P69905)."""
    return (
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        "GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKL"
        "LSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    )


@pytest.fixture
def hemoglobin_beta() -> str:
    """Human hemoglobin beta subunit (147 residues, UniProt P68871)."""
    return (
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"
        "PDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDP"
        "ENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    )


@pytest.fixture
def lysozyme() -> str:
    """Hen egg-white lysozyme (129 residues, UniProt P00698).
    Contains 4 disulfide bonds: C6-C127, C30-C115, C64-C80, C76-C94.
    """
    return (
        "KVFGRCELAAALKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGST"
        "DYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDC"
        "PGMDQRFSGKHIAATIKGNLPGWAWLNVSE"
    )


@pytest.fixture
def short_peptide() -> str:
    """Short test peptide."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def polyalanine() -> str:
    """Polyalanine - strong helix former."""
    return "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def disordered_sequence() -> str:
    """Sequence with characteristics of intrinsically disordered regions.
    Enriched in E, K, P, S, Q - disorder-promoting residues.
    """
    return "EPSQKEPESQKPESQKEPESQKPESQKEPESQKPESQKEPESQKPESQKEPESQKEPESQKPESQKEPESQKPESQKE"
