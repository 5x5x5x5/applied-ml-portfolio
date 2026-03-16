"""
PharmaDataVault - Pharmaceutical Data Warehouse using Data Vault 2.0.

This package implements a complete pharmaceutical data warehouse solution
for managing clinical trial data, drug manufacturing records, and regulatory
submissions using the Data Vault 2.0 methodology.

Architecture Overview:
    - Raw Vault: Hubs, Links, and Satellites storing raw source data
    - Business Vault: PIT tables and Bridge tables for query optimization
    - Data Marts: Star schema views for clinical analytics and reporting
    - ETL: PL/SQL procedures orchestrated via Python and Control-M

Data Vault 2.0 Conventions:
    - Hash keys (MD5) used for hub surrogate keys
    - Hashdiff columns for satellite change detection
    - Record source and load date on all raw vault tables
    - Same-as links for master data management
    - Effectivity satellites for temporal relationships
"""

__version__ = "1.0.0"
__author__ = "PharmaDataVault Team"

from pharma_vault._config import VaultConfig

__all__ = ["VaultConfig", "__version__"]
