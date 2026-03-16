-- CloudGenomics - Database initialization
-- PostgreSQL schema for variant classification results

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Classification results table
CREATE TABLE IF NOT EXISTS classification_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    variant_id VARCHAR(64) NOT NULL UNIQUE,
    chrom VARCHAR(10) NOT NULL,
    pos BIGINT NOT NULL,
    ref_allele VARCHAR(1000) NOT NULL,
    alt_allele VARCHAR(1000) NOT NULL,
    classification VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    class_probabilities JSONB NOT NULL DEFAULT '{}',
    explanation JSONB NOT NULL DEFAULT '[]',
    features JSONB NOT NULL DEFAULT '{}',
    model_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Processing jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    vcf_s3_key VARCHAR(500),
    total_variants INT DEFAULT 0,
    passed_variants INT DEFAULT 0,
    classified_variants INT DEFAULT 0,
    processing_time_seconds DOUBLE PRECISION,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(128) NOT NULL,
    action VARCHAR(50) NOT NULL,
    principal VARCHAR(256) NOT NULL,
    resource VARCHAR(500) NOT NULL,
    data_classification VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    details JSONB DEFAULT '{}',
    source_ip INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_results_chrom_pos ON classification_results(chrom, pos);
CREATE INDEX IF NOT EXISTS idx_results_classification ON classification_results(classification);
CREATE INDEX IF NOT EXISTS idx_results_created ON classification_results(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON processing_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_principal ON audit_log(principal, created_at);
