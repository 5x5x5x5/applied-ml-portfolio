# PharmaFlow

**Informatica PowerCenter-inspired ETL framework for pharmaceutical data processing.**

PharmaFlow demonstrates enterprise ETL patterns used in pharma data engineering (DE4 level),
implementing Informatica-style mappings, sessions, workflows, and transformations entirely in Python.
It is paired with Oracle PL/SQL procedures and Control-M job definitions to represent a complete
production ETL environment.

## Architecture

```
                  Control-M Scheduler
                        |
                   run_etl.sh (UNIX wrapper)
                        |
              +------- PharmaFlow Engine --------+
              |                                   |
         Workflow Manager                    Session Runner
         (parallel/sequential tasks,         (executes mappings,
          decision branches,                  error handling,
          event waits, email)                 commit strategy,
              |                               recovery)
              |                                   |
         Session Task(s)                    Mapping Pipeline
              |                             Source -> Transform -> Target
              |                                   |
         Mapping Definitions              Transformations:
         - SourceDefinition                 SourceQualifier, Expression,
         - TargetDefinition                 Filter, Aggregator, Joiner,
         - MappingParameter ($$)            Lookup, Router, Sorter,
         - SessionConfig                    UpdateStrategy, Sequence,
                                            Normalizer, Rank
```

## Pipelines

| Pipeline | Source | Key Transformations | Target |
|----------|--------|---------------------|--------|
| Drug Master (SCD2) | Supplier CSV files | Standardize, validate NDC, fuzzy dedup, SCD comparison | `dim_drug` |
| Clinical Trial | XML + reference tables | Parse XML, join, aggregate enrollment, classify phase | `fact_clinical_trial`, `dim_trial` |
| FAERS Adverse Events | FDA quarterly files | Cross-reference, MedDRA standardize, PRR/ROR signals | `fact_adverse_event`, `safety_signal` |

## Informatica Concepts Mapped

| Informatica PowerCenter | PharmaFlow Equivalent |
|------------------------|----------------------|
| Mapping | `Mapping` class with sources, transforms, targets |
| Session | `Session` class with runtime execution |
| Workflow | `Workflow` class with task orchestration |
| Source Qualifier | `SourceQualifier` transformation |
| Expression | `Expression` transformation |
| Filter | `Filter` transformation |
| Aggregator | `Aggregator` transformation (sorted/unsorted) |
| Joiner | `Joiner` transformation (inner/outer/master-only) |
| Lookup | `Lookup` transformation (connected/unconnected) |
| Router | `Router` transformation (multi-group) |
| Sorter | `Sorter` transformation |
| Update Strategy | `UpdateStrategy` (DD_INSERT/UPDATE/DELETE/REJECT) |
| Sequence Generator | `SequenceGenerator` transformation |
| Normalizer | `Normalizer` transformation |
| Rank | `Rank` transformation |
| $$Parameters | `MappingParameter` with runtime resolution |
| Session Config | `SessionConfig` (commit interval, error threshold) |
| Workflow Manager | `Workflow` with `TaskLink` conditions |

## Database Layer (Oracle PL/SQL)

- **DDL**: `sql/ddl/operational_data_store.sql` -- staging, ODS, dimension, fact, error log, metadata tables
- **SCD Type 2**: `sql/procedures/scd_type2_merge.sql` -- hash comparison, close/insert pattern
- **Data Quality**: `sql/procedures/data_quality_checks.sql` -- referential integrity, null checks, range validation, cross-table consistency

## Scheduling (Control-M)

- **Daily Drug Master**: `controlm/jobs/daily_drug_master.xml` -- file check, validate, ETL, SCD merge, DQ, report, archive
- **Weekly FAERS**: `controlm/jobs/weekly_faers_load.xml` -- parallel staging loads, cross-reference, signal detection

## Setup

```bash
cd 16-pharma-etl-informatica
uv sync
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
16-pharma-etl-informatica/
  pyproject.toml
  src/pharma_flow/
    __init__.py
    framework/
      mapping.py            # Mapping, Source, Target, Parameters
      transformations.py    # All Informatica-style transformations
      session.py            # Session runner with error handling
      workflow.py           # Workflow orchestration
    pipelines/
      drug_master_etl.py    # Drug master SCD2 pipeline
      clinical_trial_etl.py # Clinical trial data pipeline
      adverse_event_etl.py  # FAERS adverse event pipeline
  sql/
    ddl/operational_data_store.sql
    procedures/scd_type2_merge.sql
    procedures/data_quality_checks.sql
  controlm/jobs/
    daily_drug_master.xml
    weekly_faers_load.xml
  scripts/unix/
    run_etl.sh
  tests/
    conftest.py
    test_transformations.py
    test_session.py
    test_pipelines.py
```
