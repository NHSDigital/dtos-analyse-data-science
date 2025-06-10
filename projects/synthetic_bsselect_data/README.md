# Synthetic BS Select Data Generation + Metric Calculation

A Python toolkit for generating, processing, and analyzing synthetic data for the Breast Screening Select (BS Select) project.

## Overview

This toolkit provides a reproducible way to create and analyze synthetic breast screening data, supporting development, testing, and metric calculation without using real patient data.

Key features:

- Generation of synthetic subject and episode data with realistic structure
- Mapping tables for organizations, postcodes, and codes
- Output templates for consistent data structure
- Metric calculation notebooks for screening analysis (e.g., uptake, round length)
- Modular, extensible codebase

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - `pandas`
  - `numpy`
  - `jupyter`
- For data generation, the following mapping tables in data_generation/mapping_tables:
  - CeaseReason.csv
  - EndCode.csv
  - EpisodeType.csv
  - FinalActionCode.csv
  - Language.csv
  - ReasonClosedCode.csv
  - RemovalReason.csv
  - bso_gp_mapping.csv
  - bso_organisations.csv
  - higher_risk_genes.csv
  - higher_risk_referral_reasons.csv

### Setup

1. Install the requirements:

```bash
poetry install
```

## Usage

### Generate Synthetic Data

Run the data generation notebook:

- Execute all cells to generate synthetic episodes and subjects data.
- Output files will be saved in `output_data/` with a datestamp.

### Calculate Metrics

Open the metric_test notebook and run the analysis required:

## Project Structure

- `data_generation/` — Notebook for synthetic data generation
- `metric_calculations/` — Notebooks and scripts for metric calculation and analysis
- `output_data/` — Generated synthetic datasets
- `output_templates/` — CSV templates for output structure
- `mapping_tables/` — Lookup tables for data generation

## Notes

- All data generated is synthetic and does **not** contain any real patient information.
- Mapping tables and templates are based on real data structure.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
