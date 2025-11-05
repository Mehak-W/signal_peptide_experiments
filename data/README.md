# Data Directory

This directory contains pre-computed Protein Language Model (PLM) embeddings for signal peptide sequences.

## Files

### PLM Embeddings (Parquet Format)

The following files contain pre-computed embeddings from three different PLM models:

**ESM-2 650M:**
- `trainAA_esm2-650M.parquet` - Training set embeddings (1280-dimensional)
- `testAA_esm2-650M.parquet` - Test set embeddings (1280-dimensional)

**ESM-2 3B:**
- `trainAA_esm2-3B.parquet` - Training set embeddings (2560-dimensional)
- `testAA_esm2-3B.parquet` - Test set embeddings (2560-dimensional)

**Ginkgo AA0 650M:**
- `trainAA_ginkgo-AA0-650M.parquet` - Training set embeddings (1280-dimensional)
- `testAA_ginkgo-AA0-650M.parquet` - Test set embeddings (1280-dimensional)

### Original Dataset

**`sb2c00328_si_011.xlsx`**

Supplementary Table S2 from [Grasso et al. (2023)](https://dx.doi.org/10.1021/acssynbio.2c00328).

Columns:
- `SP_nt`: Nucleotide sequence of signal peptide
- `SP_aa`: Amino acid sequence of signal peptide
- `WA`: Weighted average efficiency score (1=best, 10=worst)
- `Set`: Train/Test split indicator

Note: Only 83% of sequences have WA values (successful retrieval after screening).

## Data Format

### Parquet Files Structure

Each parquet file contains three columns:

1. **sequence** (string): Amino acid sequence of the signal peptide
2. **embedding** (array): Dense vector representation from the PLM
   - ESM-2 650M / Ginkgo AA0: 1280 dimensions
   - ESM-2 3B: 2560 dimensions
3. **WA** (float): Weighted average efficiency score
   - Range: 1 (best) to 10 (worst)
   - Based on screening results from Grasso et al.

### Dataset Statistics

- **Training set**: ~3,095 sequences
- **Test set**: ~1,326 sequences
- **Total**: 4,421 informative signal peptides
- **Feature dimensions**:
  - ESM-2 650M: 1280
  - ESM-2 3B: 2560
  - Ginkgo AA0: 1280

## Data Generation

The PLM embeddings were generated using:
- **ESM-2 models**: Facebook/Meta AI's ESM-2 (650M and 3B parameter versions)
- **Ginkgo AA0**: Ginkgo BioWorks' protein language model
- **API**: Ginkgo BioWorks API (<$0.10 USD total cost as of 03/2025)
- **Generation script**: See `signal_peptides-main/scripts/06_Retrieve_Peptide_Embeddings.wls`

## Loading Data

Use the provided utility functions:

```python
from src.data_utils import load_plm_embeddings

# Load a specific embedding
X_train, X_test, y_train, y_test = load_plm_embeddings('esm2-650M')

# Available models: 'esm2-650M', 'esm2-3B', 'ginkgo-AA0-650M'
```

## Cross-Dataset Files

### Xue et al. (2021)

**Source**: Xue, S., et al. (2021). "Improvement of the secretion of multiple heterologous proteins in Saccharomyces cerevisiae by modulation of the α-factor prepro signal sequence." *Biotechnology Letters*, 43, 1471-1483.

**File**: `xue_esm_embeddings.parquet`

**Description**: 322 signal peptide variants tested in *S. cerevisiae* (yeast) for heterologous protein secretion.

**Columns**:
- `sequence`: Signal peptide amino acid sequence
- `embedding`: ESM-2 650M embeddings (1280-dim)
- `WA`: Protein titer (U/L), range 0-10,437 U/L

**Available Embeddings**: ESM-2 650M only
**Missing**: ESM-2 3B, Ginkgo AA0

---

### Zhang et al. (2022)

**Source**: Zhang, B., et al. (2022). "Combining promoter and signal peptide engineering for high-level protein secretion in Saccharomyces cerevisiae." *ACS Synthetic Biology*, 11(5), 1989-1999.

**Files**:
- `zhang_p43_esm_embeddings.parquet` - P43 promoter condition
- `zhang_pglvm_esm_embeddings.parquet` - pGLVM promoter condition

**Description**: 114 signal peptides tested with two different promoters in yeast. Same signal peptides, different promoter contexts.

**Columns**:
- `sequence`: Signal peptide amino acid sequence
- `embedding`: ESM-2 650M embeddings (1280-dim)
- `WA`: Protein titer (U/L) for the specific promoter, range 0-327 U/L

**Available Embeddings**: ESM-2 650M only
**Missing**: ESM-2 3B, Ginkgo AA0

---

### Wu et al. (2020)

**Source**: Wu, M., et al. (2020). "Signal peptide library design for enhanced protein secretion in Pichia pastoris." *Microbial Cell Factories*, 19(1), 1-15.

**File**: `wu_esm_embeddings.parquet`

**Description**: 81 signal peptides tested for binary secretion success in *P. pastoris* (yeast).

**Columns**:
- `sequence`: Signal peptide amino acid sequence
- `embedding`: ESM-2 650M embeddings (1280-dim)
- `WA`: Binary label (0 = no secretion, 1 = successful secretion)

**Available Embeddings**: ESM-2 650M only
**Missing**: ESM-2 3B, Ginkgo AA0

**Note**: This is a classification task, unlike the regression tasks in other datasets.

---

## Known Limitations

1. **Cross-dataset embeddings incomplete**: Wu, Xue, and Zhang datasets only have ESM-2 650M embeddings. ESM-2 3B and Ginkgo AA0 embeddings would need to be generated for complete model comparison across datasets.

2. **Scale mismatch**: Different datasets measure efficiency on different scales (WA 1-10 for Grasso, protein titers in U/L for Xue/Zhang, binary for Wu). Use rank-based metrics (Spearman correlation) for cross-dataset evaluation.

3. **Organism differences**: Grasso uses bacteria (*B. subtilis*), while cross-datasets use yeast (*S. cerevisiae*, *P. pastoris*). Cross-species transfer is expected to be imperfect.

4. **Bin count data**: The Grasso dataset includes 10 columns with bin probability distributions (`Perc_unambiguousReads_BIN01_bin` through `BIN10_bin`) in the xlsx file. These are not included in the parquet files to keep sizes manageable but could be added if needed for model training.

---

## References

1. Grasso, S., et al. (2023). "Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation." *ACS Synthetic Biology*. DOI: 10.1021/acssynbio.2c00328

2. Xue, S., et al. (2021). "Improvement of the secretion of multiple heterologous proteins in Saccharomyces cerevisiae by modulation of the α-factor prepro signal sequence." *Biotechnology Letters*, 43, 1471-1483.

3. Zhang, B., et al. (2022). "Combining promoter and signal peptide engineering for high-level protein secretion in Saccharomyces cerevisiae." *ACS Synthetic Biology*, 11(5), 1989-1999.

4. Wu, M., et al. (2020). "Signal peptide library design for enhanced protein secretion in Pichia pastoris." *Microbial Cell Factories*, 19(1), 1-15.

5. Lin, Z., et al. (2022). "Language models of protein sequences at the scale of evolution enable accurate structure prediction." *bioRxiv*. (ESM-2 paper)

## License

The datasets are from published research and reproduced here for research purposes. Please cite the original papers if you use this data.
