# Dataset Information

## OASIS-1 Cross-Sectional MRI Dataset

### Download Instructions

1. Visit [OASIS Brains](https://www.oasis-brains.org/)
2. Register for an account
3. Download OASIS-1 Cross-Sectional dataset
4. You will receive 12 disc folders (`disc1` to `disc12`)

### Dataset Structure

After downloading, organize as follows:

HD_Diagnosis_Project/data/raw/
├── oasis_cross-sectional.csv
├── oasis_cross-sectional_disc1/
│ └── disc1/
│ ├── OAS1_0001_MR1/
│ │ └── RAW/
│ │ ├── OAS1_0001_MR1_mpr-1_anon.img
│ │ ├── OAS1_0001_MR1_mpr-1_anon.hdr
│ │ ├── ... (3-4 scans per subject)
│ ├── OAS1_0002_MR1/
│ └── ...
├── oasis_cross-sectional_disc2/
├── ... (disc3 to disc12)

### Dataset Statistics

- **Total Subjects**: 436
- **Age Range**: 18-96 years
- **Gender**: Male and Female
- **Control (CDR=0.0)**: 336 subjects (77%)
- **Dementia (CDR>0.0)**: 100 subjects (23%)
  - CDR 0.5: 70 subjects
  - CDR 1.0: 28 subjects
  - CDR 2.0: 2 subjects

### Clinical Variables

The `oasis_cross-sectional.csv` contains:
- **ID**: Subject identifier
- **M/F**: Gender
- **Hand**: Handedness
- **Age**: Age in years
- **Educ**: Years of education
- **SES**: Socioeconomic status (1-5)
- **MMSE**: Mini-Mental State Examination score
- **CDR**: Clinical Dementia Rating (0, 0.5, 1, 2)
- **eTIV**: Estimated total intracranial volume
- **nWBV**: Normalized whole-brain volume
- **ASF**: Atlas scaling factor

### Label Definition

For binary classification:
- **Label 0 (Control)**: CDR = 0.0
- **Label 1 (Dementia)**: CDR > 0.0 (includes 0.5, 1.0, 2.0)

### Citation

If you use this dataset, please cite:

Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007).
Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged,
nondemented, and demented older adults. Journal of cognitive neuroscience, 19(9), 1498-1507.

### License

The OASIS dataset is available under its own license. Please review the terms at:
https://www.oasis-brains.org/

### Important Notes

⚠️ **DO NOT upload the OASIS dataset to GitHub**  
⚠️ The dataset is ~150GB and is proprietary  
⚠️ Users must download it themselves from the official source  
