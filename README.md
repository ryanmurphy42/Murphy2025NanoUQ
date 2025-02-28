# Murphy et al. (2025) Quantifying biological heterogeneity in nano-engineered particle-cell interaction experiments

Preprint: https://www.biorxiv.org/content/10.1101/2025.02.01.636020v1

This repository holds the key Julia code used to generate figures in the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in Jan 2025 using:

- Julia Version 1.9.3 (see https://julialang.org/downloads/ )
- Julia packages:
- This code utilises the ABC-SMC inference code from https://github.com/ap-browning/internalisation

## Data

### Experimental data 
We analyse previously published from Faria M et al. (2019). Revisiting cell–particle association in vitro: A quantitative method to compare particle performance. Journal of Controlled Release. 307, 355–367. (https://doi.org/10.1016/j.jconrel.2019.06.027) available on FigShare (https://figshare.com/articles/dataset/FCS_and_INI_files/7623671). This includes fluorescence data from Flow Cytometry Standard (FCS) files and experimental details from INI files. Raw FCS data files were converted to CSV using https://floreada.io/.

We focus on data for the following particle-cell pairs.
- 150 nm polymethacrylic acid(PMA) core-shell particles with THP-1 cells.
- 214 nm PMA-capsules with THP-1 cells.
- 633 nm PMA core-shell particles with THP-1 cells.


### Synthetic data.
We generate synthetic data using the following code:
| | Filename      | Description           | 
| :---:    | :---: | :---: |
|1| S01_genrate_syntheticdata | Based on data in R1 |
|2| S05_genrate_syntheticdata  | Low r (relative to K) |
|3| S07_genrate_syntheticdata  | Intermediate r (relative to K) |
|4| S09_genrate_syntheticdata  | High r (relative to K) |

## Code

Code is included to reproduce results for the synthetic data studies and the experimental data studies.

| | Code_ID       | Synthetic/Experimental | Description           | 
| :---:   | :---: | :---: | :---: |
|1| R1 | Experimental | 150 nm PMA core-shell particles with THP-1 cells |
|2| R5 | Experimental | 214 nm PMA-capsules with THP-1 cells |
|3| R6 | Experimental | 633 nm PMA core-shell particles with THP-1 cells |
|4| S1 | Synthetic | Based on data in R1 |
|5| S5 | Synthetic | Low r (relative to K) |
|6| S7 | Synthetic | Intermediate r (relative to K) |
|7| S9 | Synthetic | High r (relative to K) |


Each of these Code_ID's are associated with the following files (where xx represents the Code_ID from the above table).

| | Filename       | Description | ABC distance metric | ABC-SMC algorithm | 
| :---:   | :---: | :---: | :---: | :---: |
|1| xx_lesq | Method of least squares using homogeneous model  |  NA | NA |
|2| xxAD_Setup | Set up file including model, priors, reference to data  |  Anderson-Darling | NA |
|3| xxAD_Inference | ABC-SMC algorithm  | Anderson-Darling  | Target acceptance probability |
|4| xxAD_ABCE_Inference |  ABC-SMC algorithm |  Anderson-Darling | Target acceptance threshold |
|5| xxCV_Setup | Set up file including model, priors, reference to data  |  Cramer von Mises | NA |
|6| xxCV_Inference | ABC-SMC algorithm  | Cramer von Mises  | Target acceptance probability |
|7| xxCV_ABCE_Inference |  ABC-SMC algorithm |  Cramer von Mises | Target acceptance threshold |
|8| xxKS_Setup | Set up file including model, priors, reference to data  |  Kolmogorov-Smirnov | NA |
|9| xxKS_Inference | ABC-SMC algorithm  | Kolmogorov-Smirnov  | Target acceptance probability |
|10| xxKS_ABCE_Inference |  ABC-SMC algorithm |  Kolmogorov-Smirnov | Target acceptance threshold |





