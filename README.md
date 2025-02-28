# Murphy et al. (2025) Quantifying biological heterogeneity in nano-engineered particle-cell interaction experiments

Preprint: https://www.biorxiv.org/content/10.1101/2025.02.01.636020v1

This repository holds the key Julia code used to generate figures in the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in Jan 2025 using:

- Julia Version 1.9.3 (see https://julialang.org/downloads/ )
- Julia packages:
- This code utilises the ABC-SMC inference code from https://github.com/ap-browning/internalisation

## Data

We analyse previously published from Faria M et al. (2019). Revisiting cell–particle association in vitro: A quantitative method to compare particle performance. Journal of Controlled Release. 307, 355–367. (https://doi.org/10.1016/j.jconrel.2019.06.027) available on FigShare (https://figshare.com/articles/dataset/FCS_and_INI_files/7623671). This includes fluorescence data from Flow Cytometry Standard (FCS) files and experimental details from INI files. Raw FCS data files were converted to CSV using https://floreada.io/.

We focus on data for the following particle-cell pairs.
- 150 nm polymethacrylic acid(PMA) core-shell particles with THP-1 cells.
- 214 nm PMA-capsules with THP-1 cells.
- 633 nm PMA core-shell particles with THP-1 cells.


## Code

Code is included to reproduce results for the synthetic data studies and the experimental data studies.


| | Script        | Figures in manuscript | Short description           | 
| :---:   | :---: | :---: | :---: |
|1| @@@ | Figure @@ |  |

