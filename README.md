# Murphy et al. (2025) Quantifying biological heterogeneity in nano-engineered particle-cell interaction experiments

Preprint: https://www.biorxiv.org/content/10.1101/2025.02.01.636020v1

This repository holds the key Julia code used to generate figures in the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in Jan 2025 using:

- Julia Version 1.9.3 (see https://julialang.org/downloads/ )
- Julia packages: Base.Threads, Combinatorics, CSV, DataFrames, DataFramesMeta, DifferentialEquations, Distances, Distributions, HypothesisTests, IniFile, Inference, Interpolations, JLD2, KernelDensity, LaTeXStrings, LinearAlgebra, Model, NLopt, Plots, Printf, Random, Statistics, StatsBase, StatsFuns, StatsPlots, Sundials, Threads, Trapz
- This code utilises the ABC-SMC inference code from https://github.com/ap-browning/internalisation

## Data

### Experimental data 
We analyse previously published from Faria M et al. (2019). Revisiting cell–particle association in vitro: A quantitative method to compare particle performance. Journal of Controlled Release. 307, 355–367. (https://doi.org/10.1016/j.jconrel.2019.06.027) available on FigShare (https://figshare.com/articles/dataset/FCS_and_INI_files/7623671). This includes fluorescence data from Flow Cytometry Standard (FCS) files and experimental details from INI files. Raw FCS data files were converted to CSV using https://floreada.io/.

We focus on data for the following particle-cell pairs:
- 150 nm polymethacrylic acid(PMA) core-shell particles with THP-1 cells.
- 214 nm PMA-capsules with THP-1 cells.
- 633 nm PMA core-shell particles with THP-1 cells.


### Synthetic data.
We generate synthetic data using the following code:
| | Filename      | Description           | 
| :---:    | :---: | :---: |
|1| S01_generate_syntheticdata | Based on data in R1 |
|2| S07_generate_syntheticdata  | Low r (relative to K) |
|3| S05_generate_syntheticdata  | Intermediate r (relative to K) |
|4| S09_generate_syntheticdata  | High r (relative to K) |

### Synthetic data (exploring role of control data)
| | Filename      | Description           | 
| :---:    | :---: | :---: |
|5| SP05_generate_syntheticdata | Low r (relative to K); heterogeneous model; synthetic control data. |
|6| SP07_generate_syntheticdata  | Intermediate r (relative to K); heterogeneous model; synthetic control data. |
|7| SP09_generate_syntheticdata  | High r (relative to K); heterogeneous model;synthetic control data. |
|8| SQ05_generate_syntheticdata  | Low r (relative to K); homogeneous model; synthetic control data. |
|9| SQ07_generate_syntheticdata  | Intermediate r (relative to K); homogeneous model; synthetic control data. |
|10| SQ09_generate_syntheticdata  | High r (relative to K); homogeneous model; synthetic control data. |
|11| SR05_generate_syntheticdata  | Low r (relative to K); homogeneous model; experimental control data. |
|12| SR07_generate_syntheticdata  | Intermediate r (relative to K);  homogeneous model; experimental control data. |
|13| SR09_generate_syntheticdata  | High r (relative to K);  homogeneous model; experimental control data. |


## Code

### Parameter inference, identifiability analysis, and prediction

Code is included to reproduce results for the synthetic data studies and the experimental data studies.

| | Code_ID       | Synthetic/Experimental | Description           | Model | Control data |
| :---:   | :---: | :---: | :---: |  :---: |  :---: |
|1| R1 | Experimental | 150 nm PMA core-shell particles with THP-1 cells | Heterogeneous | Experimental |
|2| R5 | Experimental | 214 nm PMA-capsules with THP-1 cells | Heterogeneous | Experimental |
|3| R6 | Experimental | 633 nm PMA core-shell particles with THP-1 cells | Heterogeneous | Experimental |
|4| S1 | Synthetic | Based on data in R1 | Heterogeneous | Experimental |
|5| S7 | Synthetic | Low r (relative to K) | Heterogeneous | Experimental |
|6| S5 | Synthetic | Intermediate r (relative to K) | Heterogeneous | Experimental |
|7| S9 | Synthetic | High r (relative to K) | Heterogeneous | Experimental |
|8| SP5** | Synthetic | Low r (relative to K) | Heterogeneous | Synthetic |
|9| SP7** | Synthetic | Intermediate r (relative to K) | Heterogeneous | Synthetic |
|10| SP9** | Synthetic | High r (relative to K) | Heterogeneous | Synthetic |
|11| SQ5** | Synthetic | Low r (relative to K) | Homogeneous | Synthetic |
|12| SQ7** | Synthetic | Intermediate r (relative to K) | Homogeneous | Synthetic |
|13| SQ9** | Synthetic | High r (relative to K) | Homogeneous | Synthetic |
|14| SR5** | Synthetic | Low r (relative to K) | Homogeneous | Experimental |
|15| SR7** | Synthetic | Intermediate r (relative to K) | Homogeneous | Experimental |
|16| SR9** | Synthetic | High r (relative to K) | Homogeneous | Experimental |

** These data are only used for Supp. Sec. 2.3.

Each of these Code_ID's are associated with the following files (where xx represents the Code_ID from the above table).

| | Filename       | Description | ABC distance metric | ABC-SMC algorithm | 
| :---:   | :---: | :---: | :---: | :---: |
|1| xx_lesq | Method of least squares using homogeneous model  |  NA | NA |
|2| xxAD_Setup | Set up file including model, priors, reference to data  |  Anderson-Darling | NA |
|3| xxAD_Inference | ABC-SMC algorithm  | Anderson-Darling  | Target acceptance probability |
|4| xxADE_Inference* |  ABC-SMC algorithm |  Anderson-Darling | Target acceptance threshold  (chosen for comparison across data sets) |
|5| xxAD_ABCE_Inference |  ABC-SMC algorithm |  Anderson-Darling | Target acceptance threshold (chosen for comparison across ABC distances) |
|6| xxCV_Setup | Set up file including model, priors, reference to data  |  Cramer von Mises | NA |
|7| xxCV_Inference | ABC-SMC algorithm  | Cramer von Mises  | Target acceptance probability |
|8| xxCV_ABCE_Inference |  ABC-SMC algorithm |  Cramer von Mises | Target acceptance threshold (chosen for comparison across ABC distances) |
|9| xxKS_Setup | Set up file including model, priors, reference to data  |  Kolmogorov-Smirnov | NA |
|10| xxKS_Inference | ABC-SMC algorithm  | Kolmogorov-Smirnov  | Target acceptance probability |
|11| xxKS_ABCE_Inference |  ABC-SMC algorithm |  Kolmogorov-Smirnov | Target acceptance threshold (chosen for comparison across ABC distances) |

*For synthetic data studies S1, S5, S7, S9 only.
*** SPx, SQx, SRx code_id's only include scripts for the Anderson-Darling ABC distance metric.

Code for the ABC-SMC algorithm (adapted from https://github.com/ap-browning/internalisation), together with the homogeneous mathematical model and heterogeneous mathematical model are included in the following Modules:
| | Module       | Filename | Description | 
| :---:   | :---: | :---: | :---: |
|1| Model | Model | Header file for Model module   | 
|2| Model |  deterministic |  Homogeneous mathematical model and approximate solution to heterogeneous model  |
|3| Model |  statistical |  Heterogeneous model with noise |
|4| Inference |  Inference | Header file for Inference module  |
|5| Inference |  abc |  ABC-SMC algorithms |
|6| Inference |  particles | Implements the type 'Particle' used in ABC-SMC  |


Code for plotting
| | Filename  |  Description | 
| :---:   | :---: | :---: |
|1| plots_ABCSMC | Plots for posterior densities, inferred distributions, and predictions |    
|2| plots_comparing_ABCdistances | Plots for comparing results obtained with different ABC distance functions  |  


### Experimental design

Code is included to reproduce results for the experimental design synthetic data studies. This follows the ABCDE algorithm (see manuscript).

| | Code_ID       | Synthetic/Experimental | Description           | 
| :---:   | :---: | :---: | :---: |
|1| lowr | Synthetic | Low r (relative to K) |
|2| intr | Synthetic | Intermediate r (relative to K) |
|3| highr | Synthetic | High r (relative to K) |

Note that lowr, intr, and highr use different parameter values and are distinct to S5, S7, and S9.

Each of these Code_ID's are associated with the following files (where xx represents the Code_ID from the above table).

| | Filename       | Description | ABC distance metric | ABC algorithm | 
| :---:   | :---: | :---: | :---: | :---: |
|1| OPTD_xx_Setup | Set up file including model, priors, reference to data  |  NA | NA |
|2| OPTD_xx_1A | Pre-simulate synthetic data sets  |  NA | NA |
|3| OPTD_xx_1B | Perform ABC rejection  |  Anderson-Darling | ABC rejection |

Code for comparing designs and plotting
| | Filename  |  Description | 
| :---:   | :---: | :---: |
|1| OPTD_1C_computations | Compute ABC error threshold and discrepancy using ABC posteriors with at least 200 samples |    
|2| OPTD_1D_plots | Compute overall rank and plots comparing experimental designs |    



### Verification of the approximate solution to the heterogeneous model

| | Filename  |  Description | 
| :---:   | :---: | :---: |
|1| A1_verifyapproxhetsoln_chosen | Verification for chosen parameter values |   
|2| A1_verifyapproxhetsoln_abc1 | Compute error for verification using ABC output |   
|3| A1_verifyapproxhetsoln_abc2 | Plotting error for verification using ABC output |   


