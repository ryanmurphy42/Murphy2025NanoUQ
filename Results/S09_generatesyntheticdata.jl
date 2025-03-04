#=
    S09 - Generate synthetic data based on R1
=#

# 1 - Load packages and modules
# 2 - Setup problem
# 3 - Synthetic parameters
# 4 - Sample cell fluorescence
# 5 - Compute particles per cell, P(t), from mathematical model
# 6 - Sample particle fluorescence (one value for each particle)
# 7 - Compute Total fluorescence = Cell fluorescence + P(t)*Particle fluorescence
# 8 - Save as csv files

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using CSV, DataFrames, DataFramesMeta
using Distributions
using JLD2
using Plots
using StatsBase
using StatsPlots
using Random
using LaTeXStrings

##############################################################
## 2 - Setup problem
##############################################################

include("R1_Setup.jl")

##############################################################
## 3 - Synthetic parameters
##############################################################

μr=3.86125e-7*100;
σr=4.56962e-7*100;
μK=10.0;
σK=2.0;
θsyn = [μr,σr,μK,σK];

samples_per_timepoint = 20_000;

##############################################################
## 4 - Sample cell fluorescence
##############################################################

# measurements at t0
Random.seed!(1)
F_cell_t0 = pmt_correctionfactor(voltage_cellonly,voltage_particleonly)*rand(data_cellonly[:,"638 Red(Peak)"],samples_per_timepoint);

# measurements at t > 0
Random.seed!(2)
F_cell = pmt_correctionfactor(voltage_cellonly,voltage_particleonly)*[rand(data_cellonly[:,"638 Red(Peak)"],samples_per_timepoint) for i in T];

##############################################################
## 5 - Compute particles per cell, P(t), from mathematical model
##############################################################

Random.seed!(3)
Psyndeter = [simulate_model_noiseless_longt(θsyn,θfixed,[t]*3600,param_dist;N=samples_per_timepoint)[1] for t in T];

# plot boxplots P(t) from mathematical model
fig_Psyndeter = plot(layout=grid(1,1),xlab=L"t \ \mathrm{[hours]}",ylab=L"P(t)",legend=false)
bar_width_val_box =1;
[boxplot!(fig_Psyndeter,[T[i]], Psyndeter[i],bar_width=bar_width_val_box,color=:white) for i=1:length(T)]
display(fig_Psyndeter)

# plot median P(t) 
fig_Psyndeter_median = plot(layout=grid(1,1),xlab=L"t \ \mathrm{[hours]}",ylab=L"P(t)",legend=false)
scatter!(fig_Psyndeter_median,T, [median(Psyndeter[i]) for i=1:length(T)]) 
display(fig_Psyndeter_median)

##############################################################
## 6 - Sample particle fluorescence (one value for each particle)
##############################################################

# measurements at t0
Random.seed!(1)
F_p_t0 = rand(data_particleonly[:,"638 Red(Peak)"],samples_per_timepoint);

# measurements at t > 0
# for each cell, generate ceil(P(t)) samples of particle-only fluorescence
Random.seed!(2)
F_p = [[rand(data_particleonly[:,"638 Red(Peak)"],Int(ceil(Psyndeter[i][j]))) for j=1:length(Psyndeter[i])] for i=1:length(T)];

##############################################################
## 7 - Compute Total fluorescence = Cell fluorescence + P(t)*Particle fluorescence
##############################################################

Psyn = [pmt_correctionfactor(voltage_particleonly,voltage_data[i])*[F_cell[i][j] + sum(F_p[i][j][1:Int(floor(Psyndeter[i][j]))]) + (Psyndeter[i][j]-floor(Psyndeter[i][j]))*F_p[i][j][Int(ceil(Psyndeter[i][j]))] for j=1:length(Psyndeter[i])] for i=1:length(T)];

# plot total fluorescence
fig_Psyn = plot(layout=grid(1,1),xlab=L"t \ \mathrm{[hours]}",ylab=L"P(t)",legend=false)
bar_width_val_box =1;
[boxplot!(fig_Psyn,[T[i]], Psyn[i],bar_width=bar_width_val_box,color=:white) for i=1:length(T)]
display(fig_Psyn)

##############################################################
## 8 - Save as csv files
##############################################################

filepath_save = pwd() * "/Data/Synthetic/"
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

#### Cell-only
# ID | F_cell | 638 Red(Peak) (where 638 Red(Peak) is the same as F_cell)
data_cellonly_syn = DataFrame(ID=1:samples_per_timepoint, F_cell = F_cell_t0, c= F_cell_t0);
rename!(data_cellonly_syn,:c => "638 Red(Peak)")
CSV.write(filepath_save * "data_cellonly_syn_S09.csv", data_cellonly_syn) # Export to csv

#### Particle-only
# ID | F_p | 638 Red(Peak) (where 638 Red(Peak) is the same as F_p)
data_particleonly_syn = DataFrame(ID=1:samples_per_timepoint, F_p = F_p_t0, c= F_p_t0);
rename!(data_particleonly_syn,:c => "638 Red(Peak)")
CSV.write(filepath_save * "data_particleonly_syn_S09.csv", data_particleonly_syn) # Export to csv

#### Cell-particle associations
# ID | F_cell | F_p | Psyndeter | Psyn | 638 Red(Peak) (where 638 Red(Peak) is the same as Psyn)
for i=1:length(T)
    data_syn = DataFrame(ID=1:samples_per_timepoint, F_cell = F_cell[i], F_p = F_p[i], Psyndeter = Psyndeter[i], Psyn = Psyn[i], c =Psyn[i] );
    rename!(data_syn,:c => "638 Red(Peak)")
    CSV.write(filepath_save * "data_syn_S09_" * string(i) * ".csv", data_syn) # Export to csv
end
