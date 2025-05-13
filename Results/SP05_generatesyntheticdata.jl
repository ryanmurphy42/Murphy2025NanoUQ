#=
    SP05 - Generate synthetic data with intermediate r (relative to K) - KNOWN SYNTHETIC CONTROL DATA
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

include("R1AD_Setup.jl")

##############################################################
## 3 - Synthetic parameters
##############################################################

μr=3.86125e-7;
σr=4.56962e-7;
μK=10.0;
σK=2.0;
θsyn = [μr,σr,μK,σK];

samples_per_timepoint = 20_000;

##############################################################
## 4A - Generate synthetic cell-only control data
##############################################################

# Compute number of samples in the experimental cell-only control data
nsamples_cell_only = length(data_cellonly[:,"638 Red(Peak)"]);

# Sample a known distribution
Random.seed!(101)
μC = mean(data_cellonly[:,"638 Red(Peak)"]);
σC = std(data_cellonly[:,"638 Red(Peak)"]);
cell_distribution_known = Normal(μC,σC);
cell_distribution_known_samples_600v = rand(cell_distribution_known,nsamples_cell_only);

mean(cell_distribution_known_samples_600v)
std(cell_distribution_known_samples_600v)
skewness(cell_distribution_known_samples_600v)

# Convert from 500volt measurements to 600volt measurements (for consistency with other collected data)
cell_distribution_known_samples_500v = pmt_correctionfactor(600,500)*cell_distribution_known_samples_600v # convert to 500volt

# Save as dataframe in the same format as the corresponding experimental data files
df_cellonly = DataFrame(tmpcolumnheader=cell_distribution_known_samples_500v);
rename!(df_cellonly,:tmpcolumnheader => :"638 Red(Peak)")

# Save to csv
filepath_save = pwd() * "/Data/Synthetic/"
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist
CSV.write(filepath_save * "data_syn_SP05_CELLONLY500V" * ".csv", df_cellonly) # Export to csv


##############################################################
## 4B - Sample cell fluorescence
##############################################################

# Load CSV
data_cellonly = CSV.read([filepath_save * "data_syn_SP05_CELLONLY500V" * ".csv"],DataFrame);

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
Psyndeter = [simulate_model_noiseless(θsyn,θfixed,[t]*3600,param_dist;N=samples_per_timepoint)[1] for t in T];

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
## 6A - Generate synthetic particle-only control data
##############################################################

# Compute number of samples in the experimental particle-only control data
nsamples_particle_only = length(data_particleonly[:,"638 Red(Peak)"]);

# Sample a known distribution
Random.seed!(102)
μP = mean(data_particleonly[:,"638 Red(Peak)"]);
σP = std(data_particleonly[:,"638 Red(Peak)"]);
particle_distribution_known = Normal(μP,σP);
particle_distribution_known_samples_600v = rand(particle_distribution_known,nsamples_particle_only);

mean(particle_distribution_known_samples_600v)
std(particle_distribution_known_samples_600v)
skewness(particle_distribution_known_samples_600v)

# Save as dataframe in the same format as the corresponding experimental data files
df_particleonly = DataFrame(tmpcolumnheader=particle_distribution_known_samples_600v);
rename!(df_particleonly,:tmpcolumnheader => :"638 Red(Peak)")

# Save to csv
filepath_save = pwd() * "/Data/Synthetic/"
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist
CSV.write(filepath_save * "data_syn_SP05_PARTICLEONLY600V" * ".csv", df_particleonly) # Export to csv


##############################################################
## 6B - Sample particle fluorescence (one value for each particle)
##############################################################

# Load CSV
data_particleonly = CSV.read([filepath_save * "data_syn_SP05_PARTICLEONLY600V" * ".csv"],DataFrame);

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
CSV.write(filepath_save * "data_cellonly_syn_SP05.csv", data_cellonly_syn) # Export to csv

#### Particle-only
# ID | F_p | 638 Red(Peak) (where 638 Red(Peak) is the same as F_p)
data_particleonly_syn = DataFrame(ID=1:samples_per_timepoint, F_p = F_p_t0, c= F_p_t0);
rename!(data_particleonly_syn,:c => "638 Red(Peak)")
CSV.write(filepath_save * "data_particleonly_syn_SP05.csv", data_particleonly_syn) # Export to csv

#### Cell-particle associations
# ID | F_cell | F_p | Psyndeter | Psyn | 638 Red(Peak) (where 638 Red(Peak) is the same as Psyn)
for i=1:length(T)
    data_syn = DataFrame(ID=1:samples_per_timepoint, F_cell = F_cell[i], F_p = F_p[i], Psyndeter = Psyndeter[i], Psyn = Psyn[i], c =Psyn[i] );
    rename!(data_syn,:c => "638 Red(Peak)")
    CSV.write(filepath_save * "data_syn_SP05_" * string(i) * ".csv", data_syn) # Export to csv
end
