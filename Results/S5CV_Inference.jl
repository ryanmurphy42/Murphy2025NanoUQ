#=
    S5CV - Synthetic data with intermediate r (relative to K)
    CramerVonMises ABC distance metric
    ABC-SMC Target acceptance probability
=#

# 1 - Load packages and modules
# 2 - Setup location to save files
# 3 - Setup problem
# 4 - ABC SMC - target acceptance probability
# 5 - Determine discrepancy threshold for best fit particle
# 6 - Save results

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using JLD2
using Random

##############################################################
## 2 - Setup location to save files
##############################################################

filepath_save = pwd() * "/" * "Results" * "/" * "Files_syn" * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 3 - Setup problem
##############################################################

include("S5CV_Setup.jl")

##############################################################
##  4 - ABC SMC - target acceptance probability
##############################################################

Random.seed!(1)
@time P = abc_smc(dist,prior,maxsteps=20;n=2000,ptarget=0.005,α=0.75);

##############################################################
## 5 - Determine discrepancy threshold for best fit particle
##############################################################

# The threshold is chosen so the "best fit particle" will be accepted ~50% of the time
@time D = [dist(minimum(P).θ) for i = 1:1_000];
ε = quantile(D,0.5);

##############################################################
## 6 - Save results
##############################################################

@save "Results/Files_syn/S5CV_inference.jld2" P ε 
