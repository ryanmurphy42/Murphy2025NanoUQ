#=
    S7ADE - Synthetic data  with low r (relative to K)
    Anderson-Darling ABC distance metric
    ABC-SMC Target acceptance threshold
=#

# 1 - Load packages and modules
# 2 - Setup location to save files
# 3 - Setup problem
# 4 - ABC SMC - target acceptance threshold
# 5 - Save results

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

include("S7AD_Setup.jl")

##############################################################
##  4 - ABC SMC - target acceptance threshold
##############################################################

Random.seed!(1)
@time P, p_accept, ε_seq  = abc_smc_εtarget(dist,prior,maxsteps=20,n=2000,α=0.75,εtarget=12.4);

##############################################################
## 5 - Save results
##############################################################

@save "Results/Files_syn/S7ADE_inference.jld2" P p_accept ε_seq
