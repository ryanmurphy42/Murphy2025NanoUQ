#=
    OPTD_highr_1A - Synthetic data for optimal experimental design study with high r (relative to K)
    Generate J=20 synthetic data sets
=#

# 1 - Load packages and modules
# 2 - Setup location to save files
# 3 - Setup problem
# 4 - Setup model
# 5 - ABC Distance metric - Anderson Darling

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using JLD2
using LinearAlgebra
using Random

##############################################################
## 2 - Setup location to save files
##############################################################

filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 3 - Setup problem
##############################################################

include("OPTD_highr_Setup.jl")

T = [0.5;1.0:1.0:24.0;];

##############################################################
## 4 - Observed data - J data sets per time point (sample normal distributions for statistical hyperparameters)
##############################################################

J=20; # number of data sets per time point 

# prior distribution for J=20 synthetic data sets
prior_observed = Product([
    Truncated(Normal(3.86125e-7*100,3.86125e-7*100/100),0,Inf), # μr
    Truncated(Normal(4.56962e-7*100,4.56962e-7*100/100),0,Inf), # σr
    Truncated(Normal(10.0,0.1),0,Inf), # μK
    Truncated(Normal(2.0,0.02),0,Inf)  # σK
])

# initialise variable to save J=20 synthetic data sets
observed_data_J = Array{Vector{Float64},2}(undef,J,length(T))    ;
θobserved_J = [];

# generate J=20 synthetic data sets
@time for j=1:J
            Random.seed!(j)
            θobserved_tmp = rand(prior_observed,1)[:,1];
            observed_data_J_tmp = model(θobserved_tmp);
            push!(θobserved_J,θobserved_tmp)
            for t=1:length(T)
                observed_data_J[j,t] = sort(observed_data_J_tmp[t]);
            end
    end

# save the the J=20 synthetic data sets
@save filepath_save * "OPTD_highr_1A.jld2" observed_data_J  θobserved_J
