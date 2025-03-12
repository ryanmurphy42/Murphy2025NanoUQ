#=
    OPTD_lowr_1B - Synthetic data for optimal experimental design study with low r (relative to K)
    Compute the ABC discrepancy for 200_000 random samples for each time point of the J=20 synthetic data sets 
=#

# 1 - Load packages and modules
# 2 - Setup location to save files
# 3 - Setup problem
# 4 - Load J synthetic data sets
# 5 - Compute ABC distance at each time point for the J synthetic data sets for each of the N=200_000 samples of priors

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
using StatsBase
using Base.Threads

println("Threads.nthreads = " * string(Threads.nthreads()))

##############################################################
## 2 - Setup location to save files
##############################################################

filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 3 - Setup problem
##############################################################

include("OPTD_lowr_Setup.jl")

T = [0.5;1.0:1.0:24.0;];

##############################################################
## 4 - Load Observed data - J data sets per time point (sample normal distributions for statistical hyperparameters)
##############################################################

J=20;

@load filepath_save * "OPTD_lowr_1A.jld2" observed_data_J

##############################################################
# 5 - Compute ABC distance at each time point for the J synthetic data sets for each of the N=200_000 samples of priors
# NOTE THIS SCRIPT IS PERFORMED IN A LOOP ON HPC FOR COMPUTATIONAL EFFICIENCY.
##############################################################

# High Performance Computing variables
println("ARGS ....")
# HPC_loop_id = 1;
HPC_loop_id = parse(Int,ARGS[1])
println(HPC_loop_id)



N=50_000; # number of data sets per time point (for this HPC loop)


# initialise variables to save outputs
discrepancy = zeros(N,J,length(T))
θsim_save = Vector{Any}(undef, N)

# create vector for calculations
k_vals = 2 .* (1:20_000) .- 1;

# convert the observed data to an ecdf for use in the Anderson-Darling distance
observed_data_J_ecdf = [[ecdf(observed_data_J[i,j]) for j=1:25] for i=1:20]

@time @threads for n=1:N    # loop through the N samples of prior distribution, simulate mathematical model
    println(n)
    Random.seed!(N*HPC_loop_id + J+n); # seed depends on println(HPC_loop_id)
    θsim = rand(prior,1)[:,1]; # sample the priors
    θsim_save[n]=θsim;
    sim_data_J_tmp = model(θsim); # simulate the mathematical model
    sim_data_J_tmp_sort = [sort!(sim_data_J_tmp[t]) for t in 1:length(T)];
    for j=1:J # for each of the J synthetic data sets compute the ABC distance
        for t=1:length(T) # for each of the length(T) timepoints compute the ABC distance
            # compute the Anderson-Darling distance
            a_ecdf = observed_data_J_ecdf[j][t]; 
            b_i_sort = sim_data_J_tmp_sort[t]; 
            Fobs = clamp.(a_ecdf.(b_i_sort[1:20_000]), 1e-9, 1 - 1e-9)
            discrepancy_j_t = -20_000 - sum(k_vals .* (log.(Fobs) .+ log.(1 .- reverse(Fobs)))) / 20_000
            discrepancy[n,j,t] =  discrepancy_j_t;
        end
    end
end


##############################################################
# 6- Save the outputs
##############################################################

@save filepath_save * "OPTD_lowr_1B_" * "θsim_save" * "_seed_" * string(HPC_loop_id)  * ".jld2" θsim_save
@save filepath_save * "OPTD_lowr_1B_" * "discrepancy" * "_seed_" * string(HPC_loop_id)  * ".jld2" discrepancy
