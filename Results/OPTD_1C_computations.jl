#=
    OPTD_1C_computations - Compute ABC error threshold and discrepancy using ABC posteriors with at least 200 samples
=#

# 1 - Load packages and modules
# 2 - Loop through experimental conditions (using HPC)
    # 3 - Setup location to save files
    # 4 - Setup problem
    # 5 - Load pre-simulated parameter values and discrepancies
    # 6 - Generate possible times to measure Tdesign
    # 7 - Calculate ABC error threshold using all designs (ensure all ABC posteriors have at least 200 particles)
    # 8 - Calculate utility based on ABC error threshold

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Base.Threads
using Combinatorics
using Distributions
using Inference
using JLD2
using LaTeXStrings
using LinearAlgebra
using Model
using Plots
using Random
using StatsBase

##############################################################
# 2 - Loop through experimental conditions (using HPC)
##############################################################

if parse(Int,ARGS[1]) == 1
    save_name = "intr"
elseif parse(Int,ARGS[1]) == 2
    save_name = "lowr"
elseif parse(Int,ARGS[1]) == 3
    save_name = "highr"
end

println(save_name)

##############################################################
# 3 - Setup location to save files
##############################################################

filepath_load = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * "/"; # location to save figures 
filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * save_name * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 4 - Setup problem
##############################################################

T = [0.5;1.0:1.0:24.0;];
J=20;

##############################################################
## 5 - Load pre-simulated parameter values and discrepancies
##############################################################

# load the save files
Array_N = 4;

# θsim
θsim_save_all = [];
for i in 1:Array_N
    @load filepath_load * "OPTD_" * save_name * "_1B_" * "θsim_save" * "_seed_" * string(i) * ".jld2" θsim_save
    push!(θsim_save_all, θsim_save) 
end
θsim_save_all = vcat(θsim_save_all...)
size(θsim_save_all)

# discrepancies
discrepancy_all = [];
for i in 1:Array_N
    @load filepath_load * "OPTD_" * save_name * "_1B_" * "discrepancy" * "_seed_" * string(i) * ".jld2" discrepancy
    push!(discrepancy_all, discrepancy) 
end
discrepancy_all = vcat(discrepancy_all...)
size(discrepancy_all)

N=length(θsim_save_all);

##############################################################
## 6 - Generate possible times to measure Tdesign
##############################################################

possibleT = [0.5;1.0;2.0:2.0:24.0;]; # possible time points
possibleTcombinations1 = Combinatorics.combinations(possibleT, 6); # possible experimental designs
T_design = collect(possibleTcombinations1) # collect the possible experimental designs 
D = length(T_design); # total number of designs

##############################################################
## 7 - Calculate ABC error threshold using all designs (ensure all ABC posteriors have at least 200 particles)
##############################################################

ABC_N_lowest = 200;
ABC_error_N_lowest_D_all = zeros(D,J);

@time @threads for d=1:D 
    println("Design ..... " * string(d))
    # Identify index times
    T_index = [findall(x->x==T_design[d][i],T)[1] for i=1:length(T_design[d])];
    # Compute ABC error for each J and N
    ABC_error_N_lowest = zeros(J);
    for j=1:J
        # println("Design ..... " * string(d) * " --- " * "Data set ..... " * string(j))
        ##### ABC error 
        # Find lowest 200 particles
        ABC_error_tmp = [sum(discrepancy_all[n,j,T_index]) for n=1:N]; 
        ABC_error_N_lowest[j] = sort(ABC_error_tmp)[ABC_N_lowest];

    end
    ABC_error_N_lowest_D_all[d,:] = ABC_error_N_lowest;
end
# 1383 seconds

## Compute ABC error threshold
ε_comparison_threshold = maximum(ABC_error_N_lowest_D_all);

##############################################################
## 8 - Calculate utility based on ABC error threshold
##############################################################

utility_D = zeros(D,J);

@time @threads for d=1:D 
    println("Design ..... " * string(d))
    # Identify index times
    T_index = [findall(x->x==T_design[d][i],T)[1] for i=1:length(T_design[d])];
    for j=1:J
        # println("Design ..... " * string(d) * " --- " * "Data set ..... " * string(j))
        ##### ABC error 
        ABC_error_tmp = [sum(discrepancy_all[n,j,T_index]) for n=1:N];
        ##### Estimate utility
        θsim_tmp = [];
        for n=1:N
            if ABC_error_tmp[n] <= ε_comparison_threshold
                push!(θsim_tmp,θsim_save_all[n]);
            end
        end
        # Convert θsim_tmp to a matrix
        θsim_tmp2= mapreduce(permutedims, vcat, θsim_tmp);
        utility_D[d,j] = 1/det(cov(θsim_tmp2));
    end
end

# Compute mean utility (average over the J synthetic data sets)
utility_mean_D = [mean(utility_D[i,:]) for i=1:D];

##############################################################
## 9 - Save output 
##############################################################

@save filepath_save * "OPTD_1C_all" * ".jld2" ε_comparison_threshold utility_D  utility_mean_D θsim_save_all discrepancy_all  T_design D N  T J
