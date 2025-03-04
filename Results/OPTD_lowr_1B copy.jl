#=
    OPTD_1B.jl  
=#

##############################################################
## Load packages
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model

using JLD2
using LinearAlgebra
using Random
using StatsBase
using BenchmarkTools

##############################################################
## Setup save location
##############################################################

filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## Setup problem
##############################################################

include("OPTD_lowr_Setup.jl")

model = θ -> simulate_model_bootstrap_manyp(θ,θfixed,T*3600,param_dist;N=20_000,noise_cellonly,noise_particleonly)

T = [0.5;1.0:1.0:24.0;];

##############################################################
## Observed data - J data sets per time point (sample normal distributions for statistical hyperparameters)
##############################################################

J=20;

@load filepath_save * "OPTD_lowr_1A.jld2" observed_data_J

##############################################################
## Simulated data - N data sets per time point (sample uniform distributions for statistical hyperparameters)
# split
##############################################################

# println("ARGS ....")
# println(parse(Int,ARGS[1]))

# N=10_000; # number of data sets per time point
# N=100; # number of data sets per time point

# discrepancy = zeros(N,J,length(T))
# θsim_save = [];

# @time for n=1:N
#         println(n)
#         # Random.seed!(N*parse(Int,ARGS[1]) + J+n) # seed depends on println(parse(Int,ARGS[1]))
#         Random.seed!(N*1 + J+n) # seed depends on println(parse(Int,ARGS[1]))
#         θsim = rand(prior,1)[:,1];
#         push!(θsim_save,θsim)
#         sim_data_J_tmp = model(θsim);
#         for j=1:J
#             for t=1:length(T)
#                 discrepancy_j_t = (1/20_000)*sum(abs.( sort(sim_data_J_tmp[t]) .- sort(observed_data_J[j,t])));
#                 discrepancy[n,j,t] =  discrepancy_j_t;
#             end
#         end
#     end
# ### 668 seconds for 2000


# @save filepath_save * "OPTD_lowr_1B_" * "θsim_save" * "_seed_" * string(parse(Int,ARGS[1]))  * ".jld2" θsim_save
# @save filepath_save * "OPTD_lowr_1B_" * "discrepancy" * "_seed_" * string(parse(Int,ARGS[1]))  * ".jld2" discrepancy



##############################################################
## Simulated data - N data sets per time point (sample uniform distributions for statistical hyperparameters)
# split - Anderson-Darling
##############################################################

observed_data_J_ecdf = [[ecdf(observed_data_J[i,j]) for j=1:25] for i=1:20]
# observed_data_J_ecdf[i][j] # [dataset i][time j]

# N=10_000; # number of data sets per time point
N=100; # number of data sets per time point

discrepancy = zeros(N,J,length(T))
θsim_save = [];

@time for n=1:N
        println(n)
        # Random.seed!(N*parse(Int,ARGS[1]) + J+n) # seed depends on println(parse(Int,ARGS[1]))
        Random.seed!(N*1 + J+n) # seed depends on println(parse(Int,ARGS[1]))
        θsim = rand(prior,1)[:,1];
        push!(θsim_save,θsim)
        sim_data_J_tmp = model(θsim)
        for j=1:J
            for t=1:length(T)
                a_ecdf = observed_data_J_ecdf[j][t]
                b_i_sort = sort(sim_data_J_tmp[t])
                Fobs = [min(1-1e-9, max(1e-9,a_ecdf(b_i_sort[k]))) for k=1:20_000]
                discrepancy_j_t = -20_000 - sum([((2*k-1))*( log(Fobs[k]) + log(1 - Fobs[20_000+1-k])) for k=1:20_000])/20_000
                discrepancy[n,j,t] =  discrepancy_j_t;
            end
        end
    end
###  191.166852 seconds (103.49 M allocations: 38.181 GiB, 2.50% gc time, 1.39% compilation time)


# @save filepath_save * "OPTD_lowr_1B_" * "θsim_save" * "_seed_" * string(parse(Int,ARGS[1]))  * ".jld2" θsim_save
# @save filepath_save * "OPTD_lowr_1B_" * "discrepancy" * "_seed_" * string(parse(Int,ARGS[1]))  * ".jld2" discrepancy


##################### Optimising the code



# using BenchmarkTools

# @benchmark Fobs = [min(1-1e-9, max(1e-9,a_ecdf(b_i_sort[k]))) for k=1:20_000]
# BenchmarkTools.Trial: 692 samples with 1 evaluation.
#  Range (min … max):  5.030 ms … 49.664 ms  ┊ GC (min … max): 0.00% … 80.25%
#  Time  (median):     6.298 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   6.929 ms ±  3.575 ms  ┊ GC (mean ± σ):  4.39% ±  7.74%

#         ▁█▃▁    
#   █▇▅▃▃▃█████▇▆▄▄▃▃▃▃▃▃▃▁▃▃▃▃▃▂▂▃▃▃▂▂▁▂▂▂▂▁▁▁▂▂▂▂▁▂▁▁▂▁▂▂▁▁▂ ▃
#   5.03 ms        Histogram: frequency by time        13.2 ms <

#  Memory estimate: 1.67 MiB, allocs estimate: 99493.
# @benchmark Fobs = clamp.(a_ecdf.(b_i_sort[1:20_000]), 1e-9, 1 - 1e-9)
# BenchmarkTools.Trial: 2394 samples with 1 evaluation.
#  Range (min … max):  1.665 ms …  17.863 ms  ┊ GC (min … max): 0.00% … 85.90%
#  Time  (median):     1.874 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.079 ms ± 729.379 μs  ┊ GC (mean ± σ):  1.21% ±  3.58%

#   ▅▇   ▃█▂         
#   ██▆▃▃███▇▅▄▃▃▃▂▃▂▃▂▃▂▃▃▂▃▃▄▃▃▃▂▃▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁ ▂
#   1.67 ms         Histogram: frequency by time        3.21 ms <

#  Memory estimate: 312.72 KiB, allocs estimate: 7.


# @benchmark discrepancy_j_t_1 = -20_000 - sum([((2*k-1))*( log(Fobs[k]) + log(1 - Fobs[20_000+1-k])) for k=1:20_000])/20_000
# BenchmarkTools.Trial: 455 samples with 1 evaluation.
#  Range (min … max):   7.254 ms … 56.092 ms  ┊ GC (min … max): 0.00% … 79.77%
#  Time  (median):      9.742 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   10.511 ms ±  5.010 ms  ┊ GC (mean ± σ):  5.62% ±  9.81%

#   ▅▃▁██▅▅▃▂ ▂
#   █████████▇█▅▅▇▄▄▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄ ▇
#   7.25 ms      Histogram: log(frequency) by time      41.3 ms <

#  Memory estimate: 3.18 MiB, allocs estimate: 198729.

# k_vals = 2 .* (1:20_000) .- 1
# @benchmark discrepancy_j_t_2 = -20_000 - sum(k_vals .* (log.(Fobs) .+ log.(1 .- reverse(Fobs)))) / 20_000
# BenchmarkTools.Trial: 8783 samples with 1 evaluation.
#  Range (min … max):  477.100 μs …   9.955 ms  ┊ GC (min … max): 0.00% … 92.80%
#  Time  (median):     521.200 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   563.808 μs ± 297.795 μs  ┊ GC (mean ± σ):  1.84% ±  3.64%

#    █▂   ▃     
#   ▇██▇▆▇█▆▄▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
#   477 μs           Histogram: frequency by time          965 μs <

#  Memory estimate: 312.98 KiB, allocs estimate: 17.
# @benchmark discrepancy_j_t = -20_000 - dot(k_vals, log.(Fobs) .+ log.(1 .- reverse(Fobs))) / 20_000
# BenchmarkTools.Trial: 8165 samples with 1 evaluation.
#  Range (min … max):  478.600 μs …  27.920 ms  ┊ GC (min … max): 0.00% … 97.15%
#  Time  (median):     564.400 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   608.453 μs ± 760.912 μs  ┊ GC (mean ± σ):  4.28% ±  3.39%

#   █▁        ▃▂     
#   ███▆▄▄▃▃▃▅██▇▆▃▃▃▂▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
#   479 μs           Histogram: frequency by time          969 μs <

#  Memory estimate: 312.86 KiB, allocs estimate: 15.


using Base.Threads
Threads.nthreads()

k_vals = 2 .* (1:20_000) .- 1;

θsim_save = Vector{Any}(undef, N)

@time @threads for n=1:N
    println(n)
    # Random.seed!(N*parse(Int,ARGS[1]) + J+n) # seed depends on println(parse(Int,ARGS[1]))
    Random.seed!(N*1 + J+n) # seed depends on println(parse(Int,ARGS[1]))
    θsim = rand(prior,1)[:,1];
    θsim_save[n]=θsim;
    sim_data_J_tmp = model(θsim)
    sim_data_J_tmp_sort = [sort!(sim_data_J_tmp[t]) for t in 1:length(T)];
    for j=1:J
        for t=1:length(T)
            a_ecdf = observed_data_J_ecdf[j][t]
            b_i_sort = sim_data_J_tmp_sort[t]; 
            Fobs = clamp.(a_ecdf.(b_i_sort[1:20_000]), 1e-9, 1 - 1e-9)
            discrepancy_j_t = -20_000 - sum(k_vals .* (log.(Fobs) .+ log.(1 .- reverse(Fobs)))) / 20_000
            discrepancy[n,j,t] =  discrepancy_j_t;
        end
    end
end
### 10 threads (updated) ---  90.694230 seconds (102.64 M allocations: 38.225 GiB, 66.69% gc time, 8.46% compilation time)

### 12 threads --- 88.907967 seconds (101.67 M allocations: 45.510 GiB, 64.72% gc time, 1.41% compilation time)
### 10 threads --- 68.714576 seconds (104.22 M allocations: 45.677 GiB, 43.14% gc time, 45.61% compilation time)
