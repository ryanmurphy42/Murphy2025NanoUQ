#=

    Model.jl

    A Julia module containing code to solve and simulate the model.

    Modified from https://github.com/ap-browning/internalisation by Alexander P. Browning (Queensland University of Technology)

=#

module Model

    using Distributions
    using Interpolations
    using KernelDensity
    using LinearAlgebra
    using Plots
    using Random
    using Statistics
    using StatsBase
    using StatsFuns
    using StatsPlots
    using Trapz

    export solve_ode_model_independent, solve_ode_model
    export simulate_model_noiseless, simulate_model_bootstrap_manyp

    include("deterministic.jl")
    include("statistical.jl")
    
end