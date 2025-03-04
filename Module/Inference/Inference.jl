#=

    Inference.jl

    A Julia module code to perform ABC SMC with CDF/correlation matching.
    
    Modified from https://github.com/ap-browning/internalisation by Alexander P. Browning (Queensland University of Technology)

    # RM MODIFICATION - abc_smc_εtarget - is a modified function that terminates at a specified ABC error threshold

=#

module Inference

    using DataFrames
    using DataFramesMeta
    using Distributions
    using Interpolations
    using LinearAlgebra
    using .Threads
    using StatsBase
    using Statistics
    using StatsPlots
    using Printf
    
    export abc_smc, distances, locations, ess, abc_smc_εtarget
    export Particle

    include("abc.jl")
    include("particles.jl")


end
