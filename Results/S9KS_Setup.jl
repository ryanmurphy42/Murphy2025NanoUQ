#=
    S9KS - Synthetic data for high r (relative to K)
    Kolmogorov Smirnov ABC distance metric
    Setup data, model, etc to produce the main result.
=#

# 1 - Load packages
# 2 -  Load data
# 3 - Sample the data
# 4 - Setup model
# 5 - ABC Distance metric - Anderson Darling
# 6 - Inference settings (prior, parameter names)

##############################################################
## 1 - Load packages
##############################################################

using Inference
using Model
using CSV
using DataFrames
using DataFramesMeta
using Distributions
using StatsBase
using Random
using Distances

##############################################################
## 2 -  Load data
##############################################################

######### Load INI files
filepathseparatorini = "/"
using IniFile

filepath_load_ini = pwd() * filepathseparatorini * "Data" * filepathseparatorini * "P_150nm_thp1" * filepathseparatorini;
dataini = read(Inifile(), filepath_load_ini * "leo_150nm_coreshell_thp1.ini")
    
# suspension culture (ODE)
S=parse(Float64,get(dataini, "cell", "total_cells"))*parse(Float64,get(dataini, "cell", "surface_area_per_cell_in_m2")); # surface area of the cell boundary [m^2]
V=pi*parse(Float64,get(dataini, "culture", "height_in_m"))*(parse(Float64,get(dataini, "culture", "width_in_m"))/2)^2; # volume of the solution [m^3]
C=1.0; # SC (dimensionless surface coverage of cells) [-] (in suspension all surface area of cells are in contact with solution)
U=parse(Float64,get(dataini, "culture", "particles_per_cell"))/V; # initial concentration of particles in the solution []

θfixed = [S,V,C,U];

######### Load fcs files 

filepathseparator = "/"

filepath_load = pwd() * filepathseparator * "Data" * filepathseparator * "Synthetic" * filepathseparator;
datafiles = ["data_syn_S09_1.csv",
            "data_syn_S09_2.csv",
            "data_syn_S09_3.csv",
            "data_syn_S09_4.csv",
            "data_syn_S09_5.csv",
            "data_syn_S09_6.csv"];

lendatafiles = length(datafiles);
data = [CSV.read([filepath_load * datafiles[i]], DataFrame) for i=1:lendatafiles]

Traw = [1,2,4,8,16,24];

filepath_load = pwd() * filepathseparator * "Data" * filepathseparator * "P_150nm_thp1" * filepathseparator;
data_cellonly = CSV.read([filepath_load * "THP-1 - Control - 2015-08-05.fcs.csv"],DataFrame);
data_particleonly = CSV.read([filepath_load * "110 nm CS (AF647N3).fcs.csv"],DataFrame);

voltage_cellonly = 500;
voltage_particleonly = 600;
voltage_data = [500,500,500,500,500,500];

function pmt_correctionfactor(V,Vref)
    y = exp(0.016*(Vref-V));
    return y
end

##############################################################
## 3 - Sample the data
##############################################################

T = [1.0,2.0,4.0,8.0,16.0,24.0];
data_to_analyse = [1,2,3,4,5,6]; # index of data files
xnsamples = 20_000;
Random.seed!(1)
# includes pmt voltage correction factor
X =  [pmt_correctionfactor(voltage_data[i],voltage_particleonly).*shuffle(data[i][!,"638 Red(Peak)"])[1:xnsamples] for i=data_to_analyse];
        
        
##############################################################
## 4 - Setup model
##############################################################

# Parameter distributions
function param_dist(ξ)
    μr,σr,μK,σK = ξ
    r = LogNormal(log(μr^2 / sqrt(σr^2 + μr^2)),sqrt(log(σr^2/μr^2 + 1)));
    K = LogNormal(log(μK^2 / sqrt(σK^2 + μK^2)),sqrt(log(σK^2/μK^2 + 1)));
    Product([r,K])
end

# # Empirical noise values
noise_cellonly = pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"];
noise_particleonly = shuffle(data_particleonly[:,"638 Red(Peak)"]);

# Model
model = θ -> simulate_model_bootstrap_manyp(θ,θfixed,T*3600,param_dist;N=20_000,noise_cellonly,noise_particleonly)


# #################################################
# # 5 - ABC Distance metric - Kolmogorov Smirnov
# #################################################

a_ecdf = [ecdf(X[i]) for i=1:length(T)];  # Empirical cdf of data

struct MyMetricSumKS <: Distances.UnionMetric end

@inline function eval_op_KS(::MyMetricSumKS, b,a) 
    # a - observed data
    # b - simulated data
    sumksdistance = 0;
    for i=1:size(b,1) # for each timepoint
        comparevec = unique(sort([b[i];X[i];]));
        b_ecdf = ecdf(b[i]); # evaluate the ecdf of the simulated data
        ksdistance = maximum(abs.(a_ecdf[i](comparevec) .- b_ecdf(comparevec)));
        sumksdistance = sumksdistance + ksdistance;
    end
    return sumksdistance
end
@eval @inline (dist::MyMetricSumKS)(a::AbstractArray, b::AbstractArray) = eval_op_KS(dist, a, b)
const mymetricsumks = MyMetricSumKS()

# ABC distance 
function makediscrepancy(X)
    dist = mymetricsumks(X,zeros(1))
    return dist
end
disc = makediscrepancy
dist = θ -> (disc ∘ model)(θ)

# #################################################
# # 6 - Inference settings (prior, parameter names)
# #################################################

# Prior
prior = Product([
    Uniform(1.0e-12,1.0e-4),# μr
    Uniform(1.0e-12,1.0e-4), # σr
    Uniform(0.1,40.0), # μK
    Uniform(0.01,20.0)  # σK
])

# Parameter names
param_names = ["μr","σr","μK","σK"]
