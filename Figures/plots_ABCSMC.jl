#=
    PLOTTING ABC-SMC RESULTS
    Parameter inference, identifiability analysis, and prediction
=#

# 1 - Load packages and modules
# 2 - Default plot settings (plot sizes, axis limits, ticks)
# 3 - Data to plot (uncomment the relevant code)
# 4 - Create folder to save results
# 5 - Load ABC-SMC results
# 6 - Plots -  Univariate posterior distributions - with the 95% highest posterior density region shaded
# 7 - Plots  - Bivariate posterior densities
# 8 - Compute ABC mean and data for ECDF, lognormal, histogram plots
# 9 - Plots - Inferred distributions for r and K (95% prediction, mean, mode)
# 10 - Plots  - Inferred predictions for number of associated particles, P(t)
# 11 - Plot  - Time series - Histogram of fluorescence data vs uncertainity in simulated data from posterior distribution
# 12 - Plot - percentage of cells with P(t) > 0.99K

#########################################
# 1 - Load packages and modules
#########################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using Base.Threads
using CSV
using DataFrames
using DataFramesMeta
using Distributions
using JLD2
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using Plots
using Random
using StatsBase
using StatsPlots

#########################################
# 2 - Default plot settings (plot sizes, axis limits, ticks)
#########################################

pyplot() # plot options
fnt = Plots.font("sans-serif", 10); # plot options
fnt_inset = Plots.font("sans-serif", 10); # plot options
global cur_colors = palette(:default); # plot options

# default figure sizes
mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

# density - size
fig_density_μr_size=fig_3_across_size;
fig_density_σr_size=fig_3_across_size
fig_density_μK_size=fig_3_across_size
fig_density_σK_size=fig_3_across_size
# density - xlim
fig_density_μr_xlim=[0,1.2e-6];
fig_density_σr_xlim=[0,1.2e-5];
fig_density_μK_xlim=[0,100];
fig_density_σK_xlim=[0,100];
# density - xticks
fig_density_μr_xticks=[0,0.5e-6,1.0e-6];
fig_density_σr_xticks=[0,0.5e-5,1.0e-5];
fig_density_μK_xticks=[0,50,100];
fig_density_σK_xticks=[0,50,100];

# density inset r
fig_density_μr_inset_size=(22.5*mm_to_pts_scaling,15.0*mm_to_pts_scaling);
fig_density_σr_inset_size=(22.5*mm_to_pts_scaling,15.0*mm_to_pts_scaling);

# inferred distribution - r 
fig_inferreddistributions_r_size=fig_3_across_size;
fig_inferreddistributions_r_xlim  =[0,5e-6];
fig_inferreddistributions_r_xticks =[0,2e-6,4e-6];

# inferred distribution - K
fig_inferreddistributions_K_size=fig_3_across_size;
fig_inferreddistributions_K_xlim =[0,250];
fig_inferreddistributions_K_xticks = [0,100,200];

# P(t) 
fig_inferred_predictions_size =fig_3_across_size;
fig_inferred_predictions_xlim = [0,24];
fig_inferred_predictions_xticks = [0,12,24];
fig_inferred_predictions_ylim= [0,100];
fig_inferred_predictions_yticks = [0,50,100];

# Histograms of fluorescence data - particle only
fig_data_hist2_particleonly_size=fig_3_across_size;
fig_data_hist2_particleonly_xlim = [0,1000]
fig_data_hist2_particleonly_xticks = [0,500,1000]
fig_data_hist2_particleonly_ylim = [0,0.02]
fig_data_hist2_particleonly_yticks = [0,0.01,0.02]

# Histograms of fluorescence data - cell only
fig_data_hist2_cellonly_size =fig_3_across_size;
fig_data_hist2_cellonly_xlim = [0,1000]
fig_data_hist2_cellonly_xticks = [0,500,1000]
fig_data_hist2_cellonly_ylim = [0,0.01]
fig_data_hist2_cellonly_yticks = [0,0.01]

# Histograms of fluorescence data - time course
fig_data_hist2_size=fig_3_across_size;
fig_data_hist2_xlim = [0,1000]
fig_data_hist2_xticks = [0,500,1000]
fig_data_hist2_ylim = [0,0.006]
fig_data_hist2_yticks = [0,0.005]

# Percentage plot
fig_percentg99K_size = fig_2_across_size;

#########################################
# 3 - Data to plot (uncomment the relevant code)
#########################################

###### Experimental data set - R1ADE
# loadpath = "Results/Files_real/";
# include(pwd() * "/Results/R1AD_Setup.jl")
# savename = "Fig_R1ADE_Inference";
# loadfilename = "R1AD_ABCE_inference"
# syndataindicator = 0;
# θleastsquares = []; # r, K
# fig_density_σK_xlim=[0,200];
# fig_density_σK_xticks=[0,100,200];
# fig_inferreddistributions_r_xlim =[0,5e-8];
# fig_inferreddistributions_r_xticks =[0,2e-8,4e-8];
# lsdataindicator = 1;
# @load "Results/Files_syn/Lesq_S1.jld2" Tplot xopt median_data_syn
# ls_Tplot = copy(Tplot);
# ls_xopt = copy(xopt);
# ls_median_data_syn = copy(median_data_syn);
# ls_P_yticks =[0,10,20];
# ls_P_ylim = [0,20];

###### Experimental data set - R5ADE - 214nm
# loadpath = "Results/Files_real/";
# include(pwd() * "/Results/R5AD_Setup.jl")
# # savename = "Fig_R5ADE_Inference";
# # loadfilename = "R5ADE_inference"
# savename = "Fig_R5AD_ABCE_Inference";
# loadfilename = "R5AD_ABCE_inference"
# syndataindicator = 0;
# θleastsquares = []; # r, K
# # custom plot setting - density
# fig_density_μr_xlim=[0,1.2e-3];
# fig_density_σr_xlim=[0,1.2e-0];
# fig_density_μK_xlim=[0,5];
# fig_density_σK_xlim=[0,5];
# fig_density_μr_xticks=[0,0.5e-3,1.0e-3];
# fig_density_σr_xticks=[0,0.5e-0,1.0e-0];
# fig_density_μK_xticks=[0,2,4];
# fig_density_σK_xticks=[0,2,4];
# # P(t) 
# fig_inferred_predictions_ylim= [0,20];
# fig_inferred_predictions_yticks = [0,10,20];


###### Experimental data set - R6ADE - 633nm
# loadpath = "Results/Files_real/";
# include(pwd() * "/Results/R6AD_Setup.jl")
# # savename = "Fig_R6ADE_Inference";
# # loadfilename = "R6ADE_inference"
# savename = "Fig_R6AD_ABCE_Inference";
# loadfilename = "R6AD_ABCE_inference"
# syndataindicator = 0;
# θleastsquares = []; # r, K


###### Synthetic data set S1ADE - 
loadpath = "Results/Files_syn/";
include(pwd() * "/Results/S1AD_Setup.jl")
savename = "Fig_S1ADE_Inference";
loadfilename = "S1ADE_inference"
syndataindicator = 1;
θknown = [2.9636409772399683e-7,2.929418226130372e-6,51.16803788865989,53.2813251769441];
inset_r=0;
fig_density_σK_xlim=[0,200];
fig_density_σK_xticks=[0,100,200];
fig_inferreddistributions_r_xlim =[0,5e-8];
fig_inferreddistributions_r_xticks =([0,2e-8,4e-8],[0,2,4]);
fig_density_μr_xticks=([0,0.5e-6,1.0e-6],[0,5,10]);
fig_density_σr_xticks=([0,0.5e-5,1.0e-5],[0,5,10]);
fig_inferreddistributions_r_size=fig_2_across_size;
fig_inferreddistributions_K_size=fig_2_across_size;
fig_inferred_predictions_size=fig_2_across_size;
lsdataindicator = 1;
@load "Results/Files_syn/Lesq_S1.jld2" Tplot xopt median_data_syn
ls_Tplot = copy(Tplot);
ls_xopt = copy(xopt);
ls_median_data_syn = copy(median_data_syn);
ls_P_yticks =[0,10,20];
ls_P_ylim = [0,20];

###### Synthetic data set S5ADE - 
# loadpath = "Results/Files_syn/";
# include(pwd() * "/Results/S5AD_Setup.jl")
# savename = "Fig_S5ADE_Inference";
# loadfilename = "S5ADE_inference"
# syndataindicator = 1;
# θknown = [3.86125e-7,4.56962e-7,10.0,2.0];
# # θleastsquares = []; # r, K
# # custom plot setting - density
# fig_density_μr_xlim=[0,1.2e-4];
# fig_density_σr_xlim=[0,1.2e-4];
# fig_density_μK_xlim=[0,40];
# fig_density_σK_xlim=[0,20];
# fig_density_μr_xticks=([0,5e-5,10e-5],[0,5,10]);
# fig_density_σr_xticks=([0,5e-5,10e-5],[0,5,10]);
# fig_density_μK_xticks=[0,20,40];
# fig_density_σK_xticks=[0,10,20];
# # P(t) 
# fig_inferred_predictions_ylim= [0,20];
# fig_inferred_predictions_yticks = [0,10,20];
# # inferred r
# fig_inferreddistributions_r_xlim =[0,10e-7];
# fig_inferreddistributions_r_xticks =([0,4e-7,8e-7],[0,4,8]);
# # inferred K
# fig_inferreddistributions_K_xlim =[0,60];
# fig_inferreddistributions_K_xticks = [0,25,50];
# # inset
# inset_r=1;
# fig_density_μr_inset_xticks=([0,5e-7,10e-7],[0,5,10]);
# fig_density_σr_inset_xticks=([0,5e-7,10e-7],[0,5,10]);
# fig_density_μr_inset_xlim=[0,12e-7];
# fig_density_σr_inset_xlim=[0,12e-7];
# fig_inferreddistributions_r_size=fig_2_across_size;
# fig_inferreddistributions_K_size=fig_2_across_size;
# lsdataindicator = 1;
# @load "Results/Files_syn/Lesq_S5.jld2" Tplot xopt median_data_syn
# ls_Tplot = copy(Tplot);
# ls_xopt = copy(xopt);
# ls_median_data_syn = copy(median_data_syn);
# ls_P_yticks =[0,10];
# ls_P_ylim = [0,15];


###### Synthetic data set S7ADE - 
# loadpath = "Results/Files_syn/";
# include(pwd() * "/Results/S7AD_Setup.jl")
# savename = "Fig_S7ADE_Inference";
# loadfilename = "S7ADE_inference"
# syndataindicator = 1;
# θknown = [3.86125e-7*0.01,4.56962e-7*0.01,10.0,2.0];
# # θleastsquares = []; # r, K
# # custom plot setting - density
# fig_density_μr_xlim=[0,1.2e-4];
# fig_density_σr_xlim=[0,1.2e-4];
# fig_density_μK_xlim=[0,40];
# fig_density_σK_xlim=[0,20];
# fig_density_μr_xticks=([0,5e-5,10e-5],[0,5,10]);
# fig_density_σr_xticks=([0,5e-5,10e-5],[0,5,10]);
# fig_density_μK_xticks=[0,20,40];
# fig_density_σK_xticks=[0,10,20];
# # inferred r
# # fig_inferreddistributions_r_xlim =[0,5e-8];
# # fig_inferreddistributions_r_xticks =[0,2e-8,4e-8];
# fig_inferreddistributions_r_xlim =[0,10e-9];
# fig_inferreddistributions_r_xticks =([0,4e-9,8e-9],[0,4,8]);
# # inferred K
# fig_inferreddistributions_K_xlim =[0,60];
# fig_inferreddistributions_K_xticks = [0,25,50];
# # P(t) 
# fig_inferred_predictions_ylim= [0,20];
# fig_inferred_predictions_yticks = [0,10,20];
# # inset
# inset_r=1;
# fig_density_μr_inset_xticks=([0,5e-9,10e-9],[0,5,10]);
# fig_density_σr_inset_xticks=([0,5e-9,10e-9],[0,5,10]);
# fig_density_μr_inset_xlim=[0,12e-9];
# fig_density_σr_inset_xlim=[0,12e-9];
# fig_inferreddistributions_r_size=fig_2_across_size;
# fig_inferreddistributions_K_size=fig_2_across_size;
# fig_data_hist2_ylim = [0,0.01]
# fig_data_hist2_yticks = [0,0.01]
# lsdataindicator = 1;
# @load "Results/Files_syn/Lesq_S7.jld2" Tplot xopt median_data_syn
# ls_Tplot = copy(Tplot);
# ls_xopt = copy(xopt);
# ls_median_data_syn = copy(median_data_syn);
# ls_P_yticks =[0,2];
# ls_P_ylim = [0,3];

# ###### Synthetic data set 9ADE - 
# loadpath = "Results/Files_syn/";
# include(pwd() * "/Results/S9AD_Setup.jl")
# savename = "Fig_S9ADE_Inference";
# loadfilename = "S9ADE_inference"
# syndataindicator = 1;
# θknown = [3.86125e-7*100,4.56962e-7*100,10.0,2.0];
# # θleastsquares = []; # r, K
# # custom plot setting - density
# fig_density_μr_xlim=[0,1e-4];
# fig_density_σr_xlim=[0,1e-4];
# fig_density_μK_xlim=[0,40];
# fig_density_σK_xlim=[0,20];
# fig_density_μr_xticks=([0,4e-5,8e-5],[0,4,8]);
# fig_density_σr_xticks=([0,4e-5,8e-5],[0,4,8]);
# fig_density_μK_xticks=[0,20,40];
# fig_density_σK_xticks=[0,10,20];
# # inferred r
# fig_inferreddistributions_r_xlim = [0,0.00025];
# fig_inferreddistributions_r_xticks =[0.0000,0.0001,0.0002];
# # inferred K
# fig_inferreddistributions_K_xlim =[0,60];
# fig_inferreddistributions_K_xticks = [0,25,50];
# # P(t) 
# fig_inferred_predictions_ylim= [0,20];
# fig_inferred_predictions_yticks = [0,10,20];
# # inset
# inset_r=0;
# fig_inferreddistributions_r_size=fig_2_across_size;
# fig_inferreddistributions_K_size=fig_2_across_size;
# lsdataindicator = 1;
# @load "Results/Files_syn/Lesq_S9.jld2" Tplot xopt median_data_syn
# ls_Tplot = copy(Tplot);
# ls_xopt = copy(xopt);
# ls_median_data_syn = copy(median_data_syn);
# ls_P_yticks =[0,10];
# ls_P_ylim = [0,15];


#########################################
# 4 - Create folder to save results
#########################################

filepathseparator = "/";
filepath_save = pwd() * filepathseparator * "Figures" * filepathseparator * savename * filepathseparator; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

#########################################
# 5 - Load ABC-SMC results
#########################################

# @load loadpath * loadfilename * ".jld2" P ε # load ABC-SMC results with target acceptance probability
@load loadpath * loadfilename * ".jld2" P p_accept ε_seq # load ABC-SMC results with target acceptance threshold

density_samples_μr = [P[i].θ[1] for i=1:length(P)];
density_samples_σr = [P[i].θ[2] for i=1:length(P)];
density_samples_μK = [P[i].θ[3] for i=1:length(P)];
density_samples_σK = [P[i].θ[4] for i=1:length(P)];

#########################################
# 6 - Plots - Univariate posterior distributions - with the 95% highest posterior density region shaded
#########################################

# initialise figure
fig_density_μr = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_μr_size);
fig_density_σr = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_σr_size);
fig_density_μK = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_μK_size);
fig_density_σK = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_σK_size);

# function to compute highest posterior density
function compute_hpd(x::AbstractVector{<:Real}; alpha::Real=0.05)
    n = length(x)
    m = max(1, ceil(Int, alpha * n))
    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)
    return [a[i], b[i]]
end

# compute highest posterior density
hpd_μr = compute_hpd(density_samples_μr; alpha=0.05)
hpd_σr = compute_hpd(density_samples_σr; alpha=0.05)
hpd_μK = compute_hpd(density_samples_μK; alpha=0.05)
hpd_σK = compute_hpd(density_samples_σK; alpha=0.05)
# rounding
hpd_μr_rnd = round.(hpd_μr,sigdigits=3);
hpd_σr_rnd = round.(hpd_σr,sigdigits=3);
hpd_μK_rnd = round.(hpd_μK,sigdigits=3);
hpd_σK_rnd = round.(hpd_σK,sigdigits=3);

# Export MLE and bounds to csv (one file for all data) 
global df_hpd = DataFrame(hpd_μr_rnd=hpd_μr_rnd,hpd_σr_rnd=hpd_σr_rnd,hpd_μK_rnd=hpd_μK_rnd,hpd_σK_rnd=hpd_σK_rnd)
CSV.write(filepath_save * "df_hpd.csv", df_hpd)

##### plot hpd for μr
ksamp_μr = kde(density_samples_μr)
x_μr = collect(ksamp_μr.x)
y_μr = ksamp_μr.density

qa_μr, qb_μr = hpd_μr;
xi_μr = findall(i -> (qa_μr<=i) & (i<=qb_μr), x_μr)
plot!(fig_density_μr,x_μr[xi_μr], y_μr[xi_μr],fillrange=zeros(1),legend=false,color=:lightskyblue1)

##### plot hpd for σr
ksamp_σr = kde(density_samples_σr)
x_σr = collect(ksamp_σr.x)
y_σr = ksamp_σr.density

qa_σr, qb_σr = hpd_σr;
xi_σr = findall(i -> (qa_σr<=i) & (i<=qb_σr), x_σr)
plot!(fig_density_σr,x_σr[xi_σr], y_σr[xi_σr],fillrange=zeros(1),legend=false,color=:lightskyblue1)

##### plot hpd for μK
ksamp_μK = kde(density_samples_μK)
x_μK = collect(ksamp_μK.x)
y_μK = ksamp_μK.density

qa_μK, qb_μK = hpd_μK;
xi_μK = findall(i -> (qa_μK<=i) & (i<=qb_μK), x_μK)
plot!(fig_density_μK,x_μK[xi_μK], y_μK[xi_μK],fillrange=zeros(1),legend=false,color=:lightskyblue1)

##### plot hpd for σK
ksamp_σK = kde(density_samples_σK)
x_σK = collect(ksamp_σK.x)
y_σK = ksamp_σK.density

qa_σK, qb_σK = hpd_σK;
xi_σK = findall(i -> (qa_σK<=i) & (i<=qb_σK), x_σK)
plot!(fig_density_σK,x_σK[xi_σK], y_σK[xi_σK],fillrange=zeros(1),legend=false,color=:lightskyblue1)

##### plot density 
density!(fig_density_μr,density_samples_μr,lw=2,legend=false,color=:blue,yticks=[],xlab=L"m_{r} \ [\mathrm{m \ s}^{-1}]",xticks=fig_density_μr_xticks,xlim=fig_density_μr_xlim,frame=:box)
density!(fig_density_σr,density_samples_σr,lw=2,legend=false,color=:blue,yticks=[],xlab=L"s_{r} \ [\mathrm{m \ s}^{-1}]",xticks=fig_density_σr_xticks,xlim=fig_density_σr_xlim,frame=:box)
density!(fig_density_μK,density_samples_μK,lw=2,legend=false,color=:blue,yticks=[],xlab=L"m_{K} \ [-]",xticks=fig_density_μK_xticks,xlim=fig_density_μK_xlim,frame=:box)
density!(fig_density_σK,density_samples_σK,lw=2,legend=false,color=:blue,yticks=[],xlab=L"s_{K} \ [-]",xticks=fig_density_σK_xticks,xlim=fig_density_σK_xlim,frame=:box)

# display
display(fig_density_μr)
display(fig_density_σr)
display(fig_density_μK)
display(fig_density_σK)

# save
savefig(fig_density_μr,filepath_save * "fig_density_μr_hpd" * ".pdf")
savefig(fig_density_σr,filepath_save * "fig_density_σr_hpd" * ".pdf")
savefig(fig_density_μK,filepath_save * "fig_density_μK_hpd" * ".pdf")
savefig(fig_density_σK,filepath_save * "fig_density_σK_hpd" * ".pdf")

### For synthetic data add vertical lines representing the known parameter values used to generate the data
if syndataindicator == 1
    vline!(fig_density_μr,[θknown[1]],lw=2,color=:orange,ls=:dash)
    vline!(fig_density_σr,[θknown[2]],lw=2,color=:orange,ls=:dash)
    vline!(fig_density_μK,[θknown[3]],lw=2,color=:orange,ls=:dash)
    vline!(fig_density_σK,[θknown[4]],lw=2,color=:orange,ls=:dash)
    display(fig_density_μr)
    display(fig_density_σr)
    display(fig_density_μK)
    display(fig_density_σK)
    savefig(fig_density_μr,filepath_save * "fig_density_μr_hpd_synvaluesshown" * ".pdf")
    savefig(fig_density_σr,filepath_save * "fig_density_σr_hpd_synvaluesshown" * ".pdf")
    savefig(fig_density_μK,filepath_save * "fig_density_μK_hpd_synvaluesshown" * ".pdf")
    savefig(fig_density_σK,filepath_save * "fig_density_σK_hpd_synvaluesshown" * ".pdf")

    # display inset for r (to zoom in)
    if inset_r == 1

        fig_density_μr_inset = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_μr_inset_size);
        fig_density_σr_inset = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_density_σr_inset_size);
        plot!(fig_density_μr_inset,x_μr[xi_μr], y_μr[xi_μr],fillrange=zeros(1),legend=false,color=:lightskyblue1,frame=:box)
        plot!(fig_density_σr_inset,x_σr[xi_σr], y_σr[xi_σr],fillrange=zeros(1),legend=false,color=:lightskyblue1,frame=:box)
        density!(fig_density_μr_inset,density_samples_μr,lw=2,legend=false,color=:blue,yticks=[],xticks=fig_density_μr_inset_xticks,xlim=fig_density_μr_inset_xlim)
        density!(fig_density_σr_inset,density_samples_σr,lw=2,legend=false,color=:blue,yticks=[],xticks=fig_density_σr_inset_xticks,xlim=fig_density_σr_inset_xlim)
        vline!(fig_density_μr_inset,[θknown[1]],lw=2,color=:orange,ls=:dash)
        vline!(fig_density_σr_inset,[θknown[2]],lw=2,color=:orange,ls=:dash)    
        display(fig_density_μr_inset)
        display(fig_density_σr_inset)
        savefig(fig_density_μr_inset,filepath_save * "fig_density_μr_inset_hpd_synvaluesshown" * ".pdf")
        savefig(fig_density_μr_inset,filepath_save * "fig_density_σr_inset_hpd_synvaluesshown" * ".pdf")

    end
end

#################################################
# 7 - Plots  - Bivariate posterior densities
#################################################  

vcat_μr = deepcopy(density_samples_μr);
vcat_σr = deepcopy(density_samples_σr);
vcat_μK = deepcopy(density_samples_μK);
vcat_σK = deepcopy(density_samples_σK);

## μr v σr
k1 = kde((vcat_μr, vcat_σr))
fig_bivariate_μr_v_σr = contour(k1,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"m_{r}",ylabel=L"s_{r}",titlefont=fnt, guidefont=fnt, tickfont=fnt,xticks=[0,0.25e-6,0.75e-6],xlim=[1.0e-11,1.0e-6],yticks=[0,0.25e-5,0.75e-5],ylim=[1.0e-11,1.0e-5],legend = :none,frame=:box)
display(fig_bivariate_μr_v_σr)
savefig(fig_bivariate_μr_v_σr,filepath_save * "fig_bivariate_μr_v_σr" * ".pdf")

## μr v μK
k2 = kde((vcat_μr, vcat_μK))
fig_bivariate_μr_v_μK = contour(k2,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"m_{r}",ylabel=L"m_{K}",frame=:box)
savefig(fig_bivariate_μr_v_μK,filepath_save * "fig_bivariate_μr_v_μK" * ".pdf")

## μr v σK
k3 = kde((vcat_μr, vcat_σK))
fig_bivariate_μr_v_σK = contour(k3,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"m_{r}",ylabel=L"s_{K}",frame=:box)
savefig(fig_bivariate_μr_v_σK,filepath_save * "fig_bivariate_μr_v_σK" * ".pdf")

## σr v μK
k4 = kde((vcat_σr, vcat_μK))
fig_bivariate_σr_v_μK = contour(k4,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"s_{r}",ylabel=L"m_{K}",frame=:box)
savefig(fig_bivariate_σr_v_μK,filepath_save * "fig_bivariate_σr_v_μK" * ".pdf")

## σr v σK
k5 = kde((vcat_σr, vcat_σK))
fig_bivariate_σr_v_σK = contour(k5,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"s_{r}",ylabel=L"s_{K}",frame=:box)
savefig(fig_bivariate_σr_v_σK,filepath_save * "fig_bivariate_σr_v_σK" * ".pdf")

## μK v σK
k6 = kde((vcat_μK, vcat_σK))
fig_bivariate_μK_v_σK = contour(k6,c=cgrad(:devon, rev = true),linewidth=1,xlabel=L"m_{K}",ylabel=L"s_{K}",frame=:box)
savefig(fig_bivariate_μK_v_σK,filepath_save * "fig_bivariate_μK_v_σK" * ".pdf")

# display
display(fig_bivariate_μr_v_σr)
display(fig_bivariate_μr_v_μK)
display(fig_bivariate_μr_v_σK)
display(fig_bivariate_σr_v_μK)
display(fig_bivariate_σr_v_σK)
display(fig_bivariate_μK_v_σK)

### For synthetic data add dots representing the known parameter values used to generate the data
if syndataindicator == 1

    ## μr v σr
    scatter!(fig_bivariate_μr_v_σr,[θknown[1]],[θknown[2]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_μr_v_σr,filepath_save * "fig_bivariate_μr_v_σr_synvaluesshown" * ".pdf")

    ## μr v μK
    scatter!(fig_bivariate_μr_v_μK,[θknown[1]],[θknown[3]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_μr_v_μK,filepath_save * "fig_bivariate_μr_v_μK_synvaluesshown" * ".pdf")

    ## μr v σK
    scatter!(fig_bivariate_μr_v_σK,[θknown[1]],[θknown[4]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_μr_v_σK,filepath_save * "fig_bivariate_μr_v_σK_synvaluesshown" * ".pdf")

    ## σr v μK
    scatter!(fig_bivariate_σr_v_μK,[θknown[2]],[θknown[3]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_σr_v_μK,filepath_save * "fig_bivariate_σr_v_μK_synvaluesshown" * ".pdf")

    ## σr v σK
    scatter!(fig_bivariate_σr_v_σK,[θknown[2]],[θknown[4]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_σr_v_σK,filepath_save * "fig_bivariate_σr_v_σK_synvaluesshown" * ".pdf")

    ## μK v σK
    scatter!(fig_bivariate_μK_v_σK,[θknown[3]],[θknown[4]],color=:orange,legend=false,msw=0,markersize=10)
    savefig(fig_bivariate_μK_v_σK,filepath_save * "fig_bivariate_μK_v_σK_synvaluesshown" * ".pdf")

    display(fig_bivariate_μr_v_σr)
    display(fig_bivariate_μr_v_μK)
    display(fig_bivariate_μr_v_σK)
    display(fig_bivariate_σr_v_μK)
    display(fig_bivariate_σr_v_σK)
    display(fig_bivariate_μK_v_σK)
    
end


#################################################
# 8 - Compute ABC mean and data for ECDF, lognormal, histogram plots
#################################################  

# compute ABC mean
ABCmean = [mean(density_samples_μr),mean(density_samples_σr),mean(density_samples_μK),mean(density_samples_σK)];

# sample the ABC posterior 2000 times (all samples) and identify the parameter values μr, σr, μK, σK
nrandsampleshist= 2000;
X_sim_param_rand =[[density_samples_μr[i]
                    density_samples_σr[i]
                    density_samples_μK[i]
                    density_samples_σK[i]] for i=1:nrandsampleshist];

# simulate the model for all 2000 parameter samples and save the output
X_sim_mean = Vector{Vector{Vector{Float64}}}(undef, nrandsampleshist);
@time for i=1:nrandsampleshist
    Random.seed!(i)
    X_sim_mean[i] =  model(X_sim_param_rand[i]);
end


#################################################
# 9 - Plots - Inferred distributions for r and K (95% prediction, mean, mode)
#################################################  

# define lognormal distrbutions for r and K
rdist(μr,σr) = LogNormal(log(μr^2 / sqrt(σr^2 + μr^2)),sqrt(log(σr^2/μr^2 + 1)));
Kdist(μK,σK) = LogNormal(log(μK^2 / sqrt(σK^2 + μK^2)),sqrt(log(σK^2/μK^2 + 1)));

# define range to evaluate r and K
rdistquantile_min = 1e-15; 
Kdistquantile_min = 0;
rdistquantile_max = max(1e-3,min(1e-5,quantile(rdist(ABCmean[1],ABCmean[2]),0.99)));
Kdistquantile_max = max(250,min(250,quantile(Kdist(ABCmean[3],ABCmean[4]),0.99)));

# evaluate distribution for r and K 
r_eval = exp.(LinRange(log(rdistquantile_min),log(rdistquantile_max),501));
K_eval = LinRange(Kdistquantile_min,Kdistquantile_max,501);

# simulate the inferred distribution for the 2000 ABC particles
r_save = zeros(nrandsampleshist,length(r_eval));
K_save = zeros(nrandsampleshist,length(K_eval));
for i=1:nrandsampleshist
    r_save[i,:] = pdf.(rdist(X_sim_param_rand[i][1],X_sim_param_rand[i][2]),r_eval)
    K_save[i,:] = pdf.(Kdist(X_sim_param_rand[i][3],X_sim_param_rand[i][4]),K_eval)
end

## Compute the quantiles for the inferred distribution for r
r_eval_quantilea = zeros(length(r_eval));
r_eval_quantileb = zeros(length(r_eval));
r_eval_quantilec = zeros(length(r_eval));
for i=1:length(r_eval)
    r_eval_quantilea[i] = quantile(r_save[:,i],0.025)
    r_eval_quantileb[i] = quantile(r_save[:,i],0.5)
    r_eval_quantilec[i] = quantile(r_save[:,i],0.975)
end

## Compute the quantiles for the inferred distributions for K
K_eval_quantilea = zeros(length(K_eval));
K_eval_quantileb = zeros(length(K_eval));
K_eval_quantilec = zeros(length(K_eval));
for i=1:length(r_eval)
    K_eval_quantilea[i] = quantile(K_save[:,i],0.025)
    K_eval_quantileb[i] = quantile(K_save[:,i],0.5)
    K_eval_quantilec[i] = quantile(K_save[:,i],0.975)
end


# initialise figure
fig_inferreddistributions_r = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_inferreddistributions_r_size)
fig_inferreddistributions_K = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_inferreddistributions_r_size)
# plot the quantilea, quantileb, quantilec
plot!(fig_inferreddistributions_r,r_eval,r_eval_quantileb,ribbon=(r_eval_quantileb-r_eval_quantilea,r_eval_quantilec-r_eval_quantileb),label="95% interval",lw=0,xaxis=:identity,yaxis=:identity,frame=:box)
plot!(fig_inferreddistributions_K,K_eval,K_eval_quantileb,ribbon=(K_eval_quantileb-K_eval_quantilea,K_eval_quantilec-K_eval_quantileb),label="95% interval",lw=0,frame=:box)
# plot the median
plot!(fig_inferreddistributions_r,r_eval,r_eval_quantileb,lw=2,color=:black,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",yticks=[],xticks=fig_inferreddistributions_r_xticks,xlim=fig_inferreddistributions_r_xlim,ylim=[0,ylims(fig_inferreddistributions_r)[2]])
plot!(fig_inferreddistributions_K,K_eval,K_eval_quantileb,lw=2,color=:black,xlab=L"K \ [-]",label="mean",yticks=[],xticks=fig_inferreddistributions_K_xticks,xlim=fig_inferreddistributions_K_xlim,ylim=[0,ylims(fig_inferreddistributions_K)[2]])
# display
display(fig_inferreddistributions_r)
display(fig_inferreddistributions_K)
# save
savefig(fig_inferreddistributions_r,filepath_save * "fig_inferreddistributions_r" * ".pdf")
savefig(fig_inferreddistributions_K,filepath_save * "fig_inferreddistributions_K" * ".pdf")


### For synthetic data add vertical lines representing the known parameter values used to generate the data
if syndataindicator == 1
    # plot the known distributions
    
    # r known
    r_eval_known = pdf.(rdist(θknown[1],θknown[2]),r_eval)
    # K known
    K_eval_known = pdf.(rdist(θknown[3],θknown[4]),K_eval)
    # plot
    plot!(fig_inferreddistributions_r,r_eval,r_eval_known,lw=2,color=:orange,ls=:dash)
    plot!(fig_inferreddistributions_K,K_eval,K_eval_known,lw=2,color=:orange,ls=:dash)
    # display
    display(fig_inferreddistributions_r)
    display(fig_inferreddistributions_K)
    # save
    savefig(fig_inferreddistributions_r,filepath_save * "fig_inferreddistributions_r_synvaluesshown" * ".pdf")
    savefig(fig_inferreddistributions_K,filepath_save * "fig_inferreddistributions_K_synvaluesshown" * ".pdf")
end

# Plot inset for synthetic data
if savename == "Fig_S1ADE_Inference"

    #### Plot inset for r 
    fig_inferreddistributions_r_inset = plot(layout=grid(1,1),titlefont=fnt_inset, guidefont=fnt_inset, tickfont=fnt_inset,legend=false,size = (50*mm_to_pts_scaling,30*mm_to_pts_scaling))
    plot!(fig_inferreddistributions_r_inset,r_eval,r_eval_quantileb,ribbon=(r_eval_quantileb-r_eval_quantilea,r_eval_quantilec-r_eval_quantileb),label="95% interval",lw=0,xaxis=:identity,yaxis=:identity,frame=:box) #xaxis=:log,yaxis=:log)
    plot!(fig_inferreddistributions_r_inset,r_eval,r_eval_quantileb,lw=2,color=:black,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",yticks=[],xticks=[0,0.05e-8,0.1e-8],xlim=[0,0.15e-8],ylim=[0,ylims(fig_inferreddistributions_r)[2]])
    plot!(fig_inferreddistributions_r_inset,r_eval,r_eval_known,lw=2,color=:orange,ls=:dash)
    display(fig_inferreddistributions_r_inset)
    savefig(fig_inferreddistributions_r_inset,filepath_save * "fig_inferreddistributions_r_inset" * ".pdf")

end

# Plot comparison to results obtained using the method of least squares
if lsdataindicator == 1
    ## overlay best-fit from the least squares method

    ### r (if statement to change figure formatting)
     if savename == "Fig_S1ADE_Inference"
            fig_inferreddistributions_r_ls = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size)
            # plot the quantilea, quantileb, quantilec
            plot!(fig_inferreddistributions_r_ls,r_eval,r_eval_quantileb,ribbon=(r_eval_quantileb-r_eval_quantilea,r_eval_quantilec-r_eval_quantileb),label="95% interval",lw=0,xaxis=:identity,yaxis=:identity,frame=:box)
            # plot the mean
            plot!(fig_inferreddistributions_r_ls,r_eval,r_eval_quantileb,lw=2,color=:black,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",yticks=[],xticks=([0.0, 4.0e-8, 8.0e-8], [0, 4, 8]),xlim=[0,10e-8],ylim=[0,ylims(fig_inferreddistributions_r)[2]])
            plot!(fig_inferreddistributions_r_ls,r_eval,r_eval_known,lw=2,color=:orange,ls=:dash)
            vline!(fig_inferreddistributions_r_ls,[ls_xopt[1]],lw=2,ls=:dash,color=:magenta,legend=false)
            display(fig_inferreddistributions_r_ls)
            savefig(fig_inferreddistributions_r_ls,filepath_save * "fig_inferreddistributions_r_withleastsquares" * ".pdf")

    elseif savename == "Fig_R1ADE_Inference"
        fig_inferreddistributions_r_ls = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size)
        # plot the quantilea, quantileb, quantilec
        plot!(fig_inferreddistributions_r_ls,r_eval,r_eval_quantileb,ribbon=(r_eval_quantileb-r_eval_quantilea,r_eval_quantilec-r_eval_quantileb),label="95% interval",lw=0,xaxis=:identity,yaxis=:identity,frame=:box)
        # plot the mean
        plot!(fig_inferreddistributions_r_ls,r_eval,r_eval_quantileb,lw=2,color=:black,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",yticks=[],xticks=([0.0, 4.0e-8, 8.0e-8], [0, 4, 8]),xlim=[0,10e-8],ylim=[0,ylims(fig_inferreddistributions_r)[2]])
        vline!(fig_inferreddistributions_r_ls,[ls_xopt[1]],lw=2,ls=:dash,color=:magenta,legend=false)
        display(fig_inferreddistributions_r_ls)
        savefig(fig_inferreddistributions_r_ls,filepath_save * "fig_inferreddistributions_r_withleastsquares" * ".pdf")

    elseif savename == "Fig_S5ADE_Inference" || savename == "Fig_S7ADE_Inference" || savename == "Fig_S9ADE_Inference"
        fig_inferreddistributions_r_ls = deepcopy(fig_inferreddistributions_r);
        vline!(fig_inferreddistributions_r_ls,[ls_xopt[1]],lw=2,ls=:dash,color=:magenta,legend=false,size=fig_3_across_size)
        display(fig_inferreddistributions_r_ls)
        savefig(fig_inferreddistributions_r_ls,filepath_save * "fig_inferreddistributions_r_withleastsquares" * ".pdf")

    end
    
    #### K
    fig_inferreddistributions_K_ls = deepcopy(fig_inferreddistributions_K);
    vline!(fig_inferreddistributions_K_ls,[ls_xopt[2]],lw=2,ls=:dash,color=:magenta,legend=false,size=fig_3_across_size)
    display(fig_inferreddistributions_K_ls)
    # save
    savefig(fig_inferreddistributions_K_ls,filepath_save * "fig_inferreddistributions_K_withleastsquares" * ".pdf")

end


#################################################
# 10 - Plots  - Inferred predictions for number of associated particles, P(t)
#################################################  

Random.seed!(1)

# define times to evaluate P(t)
t_eval = [0:0.5:24;]*3600;

# define model without fluorescence noise
model_noiseless = θ -> simulate_model_noiseless(θ,θfixed,t_eval,param_dist;N=20_000);
model_noiseless([X_sim_param_rand[1][1]
                                X_sim_param_rand[1][2]
                                X_sim_param_rand[1][3]
                                X_sim_param_rand[1][4]])

# simulate the model without fluorescence noise for the 2000 ABC particles
@time P_save = [model_noiseless([X_sim_param_rand[i][1]
                                X_sim_param_rand[i][2]
                                X_sim_param_rand[i][3]
                                X_sim_param_rand[i][4]]) for i=1:200];

# calculate quantiles for each time point
P_pred_quantilea = zeros(length(t_eval));
P_pred_quantileb = zeros(length(t_eval));
P_pred_quantilec = zeros(length(t_eval));
P_pred_quantilea50 = zeros(length(t_eval));
P_pred_quantilec50 =  zeros(length(t_eval));
P_pred_mean =  zeros(length(t_eval));
@time for j=1:length(t_eval)
        P_save_all_tmp =  reduce(vcat, [P_save[i][j] for i=1:200]);
        #quantiles
        P_pred_quantilea[j] = quantile(P_save_all_tmp,0.025);
        P_pred_quantileb[j] = quantile(P_save_all_tmp,0.5);
        P_pred_quantilec[j] = quantile(P_save_all_tmp,0.975);
        P_pred_quantilea50[j] = quantile(P_save_all_tmp,0.25);
        P_pred_quantilec50[j] = quantile(P_save_all_tmp,0.75);
        P_pred_mean[j] = mean(P_save_all_tmp);
end

# fig_inferred_predictions = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ \mathrm{[hr]}",ylab=L"P(t)",size=fig_inferred_predictions_size)
fig_inferred_predictions = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ \mathrm{[hr]}",size=fig_inferred_predictions_size,frame=:box)
# plot the quantilea, quantileb, quantilec
plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantileb,ribbon=(P_pred_quantileb-P_pred_quantilea,P_pred_quantilec-P_pred_quantileb),label="95% interval",color=:deepskyblue1,legend=false,xticks=fig_inferred_predictions_xticks,yticks=fig_inferred_predictions_yticks,ylim=fig_inferred_predictions_ylim,frame=:box)
plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantileb,ribbon=(P_pred_quantileb-P_pred_quantilea50,P_pred_quantilec50-P_pred_quantileb),color=:darkblue,label="50% interval",legend=false)
display(fig_inferred_predictions)
# save
savefig(fig_inferred_predictions,filepath_save * "fig_inferred_predictions" * ".pdf")

# if synthetic data plot the known distributions of P(t) as boxplots
if syndataindicator == 1
    fig_inferred_predictions_boxplot = deepcopy(fig_inferred_predictions);
    bar_width_val_box =1.0;
    [boxplot!(fig_inferred_predictions_boxplot,[T[i]], data[i][!,:Psyndeter],bar_width=bar_width_val_box,color=:white,outliers=false) for i=1:length(T)]
    display(fig_inferred_predictions_boxplot)
    # save
    savefig(fig_inferred_predictions_boxplot,filepath_save * "fig_inferred_predictions_boxplot" * ".pdf")
end


## if least squares performed overlay least squares best-fit
if lsdataindicator == 1
   if syndataindicator == 1
        fig_inferred_predictions_ls = deepcopy(fig_inferred_predictions_boxplot);
    else
        fig_inferred_predictions_ls = deepcopy(fig_inferred_predictions);
    end
    plot!(fig_inferred_predictions_ls,ls_Tplot,[solve_ode_model_independent(ls_Tplot[i]*3600,ls_xopt,θfixed) for i=1:length(ls_Tplot)],lw=2,color=:magenta,ls=:dash,size=fig_3_across_size)
    scatter!(fig_inferred_predictions_ls,[0;T],[0;median_data_syn],color=:magenta,markersize=6,msw=1,yticks=ls_P_yticks,ylim=ls_P_ylim)
    display(fig_inferred_predictions_ls)
    savefig(fig_inferred_predictions_ls,filepath_save * "fig_inferred_predictions_withleastsquaresbestfit" * ".pdf")
end


#################################################
# 11 - Plot  - Time series - Histogram of fluorescence data vs uncertainity in simulated data from posterior distribution
#################################################  

# particle only 
fig_data_hist2_particleonly = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",legend=false,size=fig_data_hist2_size,frame=:box)
histogram!(fig_data_hist2_particleonly,data_particleonly[:,"638 Red(Peak)"],xlim=fig_data_hist2_particleonly_xlim,ylim=fig_data_hist2_particleonly_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:1000,xticks=fig_data_hist2_particleonly_xticks,yticks=fig_data_hist2_particleonly_yticks,lw=0.5)
display(fig_data_hist2_particleonly)
savefig(fig_data_hist2_particleonly,filepath_save * "fig_data_hist2_particleonly" * ".pdf")

# cell only
fig_data_hist2_cellonly = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",legend=false,size=fig_data_hist2_size,frame=:box)
histogram!(fig_data_hist2_cellonly,pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"],xlim=fig_data_hist2_cellonly_xlim,ylim=fig_data_hist2_cellonly_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:1000,xticks=fig_data_hist2_cellonly_xticks,yticks=fig_data_hist2_cellonly_yticks,lw=0.5)
display(fig_data_hist2_cellonly)
savefig(fig_data_hist2_cellonly,filepath_save * "fig_data_hist2_cellonly" * ".pdf")

# particle only with median
fig_data_hist2_particleonly_med = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",legend=false,size=fig_data_hist2_size,frame=:box)
histogram!(fig_data_hist2_particleonly_med,data_particleonly[:,"638 Red(Peak)"],xlim=fig_data_hist2_particleonly_xlim,ylim=fig_data_hist2_particleonly_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:1000,xticks=fig_data_hist2_particleonly_xticks,yticks=fig_data_hist2_particleonly_yticks,lw=0.5)
vline!(fig_data_hist2_particleonly_med,[median(data_particleonly[:,"638 Red(Peak)"])],color=:orange,lw=2,ls=:dash,legend=false)
display(fig_data_hist2_particleonly_med)
savefig(fig_data_hist2_particleonly_med,filepath_save * "fig_data_hist2_particleonly_med" * ".pdf")

# cell only with median
fig_data_hist2_cellonly_med = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",legend=false,size=fig_data_hist2_size,frame=:box)
histogram!(fig_data_hist2_cellonly_med,pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"],xlim=fig_data_hist2_cellonly_xlim,ylim=fig_data_hist2_cellonly_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:1000,xticks=fig_data_hist2_cellonly_xticks,yticks=fig_data_hist2_cellonly_yticks,lw=0.5)
vline!(fig_data_hist2_cellonly_med,[median(pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"])],color=:orange,lw=2,ls=:dash,legend=false)
display(fig_data_hist2_cellonly_med)
savefig(fig_data_hist2_cellonly_med,filepath_save * "fig_data_hist2_cellonly_med" * ".pdf")


### timepoints - NO median, NO prediction intervals
@time for i=1:lendatafiles
    fig_data_hist2= plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",xlim=(0,1000),size=fig_data_hist2_size,frame=:box)
    data_this_loop = X[i]; 
    histogram!(fig_data_hist2,data_this_loop,xlim=fig_data_hist2_xlim,ylim=fig_data_hist2_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:(round(maximum(data_this_loop),digits=-3)),xticks=fig_data_hist2_xticks,yticks=fig_data_hist2_yticks,legend=false,lw=0.5)
    display(fig_data_hist2)
    savefig(fig_data_hist2,filepath_save * "fig_data_hist2_nopred" * string(i) * ".pdf")
end


###  timepoints - WITH median, NO prediction intervals
@time for i=1:lendatafiles
    fig_data_hist2= plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",xlim=(0,1000),size=fig_data_hist2_size,frame=:box)
    data_this_loop = X[i];
    histogram!(fig_data_hist2,data_this_loop,xlim=fig_data_hist2_xlim,ylim=fig_data_hist2_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:(round(maximum(data_this_loop),digits=-3)),xticks=fig_data_hist2_xticks,yticks=fig_data_hist2_yticks,legend=false,lw=0.5)
    vline!(fig_data_hist2,[median(data_this_loop)],color=:orange,lw=3,ls=:dash,legend=false)
    display(fig_data_hist2)
    savefig(fig_data_hist2,filepath_save * "fig_data_hist2_nopred_withmedian" * string(i) * ".pdf")
end


### timepoints - NO median, WITH prediction intervals
@time for i=1:lendatafiles
    fig_data_hist2= plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",xlim=(0,1000),size=fig_data_hist2_size,frame=:box)
    data_this_loop = X[i];
    histogram!(fig_data_hist2,data_this_loop,xlim=fig_data_hist2_xlim,ylim=fig_data_hist2_ylim,label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:(round(maximum(data_this_loop),digits=-3)),xticks=fig_data_hist2_xticks,yticks=fig_data_hist2_yticks,lw=0.5)

    # compute prediction intervals
    bins_scatter= [0:20:1000;];
    bins_scatter_mid = (bins_scatter[1:end-1]+bins_scatter[2:end])./2;
    bins_scatter_height = zeros(nrandsampleshist,length(bins_scatter_mid));
    bins_scatter_total_area = (20*length(X_sim_mean[i][1]))
    for k=1:nrandsampleshist # for each data set
        for j=1:length(bins_scatter_mid) 
        bins_scatter_height[k,j] = sum(bins_scatter[j] .<= X_sim_mean[k][i] .< bins_scatter[j+1])./bins_scatter_total_area;
        end
    end
    bins_scatter_height_mean = [quantile(bins_scatter_height[:,j],0.500) for j=1:length(bins_scatter_mid)];
    bins_scatter_height_q025 = [quantile(bins_scatter_height[:,j],0.025) for j=1:length(bins_scatter_mid)];
    bins_scatter_height_q975 = [quantile(bins_scatter_height[:,j],0.975) for j=1:length(bins_scatter_mid)];
    plot!(fig_data_hist2,bins_scatter_mid,bins_scatter_height_mean,ribbon=(bins_scatter_height_q975-bins_scatter_height_mean,bins_scatter_height_mean-bins_scatter_height_q025),color=:deepskyblue,label="95% interval",lw=1,legend=false)
    display(fig_data_hist2)   
    savefig(fig_data_hist2,filepath_save * "fig_data_hist2_" * string(i) * ".pdf")

    # PLOT a zoomed in plot for S1 (for an inset)
    if savename == "Fig_S1ADE_Inference"
        fig_data_hist2_inset= plot(layout=grid(1,1),titlefont=fnt_inset, guidefont=fnt_inset, tickfont=fnt_inset,xlab=L"S \ \mathrm{[AU]}",xlim=(70,130),size = (40*mm_to_pts_scaling,30*mm_to_pts_scaling),frame=:box)
        histogram!(fig_data_hist2_inset,data_this_loop,ylim=[0.003,0.006],label="Raw",normalize=:pdf,color=:lightgray, bins = 0:20:(round(maximum(data_this_loop),digits=-3)),xticks=[80,100,120],yticks=[0.004,0.006],lw=0.5)
        plot!(fig_data_hist2_inset,bins_scatter_mid,bins_scatter_height_mean,ribbon=(bins_scatter_height_q975-bins_scatter_height_mean,bins_scatter_height_mean-bins_scatter_height_q025),color=:deepskyblue,label="95% interval",lw=1,legend=false)
        display(fig_data_hist2_inset)
        savefig(fig_data_hist2_inset,filepath_save * "fig_data_hist2_inset"* string(i) * ".pdf")
    end

end
     

#################################################
# 12 - Plot - percentage of cells with P(t) > 0.99K
#################################################  

# simulate the model without noise due to fluorescence and record r and K values
function simulate_model_noiseless_withK(θ::Vector,θfixed::Vector,T::Vector,d::Function;N::Int=1000)
    # Parameters
    ξᵢ = rand(d(θ[1:4]),N) # Sample from parameter distribution
    rᵢ = @view ξᵢ[1,:]
    Kᵢ = @view ξᵢ[2,:]
    # solve the hierarchical ODE model for P(t)
    P = solve_ode_model(T,[rᵢ,Kᵢ],θfixed);
    return P,Kᵢ,rᵢ
end
model_noiseless_withK = θ -> simulate_model_noiseless_withK(θ,θfixed,T_K,param_dist;N=N_percentg99Ksims);

# specify the times to solve
T_K = [0:0.5:24;]*3600;

# for each sample of the statistical hyperparameters, sample r and K N_percentg99Ksims times, simulate P(t) and K, compare P(t) and K
N_percentg99Ksims =20_000;
percentg50Kcount = zeros(length(T_K)); # 
percentg90Kcount = zeros(length(T_K)); # 
percentg95Kcount = zeros(length(T_K)); # 
percentg99Kcount = zeros(length(T_K)); # 
@time for i=1:nrandsampleshist
    Random.seed!(i)    
    modeleval = model_noiseless_withK(X_sim_param_rand[i]);
    for k=1:length(T_K)
        # for each time step simulate the mathematical model and compare to K
        percentg50Kcount[k] = percentg50Kcount[k] + sum(modeleval[1][k] .> .50*modeleval[2]);
        percentg90Kcount[k] = percentg90Kcount[k] + sum(modeleval[1][k] .> .90*modeleval[2]);
        percentg95Kcount[k] = percentg95Kcount[k] + sum(modeleval[1][k] .> .95*modeleval[2]);
        percentg99Kcount[k] = percentg99Kcount[k] + sum(modeleval[1][k] .> .99*modeleval[2]);
    end
end
# convert to percentages
percentg50K = 100*percentg50Kcount./(nrandsampleshist * N_percentg99Ksims);
percentg90K = 100*percentg90Kcount./(nrandsampleshist * N_percentg99Ksims);
percentg95K = 100*percentg95Kcount./(nrandsampleshist * N_percentg99Ksims);
percentg99K = 100*percentg99Kcount./(nrandsampleshist * N_percentg99Ksims);


## Plot results 
fig_percentg99K = plot(layout=grid(1,1),legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ [\mathrm{hr}]",ylab=L"P(t) > \beta K \ \mathrm{[\%]}",ylim=[0,100],xticks=[0,12,24],yticks=[0,50,100],size=fig_percentg99K_size)
plot!(fig_percentg99K,T_K/3600,percentg50K,lw=2,color=:black)
plot!(fig_percentg99K,T_K/3600,percentg95K,lw=2,ls=:dash,color=:black)
plot!(fig_percentg99K,T_K/3600,percentg99K,lw=2,ls=:dot,color=:black)
display(fig_percentg99K)
savefig(fig_percentg99K,filepath_save * "fig_percentg99K" * ".pdf")
