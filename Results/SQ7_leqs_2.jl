#=
    SQ7_lesq - Synthetic data with intermediate r (relative to K) - HOMOGENEOUS r,K - SYNTHETIC CONTROL DATA
    Method of least squares using median of flow data
=#

# 1 - Load packages and modules
# 2 - Plot options
# 3 - Setup location to save files
# 4 - Setup problem
# 5 - Plot histograms of the flow data
# 6 - Compute median
# 7 - Plot data median
# 8 - Method of least squares fit
# 9 - Plot data median with method of least squares fit
# 10 - Save Outputs

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using CSV, DataFrames, DataFramesMeta
using Distributions
using JLD2
using Plots
using StatsBase
using StatsPlots
using Random
using LaTeXStrings
using NLopt

#########################################
# 2 - Plot options
#########################################

pyplot() # plot options
fnt = Plots.font("sans-serif", 12); # plot options
global cur_colors = palette(:default); # plot options

# Plot sizes
mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

#########################################
# 3 - Setup location to save files
#########################################

savename = "Fig_SQ7AD_lesq";
filepathseparator = "/";
filepath_save = pwd() * filepathseparator * "Figures" * filepathseparator * savename * filepathseparator; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 4 - Setup problem
##############################################################

include("SQ7AD_Setup.jl")

##############################################################
## 5 - Plot histograms of the flow data
############################################################## 

# particle only data
fig_data_particleonly = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S",legend=false)
histogram!(fig_data_particleonly,data_particleonly[:,"638 Red(Peak)"],xlim=[0,1000],ylim=[0,0.02],label="Raw",normalize=:pdf,color=:gray80, bins = 0:20:1000)
vline!(fig_data_particleonly,[median(data_particleonly[:,"638 Red(Peak)"])],label="Median",color=:red,lw=5,xticks=[0,500,1000],yticks=[0,0.01,0.02],ls=:dash)
display(fig_data_particleonly)
savefig(fig_data_particleonly,filepath_save * "fig_data_particleonly" * ".pdf")

# cell only data
fig_data_cellonly = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S",legend=false)
histogram!(fig_data_cellonly,pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"],xlim=[0,1000],ylim=[0,0.01],label="Raw",normalize=:pdf,color=:gray80, bins = 0:20:1000)
vline!(fig_data_cellonly,[median(pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"])],label="Median",color=:red,lw=5,xticks=[0,500,1000],yticks=[0,0.01],ls=:dash)
display(fig_data_cellonly)
savefig(fig_data_cellonly,filepath_save * "fig_data_cellonly" * ".pdf")

# particle-cell interaction time points
for i=1:lendatafiles
    fig_data = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S",legend=false)
    histogram!(fig_data,pmt_correctionfactor(voltage_data[i],voltage_particleonly).*data[i][:,"638 Red(Peak)"],xlim=[0,1000],ylim=[0,0.01],label="Raw",normalize=:pdf,color=:gray80, bins = 0:20:1000)
    vline!(fig_data,[median(pmt_correctionfactor(voltage_data[i],voltage_particleonly).*data[i][:,"638 Red(Peak)"])],label="Median",color=:red,lw=5,xticks=[0,500,1000],yticks=[0,0.01],ls=:dash)
    display(fig_data)
    savefig(fig_data,filepath_save * "fig_data" * string(i) * ".pdf")
end


##############################################################
## 6 - Compute median
##############################################################

medianp=median(data_particleonly[:,"638 Red(Peak)"]);
medianc=median(pmt_correctionfactor(voltage_cellonly,voltage_particleonly).*data_cellonly[:,"638 Red(Peak)"]);
mediant1=median(pmt_correctionfactor(voltage_data[1],voltage_particleonly).*data[1][:,"638 Red(Peak)"]);
mediant2=median(pmt_correctionfactor(voltage_data[2],voltage_particleonly).*data[2][:,"638 Red(Peak)"]);
mediant4=median(pmt_correctionfactor(voltage_data[3],voltage_particleonly).*data[3][:,"638 Red(Peak)"]);
mediant8=median(pmt_correctionfactor(voltage_data[4],voltage_particleonly).*data[4][:,"638 Red(Peak)"]);
mediant16=median(pmt_correctionfactor(voltage_data[5],voltage_particleonly).*data[5][:,"638 Red(Peak)"]);
mediant24=median(pmt_correctionfactor(voltage_data[6],voltage_particleonly).*data[6][:,"638 Red(Peak)"]);


P_t0 = (medianc .- medianc) ./ medianp;
P_t1 = (mediant1 .- medianc) ./ medianp;
P_t2 = (mediant2 .- medianc) ./ medianp;
P_t4 = (mediant4 .- medianc) ./ medianp;
P_t8 = (mediant8 .- medianc) ./ medianp;
P_t16 = (mediant16 .- medianc) ./ medianp;
P_t24 = (mediant24 .- medianc) ./ medianp;

data_syn = [[P_t0]; [P_t1]; [P_t2]; [P_t4]; [P_t8]; [P_t16]; [P_t24]];

##############################################################
## 7 - Plot data median
##############################################################

median_data_syn = [median(data_syn[i+1]) for i=1:length(Traw)]
fig_raw_median = plot(layout=grid(1,1),legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ [\mathrm{hours}]",ylab=L"P(t)")
scatter!(fig_raw_median,[0;Traw],[0;median_data_syn],color=:black,markersize=10,xticks=[0,12,24],yticks=[0,20,40],ylims=(-0.5,40.5))
display(fig_raw_median)
savefig(fig_raw_median,filepath_save * "fig_raw_median" * ".pdf")

##############################################################
## 8 - Method of least squares fit
##############################################################

# Data # times v median_data_syn

function error(median_data_syn,a)
    y=[solve_ode_model_independent(T[i]*3600,a,θfixed) for i=1:length(T)];
    e=0;
    e+=-sum((y .- median_data_syn).^2) # set negative as maximising
    return e
end

function fun(a)
    return error(median_data_syn,a)
end

function optimise(fun,θ₀,lb,ub) 
    tomax = (θ,∂θ) -> fun(θ)
    opt = Opt(:LN_NELDERMEAD,length(θ₀))
    opt.max_objective = tomax
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    opt.maxtime = 30.0; # maximum time in seconds
    res = optimize(opt,θ₀)
    return res[[2,1]]
end

# MLE
r1 = 10e-8; 
K1 = 16.0;
θG = [r1,K1]; # first guess

# initial guess plot
fig_raw_median_initialguess = plot(layout=grid(1,1),legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ [\mathrm{hours}]",ylab=L"P(t)")
Tplot = LinRange(0,25,251);
scatter!(fig_raw_median_initialguess,[0;Traw],[0;median_data_syn],color=:black,markersize=10)
plot!(fig_raw_median_initialguess,Tplot,[solve_ode_model_independent(Tplot[i]*3600,θG,θfixed) for i=1:length(Tplot)],color=:black,lw=2)
display(fig_raw_median_initialguess)

# lower and upper bounds for parameter estimation
r_lb = 1e-14
r_ub = 1e-5
K_lb = 0.0
K_ub = 100.0

lb=[r_lb,K_lb];
ub=[r_ub,K_ub];

# MLE optimisation
(xopt,fopt)  = optimise(fun,θG,lb,ub)

# storing MLE 
global fmle=fopt
global r1mle=xopt[1]
global K1mle=xopt[2]

##############################################################
## 9 - Plot data median with method of least squares fit
##############################################################

fig_raw_median_leastsquares = plot(layout=grid(1,1),legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ [\mathrm{hr}]",ylab=L"P(t)")
Tplot = LinRange(0,25,251);
plot!(fig_raw_median_leastsquares,Tplot,[solve_ode_model_independent(Tplot[i]*3600,xopt,θfixed) for i=1:length(Tplot)],color=:red,lw=3,xticks=[0,12,24],ylims=(-0.5,50.0),ls=:dash)
scatter!(fig_raw_median_leastsquares,[0;Traw],[0;median_data_syn],color=:black,markersize=15)
display(fig_raw_median_leastsquares)
savefig(fig_raw_median_leastsquares,filepath_save * "fig_raw_median_leastsquares" * ".pdf")


##############################################################
## 10 - Plot data median with method of least squares fit + KNOWN P(t)
##############################################################

# if synthetic data plot the known distributions of P(t) as boxplots

fig_inferred_predictions_ylim= [0,3];
fig_inferred_predictions_yticks = [0,2];

fig_raw_median_leastsquares_knownP = plot(layout=grid(1,1),legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ [\mathrm{hr}]",frame=:box,size=fig_3_across_size) #,ylab=L"P(t)"
Tplot = LinRange(0,25,251);
plot!(fig_raw_median_leastsquares_knownP,Tplot,[solve_ode_model_independent(Tplot[i]*3600,xopt,θfixed) for i=1:length(Tplot)],color=:magenta,lw=3,xticks=[0,12,24],ylims=fig_inferred_predictions_ylim,yticks=fig_inferred_predictions_yticks,ls=:dash)
scatter!(fig_raw_median_leastsquares_knownP,[0;Traw],[0;median_data_syn],color=:magenta,markersize=10)
for i=1:length(T)
    scatter!(fig_raw_median_leastsquares_knownP,[T[i]], [median(data[i][!,:Psyndeter])],color=:black,markershape=:xcross,markersize=8,msw=0)
end
scatter!(fig_raw_median_leastsquares_knownP,[0.0], [0.0],color=:black,markershape=:xcross,markersize=8,msw=0)
display(fig_raw_median_leastsquares_knownP)
# save
savefig(fig_raw_median_leastsquares_knownP,filepath_save * "fig_raw_median_leastsquares_knownP" * ".pdf")



##############################################################
## 11 - Save Outputs
##############################################################

@save "Results/Files_syn/Lesq_SQ7.jld2" Tplot xopt θfixed median_data_syn