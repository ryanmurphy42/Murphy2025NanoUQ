
### COMPARING ABC-SMC RESULTS WITH DIFFERENT ABC DISTANCES

# 1 - Load packages and modules
# 2 - Default plot settings (plot sizes, axis limits, ticks)
# Loop through the different synthetic and experimental data sets
    # 4 - Load ABC-SMC results for the three different distances
    # 5 - Create folder to save results
    # 6 - Plots - Univariate posterior density
    # 7 - Plots  - Inferred distributions for r and K  (95% prediction, mean, mode)
    # 8 - Plots  - Inferred predictions for number of associated particles, P(t)
    # 9 - Plots  - Time series - Histogram of fluorescence data vs uncertainity in simulated data from posterior distribution
    # 10 - Plots - Comparison of ABC distances

#########################################
# 1 - Load packages and modules
#########################################

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
using LinearAlgebra
using Distributions
using HypothesisTests

#########################################
# 2 - Default plot settings (plot sizes, axis limits, ticks)
#########################################

pyplot() # plot options
fnt = Plots.font("sans-serif", 10); # plot options
global cur_colors = palette(:default); # plot options

col_vec= [:blue,:magenta,:green2]
col_vec_shading = [:blue,:magenta,:green2]
ls_vec = [:solid,:dash,:dashdot]

# default figure sizes
mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

#########################################
# Loop through the different synthetic and experimental data sets
#########################################

for comparison_id = [1,2,3,101,102,103,104]

    #########################################
    # 4 - Load ABC-SMC results for the three different distances
    #########################################

    #### Comparison 1 - Real R1
    if comparison_id == 1;
        # Dataset 1
        @load "Results/Files_real/R1AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_real/R1CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_real/R1KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);

        P_yticks = [0,50,100]
        P_ylim = [0,100]
        r_xlim = [0,1e-8]
        sim_id = "R1"

      
    #### Comparison 2 - Real R5
    elseif comparison_id == 2;
        # Dataset 1
        @load "Results/Files_real/R5AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_real/R5CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_real/R5KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);

        P_yticks = [0,15,30]
        P_ylim = [0,30]
        r_xlim = [0,1e-12]
        sim_id = "R5"

    #### Comparison 3 - Real R6
    elseif comparison_id == 3;
        # Dataset 1
        @load "Results/Files_real/R6AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_real/R6CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_real/R6KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);

        P_yticks = [0,50,100]
        P_ylim = [0,100]
        r_xlim = [0,1e-10]
        sim_id = "R6"

    #### Comparison 101 - Syn S5
    elseif comparison_id == 101;
        # Dataset 1
        @load "Results/Files_syn/S5AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_syn/S5CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_syn/S5KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);

        P_yticks = [0,10,20]
        P_ylim = [0,20]
        r_xlim = [0,1e-6]
        sim_id = "S5"

    #### Comparison 102 - Syn S7
    elseif comparison_id == 102;
        # Dataset 1
        @load "Results/Files_syn/S7AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_syn/S7CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_syn/S7KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);

        
        P_yticks = [0,10,20]
        P_ylim = [0,20]
        r_xlim = [0,1e-8]
        sim_id = "S7"

    #### Comparison 103 - Syn S9
    elseif comparison_id == 103;
        # Dataset 1
        @load "Results/Files_syn/S9AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_syn/S9CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_syn/S9KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);
        
        P_yticks = [0,10,20]
        P_ylim = [0,20]
        r_xlim = [0,1e-3]
        sim_id = "S9"

    #### Comparison 104 - Syn S1
    elseif comparison_id == 104;
        # Dataset 1
        @load "Results/Files_syn/S1AD_ABCE_inference.jld2" P p_accept ε_seq 
        P_data1 = deepcopy(P);
        # Dataset 2
        @load "Results/Files_syn/S1CV_ABCE_inference.jld2" P p_accept ε_seq 
        P_data2 = deepcopy(P);
        # Dataset 3
        @load "Results/Files_syn/S1KS_ABCE_inference.jld2" P p_accept ε_seq 
        P_data3 = deepcopy(P);
        
        P_yticks = [0,50,100]
        P_ylim = [0,100]
        r_xlim = [0,5e-8]

        sim_id = "S1"
           
    end

    println(string("-----------------") * sim_id * string("-----------------"))

    # Load relevant setup file
    include(pwd() * "/Results/" * string(sim_id[1:2]) * "AD_Setup.jl")

    #########################################
    # 5 - Create folder to save results
    #########################################

    savename = "Fig_comparing_ABCdistances_SYN_" * string(sim_id)
    filepathseparator = "/";
    filepath_save = pwd() * filepathseparator * "Figures" * filepathseparator * savename * filepathseparator; # location to save figures 
    isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

    #########################################
    # 6 - Plots - Univariate posterior density
    #########################################

    fig_density_μr = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_3_across_size,framestyle=:box);
    fig_density_σr = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_3_across_size,framestyle=:box);
    fig_density_μK = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_3_across_size,framestyle=:box);
    fig_density_σK = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_3_across_size,framestyle=:box);

    # data 1
    density!(fig_density_μr,[P_data1[i].θ[1] for i=1:length(P_data1)],lw=3,legend=false,yticks=[],xlab=L"m_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[1])
    density!(fig_density_σr,[P_data1[i].θ[2] for i=1:length(P_data1)],lw=3,legend=false,yticks=[],xlab=L"s_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[1])
    density!(fig_density_μK,[P_data1[i].θ[3] for i=1:length(P_data1)],lw=3,legend=false,yticks=[],xlab=L"m_{K} \ [-]",color=col_vec[1])
    density!(fig_density_σK,[P_data1[i].θ[4] for i=1:length(P_data1)],lw=3,legend=false,yticks=[],xlab=L"s_{K} \ [-]",color=col_vec[1])

    # # data 2
    density!(fig_density_μr,[P_data2[i].θ[1] for i=1:length(P_data2)],lw=3,legend=false,yticks=[],xlab=L"m_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[2],ls=:dash)
    density!(fig_density_σr,[P_data2[i].θ[2] for i=1:length(P_data2)],lw=3,legend=false,yticks=[],xlab=L"s_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[2],ls=:dash)
    density!(fig_density_μK,[P_data2[i].θ[3] for i=1:length(P_data2)],lw=3,legend=false,yticks=[],xlab=L"m_{K} \ [-]",color=col_vec[2],ls=:dash)
    density!(fig_density_σK,[P_data2[i].θ[4] for i=1:length(P_data2)],lw=3,legend=false,yticks=[],xlab=L"s_{K} \ [-]",color=col_vec[2],ls=:dash)

    # # data 3
    density!(fig_density_μr,[P_data3[i].θ[1] for i=1:length(P_data3)],lw=3,legend=false,yticks=[],xlab=L"m_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[3],ls=:dashdot)
    density!(fig_density_σr,[P_data3[i].θ[2] for i=1:length(P_data3)],lw=3,legend=false,yticks=[],xlab=L"s_{r} \ [\mathrm{ms}^{-1}]",color=col_vec[3],ls=:dashdot)
    density!(fig_density_μK,[P_data3[i].θ[3] for i=1:length(P_data3)],lw=3,legend=false,yticks=[],xlab=L"m_{K} \ [-]",color=col_vec[3],ls=:dashdot)
    density!(fig_density_σK,[P_data3[i].θ[4] for i=1:length(P_data3)],lw=3,legend=false,yticks=[],xlab=L"s_{K} \ [-]",color=col_vec[3],ls=:dashdot)

    ### display and save
    display(fig_density_μr)
    display(fig_density_σr)
    display(fig_density_μK)
    display(fig_density_σK)

    savefig(fig_density_μr,filepath_save * "fig_density_mr_comparison_" * string(comparison_id) * ".pdf")
    savefig(fig_density_σr,filepath_save * "fig_density_sr_comparison_" * string(comparison_id) * ".pdf")
    savefig(fig_density_μK,filepath_save * "fig_density_mK_comparison_" * string(comparison_id) * ".pdf")
    savefig(fig_density_σK,filepath_save * "fig_density_sK_comparison_" * string(comparison_id) * ".pdf")

    savefig(fig_density_μr,filepath_save * "fig_density_mr_comparison_" * string(comparison_id) * ".png")
    savefig(fig_density_σr,filepath_save * "fig_density_sr_comparison_" * string(comparison_id) * ".png")
    savefig(fig_density_μK,filepath_save * "fig_density_mK_comparison_" * string(comparison_id) * ".png")
    savefig(fig_density_σK,filepath_save * "fig_density_sK_comparison_" * string(comparison_id) * ".png")


    #################################################
    # 7 - Plots  - Inferred distributions for r and K  (95% prediction, mean, mode)
    #################################################  

    nrandsampleshist = 2000;

    rdist(μr,σr)  = LogNormal(log(μr^2 / sqrt(σr^2 + μr^2)),sqrt(log(σr^2/μr^2 + 1)));
    Kdist(μK,σK) = LogNormal(log(μK^2 / sqrt(σK^2 + μK^2)),sqrt(log(σK^2/μK^2 + 1)));

    fig_inferreddistributions_r = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size,framestyle=:box)
    fig_inferreddistributions_K = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size,framestyle=:box)

    fig_inferreddistributions_r_ribbon = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size,framestyle=:box)
    fig_inferreddistributions_K_ribbon = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,legend=false,size=fig_3_across_size,framestyle=:box)

    for i=1:3
        if i==1
            X_sim_param_rand = [P_data1[i].θ for i=1:nrandsampleshist];
        elseif i==2
            X_sim_param_rand = [P_data2[i].θ for i=1:nrandsampleshist];
        elseif i==3
            X_sim_param_rand = [P_data3[i].θ for i=1:nrandsampleshist];
        end

        ABCmean = [mean([X_sim_param_rand[i][j] for i=1:nrandsampleshist]) for j=1:4];

        rdistquantile_min = 1e-15; 
        Kdistquantile_min = 0;
        
        rdistquantile_max = 0.01;
        Kdistquantile_max = 50.0;
        
        r_eval = exp.(LinRange(log(rdistquantile_min),log(rdistquantile_max),501));
        K_eval = LinRange(Kdistquantile_min,Kdistquantile_max,501);
        
        r_save = zeros(nrandsampleshist,length(r_eval));
        K_save = zeros(nrandsampleshist,length(K_eval));

        for i=1:nrandsampleshist
            r_save[i,:] = pdf.(rdist(X_sim_param_rand[i][1],X_sim_param_rand[i][2]),r_eval)
            K_save[i,:] = pdf.(Kdist(X_sim_param_rand[i][3],X_sim_param_rand[i][4]),K_eval)
        end

        ## r quantiles
        r_eval_quantilea = zeros(length(r_eval));
        r_eval_quantileb = zeros(length(r_eval));
        r_eval_quantilec = zeros(length(r_eval));

        for i=1:length(r_eval)
            r_eval_quantilea[i] = quantile(r_save[:,i],0.025)
            r_eval_quantileb[i] = quantile(r_save[:,i],0.5)
            r_eval_quantilec[i] = quantile(r_save[:,i],0.975)
        end

        ## K quantiles
        K_eval_quantilea = zeros(length(K_eval));
        K_eval_quantileb = zeros(length(K_eval));
        K_eval_quantilec = zeros(length(K_eval));

        for i=1:length(r_eval)
            K_eval_quantilea[i] = quantile(K_save[:,i],0.025)
            K_eval_quantileb[i] = quantile(K_save[:,i],0.5)
            K_eval_quantilec[i] = quantile(K_save[:,i],0.975)
        end

        # r mean
        r_eval_mean = pdf.(rdist(ABCmean[1],ABCmean[2]),r_eval)
        # K mean
        K_eval_mean = pdf.(Kdist(ABCmean[3],ABCmean[4]),K_eval)


        # plot the quantilea, quantileb, quantilec
        plot!(fig_inferreddistributions_r_ribbon,r_eval,r_eval_quantileb,ribbon=(r_eval_quantileb-r_eval_quantilea,r_eval_quantilec-r_eval_quantileb),label="95% interval",lw=0,color=col_vec[i])
        plot!(fig_inferreddistributions_K_ribbon,K_eval,K_eval_quantileb,ribbon=(K_eval_quantileb-K_eval_quantilea,K_eval_quantilec-K_eval_quantileb),label="95% interval",lw=0,color=col_vec[i])

        # plot the mean
        plot!(fig_inferreddistributions_r_ribbon,r_eval,r_eval_mean,lw=2,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",color=col_vec[i],xlim=r_xlim)
        plot!(fig_inferreddistributions_K_ribbon,K_eval,K_eval_mean,lw=2,xlab=L"K \ [-]",label="mean",color=col_vec[i])

        plot!(fig_inferreddistributions_r,r_eval,r_eval_mean,lw=2,xlab=L"r \ [\mathrm{m \ s}^{-1}]",label="mean",color=col_vec[i],xlim=r_xlim,yticks=[],ls=ls_vec[i])
        plot!(fig_inferreddistributions_K,K_eval,K_eval_mean,lw=2,xlab=L"K \ [-]",label="mean",color=col_vec[i],yticks=[],ls=ls_vec[i])

    end

    # display
    display(fig_inferreddistributions_r)
    display(fig_inferreddistributions_K)
    # save
    savefig(fig_inferreddistributions_r,filepath_save * "fig_inferreddistributions_r" * string(comparison_id) * ".pdf")
    savefig(fig_inferreddistributions_K,filepath_save * "fig_inferreddistributions_K" * string(comparison_id) * ".pdf")

    savefig(fig_inferreddistributions_r,filepath_save * "fig_inferreddistributions_r" * string(comparison_id) * ".png")
    savefig(fig_inferreddistributions_K,filepath_save * "fig_inferreddistributions_K" * string(comparison_id) * ".png")

    # display
    display(fig_inferreddistributions_r_ribbon)
    display(fig_inferreddistributions_K_ribbon)
    # save
    savefig(fig_inferreddistributions_r_ribbon,filepath_save * "fig_inferreddistributions_r_ribbon" * string(comparison_id) * ".pdf")
    savefig(fig_inferreddistributions_K_ribbon,filepath_save * "fig_inferreddistributions_K_ribbon" * string(comparison_id) * ".pdf")

    savefig(fig_inferreddistributions_r_ribbon,filepath_save * "fig_inferreddistributions_r_ribbon" * string(comparison_id) * ".png")
    savefig(fig_inferreddistributions_K_ribbon,filepath_save * "fig_inferreddistributions_K_ribbon" * string(comparison_id) * ".png")

    
    #################################################
    # 8 - Plots  - Inferred predictions for number of associated particles, P(t)
    #################################################  

    t_eval = [0:0.5:24;]*3600;
    model_noiseless = θ -> simulate_model_noiseless(θ,θfixed,t_eval,param_dist;N=20_000);
    Random.seed!(1)
    Nsamples_for_pred = 2000;    
    
    fig_inferred_predictions = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ \mathrm{[hr]}",ylab=L"P(t)",size=fig_3_across_size,framestyle=:box)

    for i=1:3
        # load data
        if i == 1
            @time P_save = [model_noiseless(P_data1[i].θ) for i=1:Nsamples_for_pred];
        elseif i == 2
            @time P_save = [model_noiseless(P_data2[i].θ) for i=1:Nsamples_for_pred];
        elseif i == 3
            @time P_save = [model_noiseless(P_data3[i].θ) for i=1:Nsamples_for_pred];
        end

        # calculate quantiles for each time point
        P_save_all = zeros(Nsamples_for_pred*length(P_save[1][1]),length(t_eval))
        @time for j=1:length(t_eval)
            for i=1:Nsamples_for_pred
                P_save_all[:,j] =  reduce(vcat, [P_save[i][j] for i=1:Nsamples_for_pred]);
            end
        end

        # quantiles
        P_pred_quantilea = [quantile(P_save_all[:,j],0.025) for j=1:length(t_eval)];
        P_pred_quantileb = [quantile(P_save_all[:,j],0.5) for j=1:length(t_eval)];
        P_pred_quantilec = [quantile(P_save_all[:,j],0.975) for j=1:length(t_eval)];
        P_pred_quantilea50 = [quantile(P_save_all[:,j],0.25) for j=1:length(t_eval)];
        P_pred_quantilec50 = [quantile(P_save_all[:,j],0.75) for j=1:length(t_eval)];

        # plot the quantilea, quantileb, quantilec
        plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantilea,label="95% interval",color=col_vec[i],ls=:dot,lw=2)
        plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantileb,label="95% interval",color=col_vec[i],lw=2)
        plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantilec,label="95% interval",color=col_vec[i],ls=:dot,lw=2)
        plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantilea50,color=col_vec[i],ls=:dash,lw=2)
        plot!(fig_inferred_predictions,subplot=1,t_eval/3600,P_pred_quantilec50,color=col_vec[i],legend=false,xticks=[0,12,24],yticks=P_yticks,ylim=P_ylim,ls=:dash,lw=2)

    end

    # display
    display(fig_inferred_predictions)

    # save
    savefig(fig_inferred_predictions,filepath_save * "fig_inferred_predictions_comparison_" * string(comparison_id) * ".pdf")
    savefig(fig_inferred_predictions,filepath_save * "fig_inferred_predictions_comparison_" * string(comparison_id) * ".png")


    #################################################
    # 9 - Plot  - Time series - Histogram of fluorescence data vs uncertainity in simulated data from posterior distribution
    #################################################  

    nrandsampleshist = 2000;

    # plot properties - Histograms of fluorescence data - time course
    fig_data_hist2_size=fig_3_across_size;
    fig_data_hist2_xlim = [0,1000]
    fig_data_hist2_xticks = [0,500,1000]
    fig_data_hist2_ylim = [0,0.006]
    fig_data_hist2_yticks = [0,0.005]

    fig_data_hist2 = Vector{Any}(undef, 6)  # Preallocate an array for 6 plots

    # generate random samples to propogate forward

    for ii=1:3
        println(ii)

        if ii==1
            include(pwd() * "/Results/" * string(sim_id) * "AD_Setup.jl")
            X_sim_param_rand = [P_data1[i].θ for i=1:nrandsampleshist];
        elseif ii==2
            include(pwd() * "/Results/" * string(sim_id) * "CV_Setup.jl")
            X_sim_param_rand = [P_data2[i].θ for i=1:nrandsampleshist];
        elseif ii==3
            include(pwd() * "/Results/" * string(sim_id) * "KS_Setup.jl")
            X_sim_param_rand = [P_data3[i].θ for i=1:nrandsampleshist];
        end

        X_sim_mean = Vector{Vector{Vector{Float64}}}(undef, nrandsampleshist);
        @time for i=1:nrandsampleshist
            Random.seed!(ii)
            X_sim_mean[i] =  model(X_sim_param_rand[i]);
        end

        ### histogram with mean prediction
        if ii == 1
            # plot histograms
            for i=1:lendatafiles
                fig_data_hist2[i] = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"S \ \mathrm{[AU]}",xlim=(0,1000),size=fig_3_across_size,frame=:box)
                histogram!(fig_data_hist2[i],X[i],xlim=fig_data_hist2_xlim,ylim=fig_data_hist2_ylim,normalize=:pdf,color=:lightgray, bins = 0:20:(round(maximum(X[i]),digits=-3)),xticks=fig_data_hist2_xticks,yticks=fig_data_hist2_yticks,legend=false,lw=0.5) 
            end
        end

        for i=1:lendatafiles
            println(i)
            # :pdf normalisation - total area of the bins is equal to 1
            bins_scatter= [0:20:1000;];      
            bins_scatter_mid = (bins_scatter[1:end-1]+bins_scatter[2:end])./2;
            bins_scatter_height = zeros(nrandsampleshist,length(bins_scatter_mid));
            bins_scatter_total_area = (20*length(X_sim_mean[1][i]))
            
            for k=1:nrandsampleshist # for each data set
                for j=1:length(bins_scatter_mid) 
                bins_scatter_height[k,j] = sum(bins_scatter[j] .<= X_sim_mean[k][i] .< bins_scatter[j+1])./bins_scatter_total_area;
                end
            end
            
            bins_scatter_height_mean = [mean(bins_scatter_height[:,j]) for j=1:length(bins_scatter_mid)];
            plot!(fig_data_hist2[i],bins_scatter_mid,bins_scatter_height_mean,color=col_vec[ii],lw=2,legend=false,ls=ls_vec[ii])

        end

    end

    display(fig_data_hist2[1])
    display(fig_data_hist2[2])
    display(fig_data_hist2[3])
    display(fig_data_hist2[4])
    display(fig_data_hist2[5])
    display(fig_data_hist2[6])

    savefig(fig_data_hist2[1],filepath_save * "fig_data_hist2_t1_" * string(comparison_id) * ".pdf")
    savefig(fig_data_hist2[2],filepath_save * "fig_data_hist2_t2_" * string(comparison_id) * ".pdf")
    savefig(fig_data_hist2[3],filepath_save * "fig_data_hist2_t3_" * string(comparison_id) * ".pdf")
    savefig(fig_data_hist2[4],filepath_save * "fig_data_hist2_t4_" * string(comparison_id) * ".pdf")
    savefig(fig_data_hist2[5],filepath_save * "fig_data_hist2_t5_" * string(comparison_id) * ".pdf")
    savefig(fig_data_hist2[6],filepath_save * "fig_data_hist2_t6_" * string(comparison_id) * ".pdf")
            
    #########################################
    # 10 - Plots - Comparison of ABC distances 
    #########################################

    N_dist = 2000;

    # ## METRIC 1 - Anderson Darling

    include(pwd() * "/Results/" * string(sim_id[1:2]) * "AD_Setup.jl")
    @time P_data1_metric1 = [dist(P_data1[i].θ) for i=1:N_dist];
    @time P_data2_metric1 = [dist(P_data2[i].θ) for i=1:N_dist];
    @time P_data3_metric1 = [dist(P_data3[i].θ) for i=1:N_dist];

    fig_distances_box_M1 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab="ABC distance",ylab=L"\mathcal{D}_{\mathrm{AD}}",size=fig_3_across_size,legend=false,framestyle=:box)
    boxplot!(fig_distances_box_M1,[1],P_data1_metric1,label="AD",color=col_vec[1])
    boxplot!(fig_distances_box_M1,[2],P_data2_metric1,label="CV",color=col_vec[2])
    boxplot!(fig_distances_box_M1,[3],P_data3_metric1,label="KS",color=col_vec[3],xticks=(1:3, ["AD", "CVM","KS"]))
    display(fig_distances_box_M1)
    savefig(fig_distances_box_M1,filepath_save * "fig_distances_box_M1_" * string(comparison_id) * ".pdf")
    savefig(fig_distances_box_M1,filepath_save * "fig_distances_box_M1_" * string(comparison_id) * ".png")

    # ## METRIC 2 - Cramer von Mises

    include(pwd() * "/Results/" * string(sim_id[1:2]) * "CV_Setup.jl")
    @time P_data1_metric2 = [dist(P_data1[i].θ) for i=1:N_dist];
    @time P_data2_metric2 = [dist(P_data2[i].θ) for i=1:N_dist];
    @time P_data3_metric2 = [dist(P_data3[i].θ) for i=1:N_dist];

    fig_distances_box_M2 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab="ABC distance",ylab=L"\mathcal{D}_{\mathrm{CVM}}",size=fig_3_across_size,legend=false,framestyle=:box)
    boxplot!(fig_distances_box_M2,[1],P_data1_metric2,label="AD",color=col_vec[1])
    boxplot!(fig_distances_box_M2,[2],P_data2_metric2,label="CV",color=col_vec[2])
    boxplot!(fig_distances_box_M2,[3],P_data3_metric2,label="KS",color=col_vec[3],xticks=(1:3, ["AD", "CVM","KS"]))
    display(fig_distances_box_M2)
    savefig(fig_distances_box_M2,filepath_save * "fig_distances_box_M2_" * string(comparison_id) * ".pdf")
    savefig(fig_distances_box_M2,filepath_save * "fig_distances_box_M2_" * string(comparison_id) * ".png")

    # ## METRIC 3 - Kolmogorov Smirnov

    include(pwd() * "/Results/" * string(sim_id[1:2]) * "KS_Setup.jl")
    @time P_data1_metric3 = [dist(P_data1[i].θ) for i=1:N_dist];
    @time P_data2_metric3 = [dist(P_data2[i].θ) for i=1:N_dist];
    @time P_data3_metric3 = [dist(P_data3[i].θ) for i=1:N_dist];

    fig_distances_box_M3 = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab="ABC distance",ylab=L"\mathcal{D}_{\mathrm{KS}}",size=fig_3_across_size,legend=false,framestyle=:box)
    boxplot!(fig_distances_box_M3,[1],P_data1_metric3,label="AD",color=col_vec[1])
    boxplot!(fig_distances_box_M3,[2],P_data2_metric3,label="CV",color=col_vec[2])
    boxplot!(fig_distances_box_M3,[3],P_data3_metric3,label="KS",color=col_vec[3],xticks=(1:3, ["AD", "CVM","KS"]))
    display(fig_distances_box_M3)
    savefig(fig_distances_box_M3,filepath_save * "fig_distances_box_M3_" * string(comparison_id) * ".pdf")
    savefig(fig_distances_box_M3,filepath_save * "fig_distances_box_M3_" * string(comparison_id) * ".png")


end

