#=
    A1_verifyapproxhetsoln_2 - Simulate model with and without competition and compare results
    Uses output of ABC simulations
    PLOTTING ERROR (uses computation of error from A1_verifyapproxhetsoln_1)
=#

# 1 - Load packages and modules
# 2 - Plot settings (size,fnt)
# 3 - Loop through data sets (using HPC)
    # 4 - load ABC-SMC results
    # 5 - Setup location to save files
    # 6 - Specify parameter values
    # 7 - Load output and find ABC particle with the maximum absolute error
    # 8 - Plot - Error - Grid in r and K space
    # 9 -  For the maximum absolute error plot the differences
        # 10 - Identify parameter values for this loop
        # 11 -  Define Model - NO competition 
        # 12 - Define and solve Model - WITH competition 
        # 13 - Simulate approximate solution
        # 14 - Plots - Compare particles in solution per cell, U(t) from heterogeneous solution to approximate solution
        # 15 - Plots - Compare P(t;u from independent model) to P(t; u from full model)
        # 16 - Plots - P(t; u from full model) - individual trajectories
        # 17 - Plots - P(t; u from full model) - interval

######################################################################
####### 1 - Load packages and modules
######################################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using DifferentialEquations 
using Sundials
using Random
using Distributions
using Trapz
using JLD2
using Plots
using LaTeXStrings

######################################################################
# 2 - Plot settings (size,fnt)
######################################################################

pyplot() 
fnt = Plots.font("sans-serif", 10)

mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

######################################################################
####### 3 - Loop through data sets (using HPC)
######################################################################

sim_id_vec =["S1","S5","S7","S9","R1","R5","R6"]

for ii=1:length(sim_id_vec)

    sim_id = sim_id_vec[ii]
    println("-----------" * sim_id * "--------------") # print simulation id

    #########################################
    # 4 - load ABC-SMC results
    #########################################

    if sum(sim_id .== ["S1","S5","S7","S9"]) > 0
        loadpath = "Results/Files_syn/";
        savename = "Fig_" * string(sim_id) * "ADE_Inference";
        loadfilename = string(sim_id) * "ADE_inference"
    elseif sum(sim_id .== ["R1","R5","R6"]) > 0
        loadpath = "Results/Files_real/";
        savename = "Fig_" * string(sim_id) * "AD_ABCE_Inference";
        loadfilename = string(sim_id) * "AD_ABCE_inference"
    end
    include(pwd() * "/Results/" * string(sim_id) * "AD_Setup.jl")

    @load loadpath * loadfilename * ".jld2" P p_accept ε_seq # load from ABC-SMC with target acceptance threshold
    # @load loadpath * loadfilename * ".jld2" P ε # load from ABC-SMC with target acceptance probability

    ######################################################################
    # 5 - Setup location to save files
    ######################################################################

    filepath_save = pwd() * "/" * "Figures" * "/Verifyapprox_" * savename * "/"; # location to save figures 
    isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

    ######################################################################
    # 6 - Specify parameter values
    ######################################################################

    μr_vec = [P[i].θ[1] for i=1:length(P)];
    σr_vec = [P[i].θ[2] for i=1:length(P)];
    μK_vec = [P[i].θ[3] for i=1:length(P)];
    σK_vec = [P[i].θ[4] for i=1:length(P)];

    Nsamples = length(P);
    particles_per_cell = U*V;

    n_to_plot = 2000;
    n_to_plot = 500;

    ######################################################################
    # 7 - Load output and find ABC particle with the maximum absolute error
    ######################################################################

    @load loadpath * loadfilename * "approximation_error_500" * ".jld2" error_P
    Nsamples =length(error_P);

    # find ABC particle with the biggest error
    testa,index_max = findmin(-error_P)
    error_P[index_max]
    # μr = μr_vec[index_max];
    # σr = σr_vec[index_max]; 
    # μK = μK_vec[index_max]; 
    # σK = σK_vec[index_max]; 

    # ######################################################################
    # # 8 - Plot - Error - Grid in r and K space
    # ######################################################################

    # plot μr v σr
    fig_error_r_dist = plot(layout=grid(1,1),xlab=L"m_{r}",ylab=L"s_{r}",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,framestyle=:box,legend=false) #,xticks=fig_density_μr_xticks,xlim=fig_density_μr_xlim,yticks=fig_density_σr_xticks,ylim=fig_density_σr_xlim)
    scatter!(fig_error_r_dist,μr_vec[1:n_to_plot],σr_vec[1:n_to_plot],zcolor=error_P[1:n_to_plot],color=:Blues_3,msw=0)
    scatter!(fig_error_r_dist,[μr_vec[index_max]],[σr_vec[index_max]],color=:orange,msw=0,markersize=5)
    display(fig_error_r_dist)
    savefig(fig_error_r_dist,filepath_save * "fig_approximation_error_r_dist" * ".pdf")

    # plot μK v σK
    fig_error_K_dist = plot(layout=grid(1,1),xlab=L"m_{K}",ylab=L"s_{K}",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,framestyle=:box,palette=:Blues_3,legend=false) #,xticks=fig_density_μK_xticks,xlim=fig_density_μK_xlim,yticks=fig_density_σK_xticks,ylim=fig_density_σK_xlim)
    scatter!(fig_error_K_dist,μK_vec[1:n_to_plot],σK_vec[1:n_to_plot],zcolor=error_P[1:n_to_plot],msw=0,color=:Blues_3)
    scatter!(fig_error_K_dist,[μK_vec[index_max]],[σK_vec[index_max]],color=:orange,msw=0,markersize=5)
    display(fig_error_K_dist)
    savefig(fig_error_K_dist,filepath_save * "fig_approximation_error_K_dist" * ".pdf")

    ######################################################################
    # 9 -  For the maximum absolute error plot the differences
    ######################################################################

    for i=index_max

        println(i) # print index of particle with maximum absolute error 

        ######################################################################
        ####### 10 - Identify parameter values for this loop
        ######################################################################

        # parameter values of particle with maximum absolute error
        μr = μr_vec[i];
        σr = σr_vec[i]; 
        μK = μK_vec[i]; 
        σK = σK_vec[i]; 

        # define lognormal distributions for r and K
        r = LogNormal(log(μr^2 / sqrt(σr^2 + μr^2)),sqrt(log(σr^2/μr^2 + 1)));
        K = LogNormal(log(μK^2 / sqrt(σK^2 + μK^2)),sqrt(log(σK^2/μK^2 + 1)));

        total_cells = 100_000;
        # Ncells_sample = 1000;
        Ncells_sample = 20_000;

        # sample lognormal distributions for r and K
        Random.seed!(1)
        r_sample = rand(r,Ncells_sample);
        K_sample = rand(K,Ncells_sample);

        ######################################################################
        # 11 -  Define Model - NO competition 
        ######################################################################

        function solve_ode_model_independent(t,θ::Vector,θfixed::Vector)
            r,P = θ;
            S,V,C,U = θfixed;
            if P == V*U
                (C*r*S*U^2*V*t)/(P + C*r*S*U*t)
            else
                V*U .* (1 - (V*U - P)./(V*U - P*exp(-(r*C*S*(U*V - P)*t)/(P*V))  ) )
            end
        end

        ######################################################################
        # 12 - Define and solve Model - WITH competition 
        ######################################################################

        ## Plot - total concentration comparison
        function nanoodes_concpercell(du, u, p, t)
            # time evolution of total number of particles in the solution
            du[1]= -(1/V)*(S/Ncells_sample)*u[1]*sum([(r_sample[j-1]*(K_sample[j-1] - u[j])/K_sample[j-1]) for j=2:(Ncells_sample+1)]);
            # time evolution of the number of particles per cell
            for j=2:(Ncells_sample+1)
                du[j] = S*(r_sample[j-1]*(K_sample[j-1] - u[j])/K_sample[j-1])*u[1];
            end
        end

        u0 = [particles_per_cell/V;zeros(Ncells_sample)] # initial condition
        tspan = (0.0, 24.0*3600.0) # time span
        saveatvec = [0.0:0.5:24.0;]*3600.0;
        prob = ODEProblem(nanoodes_concpercell, u0, tspan) #,saveat=0.1)
        @time sol = solve(prob,CVODE_BDF(linear_solver = :GMRES),saveat=saveatvec,abstol = 1e-10, reltol = 1e-10); #,saveat=0.1 # 1057 seconds

        ######################################################################
        # 13 - Simulate approximate solution
        ######################################################################

        ### Define time points to approximate integral and to save output
        int_step = 2;
        t_int_eval = [0:1/int_step:24;]*3600;
        t_int_eval_save_index = [findfirst(t_int_eval .== saveatvec[i]) for i=1:length(saveatvec)];

        ### Estimate  u(t) from many realisations of the independent model
        U_int_eval = mean.([[(particles_per_cell  - solve_ode_model_independent(t,[r_sample[j],K_sample[j]],θfixed)) for j=1:length(r_sample)] for t=t_int_eval ])/V;

        ### Estimate  P_i(t) from many realisations of the independent model
        I=[trapz(t_int_eval[1:t_int_eval_save_index[i]],U_int_eval[1:t_int_eval_save_index[i]]) for i=1:length(t_int_eval_save_index)];

        ######################################################################
        ####### 14 - Plots - Compare particles in solution per cell, U(t) from heterogeneous solution to approximate solution
        ######################################################################

        fig_concentrationpercell_comparison2 = plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"u(t), \bar{u}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,xticks=[0,6,12,18,24],yticks=[0,25,50,75,100],framestyle=:box)
        plot!(fig_concentrationpercell_comparison2,sol.t ./3600,[sol.u[ti][1] for ti=1:length(sol.t)]*V,color=:black,lw=3)
        plot!(fig_concentrationpercell_comparison2,sol.t ./3600,mean.([[(particles_per_cell  - solve_ode_model_independent(sol.t[ti],[r_sample[j-1],K_sample[j-1]],θfixed)) for j=2:Ncells_sample+1] for ti=1:length(sol.t) ]),ls=:dash,color=:red,lw=3,ylim=(0,100))
        display(fig_concentrationpercell_comparison2)
        savefig(fig_concentrationpercell_comparison2,filepath_save * "fig_concentrationpercell_sim" * string(i) * ".pdf")

        ######################################################################
        ####### 15 - Plots - Compare P(t;u from independent model) to P(t; u from full model)
        ######################################################################

        fig_maxdifference_approx = plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"P^{(j)}(t;u(t)) - P^{(j)}(t;\bar{u}(t))",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,framestyle=:box,palette=:Blues_3)
        for j=2:100:(Ncells_sample+1) 
            plot!(fig_maxdifference_approx,sol.t ./3600,[sol.u[ti][j] for ti=1:length(sol.t)] .- [K_sample[j-1]*(1 - exp(-(S*r_sample[j-1]/K_sample[j-1])*I[mm] )) for mm=1:length(t_int_eval_save_index)]) 
        end
        plot!(fig_maxdifference_approx,sol.t ./3600,[mean([sol.u[mm][j] .- K_sample[j-1]*(1 - exp(-(S*r_sample[j-1]/K_sample[j-1])*I[mm])) for j=2:Ncells_sample+1]) for mm=1:length(t_int_eval_save_index)],lw=3,color=:black,ylim=(-40,5)) 
        display(fig_maxdifference_approx)
        savefig(fig_maxdifference_approx,filepath_save * "fig_maxdifference_approx_sim" * string(i) *  ".pdf")

        ######################################################################
        ####### 16 - Plots - P(t; u from full model) - individual trajectories
        ######################################################################

        fig_Pt= plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"P^{(j)}(t;u(t))",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,ylim=[0,200],framestyle=:box,palette=:Blues_3)
        for j=2:100:(Ncells_sample+1) 
            plot!(fig_Pt,sol.t ./3600,[sol.u[ti][j] for ti=1:length(sol.t)]) 
        end
        display(fig_Pt)
        savefig(fig_Pt,filepath_save * "fig_Pt" * string(i) *  ".pdf")

        ######################################################################
        ####### 17 - Plots - P(t; u from full model) - interval
        ######################################################################

        Pt_lb95 = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.025) for ti=1:length(sol.t)];
        Pt_lb50 = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.25) for ti=1:length(sol.t)];
        Pt_m = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.5) for ti=1:length(sol.t)];
        Pt_ub50 =  [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.75) for ti=1:length(sol.t)];
        Pt_ub95 =  [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.975) for ti=1:length(sol.t)];

        fig_Pt_interval = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ \mathrm{[hr]}",ylab=L"P(t;u(t))",size=fig_2_across_size,ylim=[0,200],xticks=[0,12,24],yticks=[0,100,200],framestyle=:box)
        plot!(fig_Pt_interval,subplot=1,sol.t ./3600,Pt_m,ribbon=(Pt_m-Pt_lb95,Pt_ub95-Pt_m),label="95% interval",color=:deepskyblue1,legend=false)
        plot!(fig_Pt_interval,subplot=1,sol.t ./3600,Pt_m,ribbon=(Pt_m-Pt_lb50,Pt_ub50-Pt_m),color=:darkblue,label="50% interval",legend=false)
        hline!(fig_Pt_interval,[100],lw=2,ls=:dash,color=:red)
        display(fig_Pt_interval)
        savefig(fig_Pt_interval,filepath_save * "fig_Pt_interval" * string(i) * ".pdf")

    end

end

