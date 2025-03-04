#=
    A1_verifyapproxhetsoln_chosen - Simulate model with and without competition and compare results
    CHOSEN PARAMETER VALUES
=#

# 1 - Load packages and modules
# 2 - Plot settings (size,fnt)
# 3 - Setup location to save files
# 4 - Setup problem
# 5 - Specify chosen parameter values
# 6 - Loop through chosen parameter values
    # 7 - Select parameter values for this loop
    # 8 - Define Model - NO competition
    # 9 - Define and solve Model - WITH competition
    # 10 - Simulate approximate solution
    # 11 - Plots - Compare particles in solution per cell, U(t) from heterogeneous solution to approximate solution
    # 12 - Plots - Compare P(t;u from independent model) to P(t; u from full model)
    # 13 - Plots - P(t) - individual trajectories
    # 14 - Plots - P(t) - interval

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
using Plots
using LaTeXStrings

######################################################################
####### 2 - Plot settings (size,fnt)
######################################################################

pyplot() 
fnt = Plots.font("sans-serif", 10) 

mm_to_pts_scaling = 283.46/72;
fig_2_across_size=(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_3_across_size=(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_4_across_size=(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

##############################################################
## 3 - Setup location to save files
##############################################################

savename = "Fig_competition_chosenvalues";
filepath_save = pwd() * "/" * "Figures" * "/" * savename * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 4 - Setup problem
##############################################################

include("R1AD_Setup.jl")

######################################################################
####### 5 - Specify chosen parameter values
######################################################################

particles_per_cell = 100;

μr_vec = [4.0e-6,4.0e-6];
σr_vec = [1.0e-6,1.0e-6];
μK_vec = [30.0,110.0];
σK_vec = [10.0,50.0];

total_cells = 100_000;
Ncells_sample = 20_000;

######################################################################
####### 6 - Loop through chosen parameter values
######################################################################

for i=1:2

    println(i)  # print dataset id

    ######################################################################
    ####### 7 - Select parameter values for this loop
    ######################################################################

    # select parameter values
    μr = μr_vec[i];
    σr = σr_vec[i]; 
    μK = μK_vec[i]; 
    σK = σK_vec[i]; 

    # define lognormal distributions for r and K 
    r = LogNormal(log(μr^2 / sqrt(σr^2 + μr^2)),sqrt(log(σr^2/μr^2 + 1)));
    K = LogNormal(log(μK^2 / sqrt(σK^2 + μK^2)),sqrt(log(σK^2/μK^2 + 1)));

    # sample lognormal distributions for r and K
    Random.seed!(1)
    r_sample = rand(r,Ncells_sample);
    K_sample = rand(K,Ncells_sample);

    ######################################################################
    ####### 8 - Define Model - NO competition 
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
    ####### 9 - Define and solve Model - WITH competition 
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
    prob = ODEProblem(nanoodes_concpercell, u0, tspan) 
    @time sol = solve(prob,CVODE_BDF(linear_solver = :GMRES),saveat=saveatvec,abstol = 1e-10, reltol = 1e-10); 

    ######################################################################
    ####### 10 - Simulate approximate solution
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
    ####### 11 - Plots - Compare particles in solution per cell, U(t) from heterogeneous solution to approximate solution
    ######################################################################

    fig_concentrationpercell_comparison2 = plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"u(t), \bar{u}(t)",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,xticks=[0,12,24],yticks=[0,25,50,75,100],framestyle=:box)
    plot!(fig_concentrationpercell_comparison2,sol.t ./3600,[sol.u[ti][1] for ti=1:length(sol.t)]*V,color=:black,lw=3)
    plot!(fig_concentrationpercell_comparison2,sol.t ./3600,mean.([[(particles_per_cell  - solve_ode_model_independent(sol.t[ti],[r_sample[j-1],K_sample[j-1]],θfixed)) for j=2:Ncells_sample+1] for ti=1:length(sol.t) ]),ls=:dash,color=:red,lw=3,ylim=(0,100))
    display(fig_concentrationpercell_comparison2)
    savefig(fig_concentrationpercell_comparison2,filepath_save * "fig_concentrationpercell_sim" * string(i) * ".pdf")

    ######################################################################
    ####### 12 - Plots - Compare P(t;u from independent model) to P(t; u from full model)
    ######################################################################

    fig_maxdifference_approx = plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"P^{(j)}(t;u(t)) - P^{(j)}(t;\bar{u}(t))",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,xticks=[0,12,24],framestyle=:box,palette=:Blues_3)
    for j=2:100:(Ncells_sample+1) 
        plot!(fig_maxdifference_approx,sol.t ./3600,[sol.u[ti][j] for ti=1:length(sol.t)] .- [K_sample[j-1]*(1 - exp(-(S*r_sample[j-1]/K_sample[j-1])*I[mm] )) for mm=1:length(t_int_eval_save_index)]) 
    end
    plot!(fig_maxdifference_approx,sol.t ./3600,[mean([sol.u[mm][j] .- K_sample[j-1]*(1 - exp(-(S*r_sample[j-1]/K_sample[j-1])*I[mm])) for j=2:Ncells_sample+1]) for mm=1:length(t_int_eval_save_index)],lw=3,color=:black,ylim=(-40,5)) 
    display(fig_maxdifference_approx)
    savefig(fig_maxdifference_approx,filepath_save * "fig_maxdifference_approx_sim" * string(i) *  ".pdf")
    
    ######################################################################
    ####### 13 - Plots - P(t; u from full model) - individual trajectories
    ######################################################################

    fig_Pt= plot(layout=grid(1,1),legend=false,xlab=L"t \ \mathrm{[hr]}",ylab=L"P^{(j)}(t;u(t))",titlefont=fnt, guidefont=fnt, tickfont=fnt,size=fig_2_across_size,ylim=[0,200],framestyle=:box)
    for j=2:100:(Ncells_sample+1) 
        plot!(fig_Pt,sol.t ./3600,[sol.u[ti][j] for ti=1:length(sol.t)]) 
    end
    display(fig_Pt)
    savefig(fig_Pt,filepath_save * "fig_Pt" * string(i) *  ".pdf")

    ######################################################################
    ####### 14 - Plots - P(t; u from full model) - interval
    ######################################################################

    Pt_lb95 = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.025) for ti=1:length(sol.t)];
    Pt_lb50 = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.25) for ti=1:length(sol.t)];
    Pt_m = [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.5) for ti=1:length(sol.t)];
    Pt_ub50 =  [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.75) for ti=1:length(sol.t)];
    Pt_ub95 =  [quantile([sol.u[ti][j] for j=2:(Ncells_sample+1)],0.975) for ti=1:length(sol.t)];
    
    fig_Pt_interval = plot(layout=grid(1,1),titlefont=fnt, guidefont=fnt, tickfont=fnt,xlab=L"t \ \mathrm{[hr]}",ylab=L"P(t;u(t))",size=fig_2_across_size,ylim=[0,200],xticks=[0,12,24],yticks=[0,100,200],framestyle=:box)
    plot!(fig_Pt_interval,subplot=1,sol.t ./3600,Pt_m,ribbon=(Pt_m-Pt_lb95,Pt_ub95-Pt_m),label="95% interval",color=:deepskyblue1,legend=false)
    plot!(fig_Pt_interval,subplot=1,sol.t ./3600,Pt_m,ribbon=(Pt_m-Pt_lb50,Pt_ub50-Pt_m),color=:darkblue,label="50% interval",legend=false)
    hline!(fig_Pt_interval,[100],lw=2,ls=:dash,color=:black)
    display(fig_Pt_interval)
    savefig(fig_Pt_interval,filepath_save * "fig_Pt_interval" * string(i) * ".pdf")

end
