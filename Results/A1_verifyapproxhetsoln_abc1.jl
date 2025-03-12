#=
    A1_verifyapproxhetsoln_1 - Simulate model with and without competition and compare results
    Uses output of ABC simulations
    COMPUTE ERROR
=#

# 1 - Load packages and modules
# 2 - Loop through data sets
# 3 - Load ABC-SMC results
# 4 - Setup location to save files
# 5 - Specify parameter values
# 6 - Compute error for all particles
	# 7 - Select parameter values for this loop
	# 8 - Define Model - NO competition 
	# 9 - Define and solve Model - WITH competition 
	# 10 - Simulate approximate solution
	# 11 - Compute error between full model and approximate solution
	# 12 - Save output

######################################################################
####### 1 - Load packages and modules
######################################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using Base.Threads
using DifferentialEquations 
using Sundials
using Random
using Distributions
using Trapz
using JLD2
using Plots
using LaTeXStrings

######################################################################
####### 2 - Loop through data sets
######################################################################

# sim_id_vec =["S1","S5","S7","S9","R1","R5","R6"]

println("ARGS ....")
println(parse(Int,ARGS[1]))

if parse(Int,ARGS[1]) == 1
    sim_id_vec =["S1"]
elseif parse(Int,ARGS[1]) == 2
    sim_id_vec =["S5"]
elseif parse(Int,ARGS[1]) == 3
    sim_id_vec =["S7"]
elseif parse(Int,ARGS[1]) == 4
    sim_id_vec =["S9"]
elseif parse(Int,ARGS[1]) == 5
    sim_id_vec =["R1"]
elseif parse(Int,ARGS[1]) == 6
    sim_id_vec =["R5"]
elseif parse(Int,ARGS[1]) == 7
    sim_id_vec =["R6"]
end



for ii=1:length(sim_id_vec)


    sim_id = sim_id_vec[ii]
    println(sim_id)

    #########################################
    # 3 - Load ABC-SMC results
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
    @load loadpath * loadfilename * ".jld2" P p_accept ε_seq

    ##############################################################
    ## 4 - Setup location to save files
    ##############################################################

    filepath_save = pwd() * "/" * "Figures" * "/Verifyapprox_" * savename * "/"; # location to save figures 
    isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

    ######################################################################
    ####### 5 - Specify parameter values
    ######################################################################

    μr_vec = [P[i].θ[1] for i=1:length(P)];
    σr_vec = [P[i].θ[2] for i=1:length(P)];
    μK_vec = [P[i].θ[3] for i=1:length(P)];
    σK_vec = [P[i].θ[4] for i=1:length(P)];

    particles_per_cell = U*V;
    Nsamples =2000; # Number of ABC particles

    total_cells = 100_000;
    # Ncells_sample = 500; # Number of simulated cells 
    Ncells_sample = 20_000;  # Number of simulated cells 

    ######################################################################
    ####### 6 - Compute error for all particles
    ######################################################################

    error_P = zeros(Nsamples);

    @threads for i=1:Nsamples
        
        println(i) # print sample id

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
        @time sol = solve(prob,CVODE_BDF(linear_solver= :GMRES),saveat=saveatvec,abstol = 1e-10, reltol = 1e-10); 

        ######################################################################
        ####### 10 - Simulate approximate solution
        ######################################################################

        ### Define time points to approximate integral and to save output
        int_step = 2;
        t_int_eval = [0:1/int_step:24;]*3600;
        t_int_eval_save_index = [findfirst(t_int_eval .== saveatvec[i]) for i=1:length(saveatvec)];
    
        ### Estimate  u(t) from many realisations of the independent model
        U_int_eval = mean.([[(particles_per_cell  - solve_ode_model_independent(t,[r_sample[j],K_sample[j]],θfixed)) for j=1:length(r_sample)] for t=t_int_eval ]);
    
        ### Estimate  P_i(t) from many realisations of the independent model
        I=[trapz(t_int_eval[1:t_int_eval_save_index[i]],U_int_eval[1:t_int_eval_save_index[i]]/V) for i=1:length(t_int_eval_save_index)];
    
        [[K_sample[j]*(1 - exp(-(S*r_sample[j]/K_sample[j])*I[mm] )) for j=1:length(r_sample) ] for mm=1:length(t_int_eval_save_index)]

        ######################################################################
        ####### 11 - Compute error between full model and approximate solution
        #####################################################################

        error_P[i] = sum([sum(abs.([sol.u[ti][j] for ti=1:length(sol.t)] .- [K_sample[j-1]*(1 - exp(-(S*r_sample[j-1]/K_sample[j-1])*I[mm] )) for mm=1:length(sol.t)])) for j=2:Ncells_sample+1])/(Ncells_sample*length(sol.t));
    
    end

    ######################################################################
    ####### 12 - Save output
    ######################################################################

    @save loadpath * loadfilename * "approximation_error_500" * ".jld2" error_P
   
end