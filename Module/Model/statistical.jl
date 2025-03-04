#=

    statistical.jl

    Contains code to simulate the statistical model.
    
    Modified from https://github.com/ap-browning/internalisation by Alexander P. Browning (Queensland University of Technology)

=#


function simulate_model_noiseless(θ::Vector,θfixed::Vector,T::Vector,d::Function;N::Int=1000)

    # Parameters
    ξᵢ = rand(d(θ[1:4]),N) # Sample from parameter distribution

    rᵢ = @view ξᵢ[1,:]
    Kᵢ = @view ξᵢ[2,:]

    # solve the hierarchical ODE model for P(t)
    P = solve_ode_model(T,[rᵢ,Kᵢ],θfixed);

    return P

end


function simulate_model_bootstrap_manyp(θ::Vector,θfixed::Vector,T::Vector,d::Function;N::Int=1000,noise_cellonly::Vector,noise_particleonly::Vector)
    # (for each time point)
    
    # Parameters
    ξᵢ = rand(d(θ[1:4]),N) # Sample from parameter distribution

    rᵢ = @view ξᵢ[1,:]
    Kᵢ = @view ξᵢ[2,:]

    # solve the hierarchical ODE model for P(t)
    Phode = solve_ode_model(T,[rᵢ,Kᵢ],θfixed);

    # sample cell fluorescence
    noise_cellonlyᵢ = [rand(noise_cellonly,N) for s=1:length(Phode)];

    # for each cell, generate ceil(P(t)) samples of particle-only fluorescence
    F_p = [[rand(noise_particleonly,Int(ceil(Phode[s][i]))) for i=1:N] for s=1:length(Phode)];

    # compute the total fluorescence
    P = [[if ceil(Phode[s][i]) >= 1 noise_cellonlyᵢ[s][i] + sum(F_p[s][i][1:Int(floor(Phode[s][i]))]) + (Phode[s][i]-floor(Phode[s][i]))*F_p[s][i][Int(ceil(Phode[s][i]))] else noise_cellonlyᵢ[s][i] end for i=1:N] for s=1:length(Phode)];

    return P
end

