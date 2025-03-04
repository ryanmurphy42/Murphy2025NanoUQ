#=

    abc.jl

    Contains code to conduct approximate Bayesian computation (ABC)

    Author:     Alexander P. Browning
                ======================
                School of Mathematical Sciences
                Queensland University of Technology
                ======================
                ap.browning@icloud.com
                alexbrowning.me

    From https://github.com/ap-browning/internalisation

    # RM MODIFICATION - abc_smc_εtarget - is a modified function that terminates at a specified ABC error threshold

=# 

##############################################################
## ABC Sequential Monte-Carlo (SMC)
##############################################################
"""
    abc_smc(dist,prior;n=100,α=0.5,S₀=100,c=0.01,ptarget=0.005,maxsteps=20,f=2.0)

Perform ABC SMC.
"""
function abc_smc(dist,prior;n=100,kwargs...)
    P = sample(prior,dist,n)    # Initialise using prior
    print("Initialised $n particles. $(round(minimum(P).d,sigdigits=4)) ≤ d ≤ $(round(maximum(P).d,sigdigits=4)).\n")
    return abc_smc(P,dist,prior;kwargs...)
end
function abc_smc(P,dist,prior;step=1,α=0.5,ptarget=0.01,maxsteps=30,f=2.0)

    # Loop until target reached
    p = 1.0
    while p > ptarget && step ≤ maxsteps    
        @printf("Step %-3s (ε = %-8s\b...",
            "$step",
            "$(round(sort(distances(P))[(Int ∘ floor)(α * length(P))],sigdigits=4))",
        )
        
        p = abc_smc_step!(P,dist,prior,α,f)
        step += 1

        @printf("\b\b\b dmin = %-8s p = %-8s ESS = %-8s)\n",
            "$(round(minimum(P).d,sigdigits=4))",
            "$(round(p,sigdigits=2))",
            "$(round(1 / sum(weights(P).^2),sigdigits=5))"
        )     

    end

    # Finish
    return P
end

"""
    abc_smc_step!(P,dist,prior,α,f=2.0)

Perform a single step of ABC SMC, where α * n particles are discarded.

Proposal is estimated as f * Σ, where Σ is the (weighted) covariance matrix of the non-dicarded particles.

"""
function abc_smc_step!(P,dist,prior,α,f=2.0)
    N = length(P); Nα = (Int ∘ floor)(α * N)

    sort!(P)                # Sort particles
    ε = P[N-Nα].d           # Current threshold   
    Σ = f*cov(P[1:N-Nα])    # Proposal covariance
    prop = MvNormal(Σ)      # Proposal

    # Resample and update particles
    tries = 0
    @threads for j = (N - Nα + 1) : N
        θs,d = similar(P[1].θ),Inf
        while d > ε
            θs = copy(sample(P[1:N-Nα],weights(P[1:N-Nα])).θ) + rand(prop)
            if insupport(prior,θs)
                d = dist(θs)
                tries += 1
            end
        end
        P[j] = Particle(θs,d,pdf(prior,θs) / sum(p.w * pdf(prop,θs - p.θ) for p in P[1:N-Nα]))
    end

    # Reweight
    w = weights(P)
    w[1:N-Nα] /= sum(w[1:N-Nα])         # Normalise
    w[N-Nα+1:N] /= sum(w[N-Nα+1:N])     # Normalise
    w[1:N-Nα] *= (N - Nα) / N           # Re-weight
    w[N-Nα+1:N] *= Nα / N               # Re-weight
    [p.w = wᵢ for (p,wᵢ) in zip(P,w)]

    # Report acceptance probability (exclude trials where we propose a value outside the prior)
    return Nα / tries
end


##############################################################
## ABC Sequential Monte-Carlo (SMC) with εtarget
##############################################################

function abc_smc_εtarget(dist,prior;n=100,kwargs...)
    P = sample(prior,dist,n)    # Initialise using prior
    print("Initialised $n particles. $(round(minimum(P).d,sigdigits=4)) ≤ d ≤ $(round(maximum(P).d,sigdigits=4)).\n")
    return abc_smc_εtarget(P,dist,prior;kwargs...)
end
function abc_smc_εtarget(P,dist,prior;step=1,α=0.5,ptarget=0.01,maxsteps=30,f=2.0,εtarget=1.0)

    ε = Inf;

    println("ε ..." * string(ε))

    p_accept = [];
    ε_seq = [];

    # Loop until εtarget reached
    while ε > εtarget && step ≤ maxsteps    
        
        p, ε = abc_smc_step_εtarget!(P,dist,prior,α,f,εtarget)
        
        push!(p_accept,p)
        push!(ε_seq,ε)
        
        @printf("Step %-3s (ε = %-8s\b...",
            "$step",
            "$(round(ε,sigdigits=4))",
        )

        @printf("\b\b\b dmin = %-8s p = %-8s  ESS = %-8s)\n",
            "$(round(minimum(P).d,sigdigits=4))",
            "$(round(p,sigdigits=2))",
            "$(round(1 / sum(weights(P).^2),sigdigits=5))"
        )     

        step += 1

    end

    # Finish
    return P, p_accept, ε_seq
end

"""
    abc_smc_step_εtarget!(P,dist,prior,α,f=2.0,εtarget)

Perform a single step of ABC SMC, where α * n particles are dicarded.

Proposal is estimated as f * Σ, where Σ is the (weighted) covariance matrix of the non-dicarded particles.

"""
function abc_smc_step_εtarget!(P,dist,prior,α,f=2.0,εtarget=1.0)
    N = length(P); 
    Nα = (Int ∘ floor)(α * N)

    sort!(P)                # Sort particles
    ε = max(P[N-Nα].d,εtarget)           # Current threshold   
    Σ = f*cov(P[1:N-Nα])    # Proposal covariance
    prop = MvNormal(Σ)      # Proposal

    # Resample and update particles
    tries = 0
    @threads for j = (N - Nα + 1) : N
        θs,d = similar(P[1].θ),Inf
        while d > ε
            θs = copy(sample(P[1:N-Nα],weights(P[1:N-Nα])).θ) + rand(prop)
            if insupport(prior,θs)
                d = dist(θs)
                tries += 1
            end
        end
        P[j] = Particle(θs,d,pdf(prior,θs) / sum(p.w * pdf(prop,θs - p.θ) for p in P[1:N-Nα]))
    end

    # Reweight
    w = weights(P)
    w[1:N-Nα] /= sum(w[1:N-Nα])         # Normalise
    w[N-Nα+1:N] /= sum(w[N-Nα+1:N])     # Normalise
    w[1:N-Nα] *= (N - Nα) / N           # Re-weight
    w[N-Nα+1:N] *= Nα / N               # Re-weight
    [p.w = wᵢ for (p,wᵢ) in zip(P,w)]

    # Report acceptance probability (exclude trials where we propose a value outside the prior)
    # return Nα / tries

    # Report acceptance probability (exclude trials where we propose a value outside the prior) AND error threshold
    # println("(Nα / tries) ....." * string(Nα / tries))
    # println("ε ....." * string(ε))


    return (Nα / tries), ε
end
