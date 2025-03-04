#=

    deterministic.jl

    Contains code to solve the deterministic ODE model

=# 

function solve_ode_model_independent(t,θ::Vector,θfixed::Vector)
    r,K = θ;
    S,V,C,U = θfixed;
    # S # surface area of the cell boundary [m^2]
    # V # volume of the solution [m^3]
    # C # SC (dimensionless surface coverage of cells) [-]
    # U # initial concentration of particles in the solution [mol m^-3]
    if K == V*U
        (C*r*S*U^2*V*t)/(K + C*r*S*U*t)
    else
        V*U .* (1 - (V*U - K)./(V*U - K*exp(-(r*C*S*(U*V - K)*t)/(K*V))  ) )
    end
end


function solve_ode_model(T::Vector,θ::Vector,θfixed::Vector)
  
    r,K = θ;
    S,V,C,U= θfixed;
    particles_per_cell = 100.0;
    # S # surface area of the cell boundary [m^2]
    # V # volume of the solution [m^3]
    # C # SC (dimensionless surface coverage of cells) [-]
    # U=100.0/V # initial concentration of particles in the solution [mol m^-3]

    ### Define time points to approximate integral and to save output
    int_step = 2;
    t_int_eval = [0:1/int_step:24;]*3600;
    t_int_eval_save_index = [findfirst(t_int_eval .== T[i]) for i=1:length(T)];

    ### Estimate  u(t) from many realisations of the independent model
    U_int_eval = mean.([[(particles_per_cell  - solve_ode_model_independent(t,[r[j],K[j]],θfixed)) for j=1:min(100,length(r))] for t=t_int_eval ]);

    ### Estimate  P_i(t) from many realisations of the independent model
    I=[trapz(t_int_eval[1:t_int_eval_save_index[i]],U_int_eval[1:t_int_eval_save_index[i]]/V) for i=1:length(t_int_eval_save_index)];

    [[K[j]*(1 - exp(-(S*r[j]/K[j])*I[mm] )) for j=1:length(r) ] for mm=1:length(t_int_eval_save_index)]

end
