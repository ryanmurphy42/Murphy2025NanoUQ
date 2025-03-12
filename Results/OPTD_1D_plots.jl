#=
    OPTD_1D_plots - Compute overall rank and plots
=#

# 1 - Load packages and modules
# 2 - Default plot settings (plot sizes, axis limits, ticks)
# 3 - Load data
# 4 - Setup location to save files
# 5 - Plot distribution of θsim_save_all (check sampling of uniform priors)
# 6 - Compute average ranking across low r, intermediate r, and high r
# 7 - Specify/identify times for selected designs (early, middle, late, equalspace, expdata, bestforlowr, bestforintr, bestforhighr, overall)
# 8 - Plot utility (within condition) - bar and box plots
# 9 - For the top 100 designs for each condition plot a histogram of the time points
# 10 -  Plot - Design times schematic
# 11 - For each selected design view overall rank
# 12 - For each selected design view individual ranks for the three scenarios
# 13 - For each selected design view average of the three ranks
# 14 - For each selected design view Normalised mean utilities
# 15 - Minimum normalised mean utility for each scenario 

##############################################################
## 1 - Load packages and modules
##############################################################

push!(LOAD_PATH,pwd() * "/Module/Inference")
push!(LOAD_PATH,pwd() * "/Module/Model")

using Inference
using Model
using Combinatorics
using Distributions
using JLD2
using LaTeXStrings
using LinearAlgebra
using Plots
using Random
using StatsBase

#########################################
# 2 - Default plot settings (plot sizes, axis limits, ticks)
#########################################

pyplot() # plot options
fnt = Plots.font("sans-serif", 10); # plot options
fnt_inset = Plots.font("sans-serif", 10); # plot options
global cur_colors = palette(:default); # plot options

# figure - sizes
mm_to_pts_scaling = 283.46/72;
fig_size_1across =(150.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_size_2across =(75.0*mm_to_pts_scaling,50.0*mm_to_pts_scaling);
fig_size_3across =(50.0*mm_to_pts_scaling,33.3*mm_to_pts_scaling);
fig_size_4across =(37.5*mm_to_pts_scaling,25.0*mm_to_pts_scaling);

## colours
# purples
color_early=RGB(220/255, 200/255, 255/255);
color_middle=RGB(160/255, 100/255, 205/255);
color_late=RGB(100/255, 0/255, 150/255);
# yellows
color_equalspace=RGB(255/255, 245/255, 200/255);
color_expdata=RGB(255/255, 180/255, 50/255);
# greens
color_bestforlowr = RGB(200/255, 255/255, 200/255)
color_bestforintr = RGB(40/255, 190/255, 0/255)
color_bestforhighr = RGB(0/255,100/255, 30/255)
# blue
color_overall_rank = RGB(30/255, 144/255, 255/255)


##############################################################
## 3 - Load data
##############################################################

# intermediate r
save_name = "intr"
filepath_save =  pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * save_name * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist
@load filepath_save * "OPTD_1C_all" * ".jld2" ε_comparison_threshold utility_D  utility_mean_D θsim_save_all discrepancy_all  T_design D N  T J
ε_comparison_threshold_intr = copy(ε_comparison_threshold);
θsim_save_all_intr = copy(θsim_save_all);
194.04256806251942
utility_mean_D_intr = copy(utility_mean_D);

# low r
save_name = "lowr"
filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * save_name * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist
@load filepath_save * "OPTD_1C_all" * ".jld2" ε_comparison_threshold utility_D  utility_mean_D θsim_save_all discrepancy_all  T_design D N  T J
ε_comparison_threshold_lowr = copy(ε_comparison_threshold);
θsim_save_all_lowr = copy(θsim_save_all);
17.15562541694817
utility_mean_D_lowr = copy(utility_mean_D);

# high r
save_name = "highr"
filepath_save = pwd() * "/" * "Results" * "/" * "Files_ExpDesign" * save_name * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist
@load filepath_save * "OPTD_1C_all" * ".jld2" ε_comparison_threshold utility_D  utility_mean_D θsim_save_all discrepancy_all  T_design D N  T J
ε_comparison_threshold_highr = copy(ε_comparison_threshold);
θsim_save_all_highr = copy(θsim_save_all);
34.57436991550276
utility_mean_D_highr = copy(utility_mean_D);

##############################################################
## 4 - Setup location to save files
##############################################################

filepathseparator = "/";
filepath_save = pwd() * "/" * "Results" * "/" * "Files_OPTD" * "_Rankings" * "/"; # location to save figures 
isdir(filepath_save) || mkdir(filepath_save); # make folder to save figures if doesnt already exist

##############################################################
## 5 - Plot distribution of θsim_save_all (check sampling of uniform priors)
##############################################################

### Plot prior
for j=1:3
    if j==1
        θsim_save_all = copy(θsim_save_all_lowr)
    elseif j==2 
        θsim_save_all = copy(θsim_save_all_intr)
    elseif j==3
        θsim_save_all = copy(θsim_save_all_highr)
    end
    fig_density_μr_prior = plot(layout=grid(1,1))
    fig_density_σr_prior = plot(layout=grid(1,1)) 
    fig_density_μK_prior = plot(layout=grid(1,1)) 
    fig_density_σK_prior = plot(layout=grid(1,1)) 
    histogram!(fig_density_μr_prior,[θsim_save_all[i][1] for i=1:length(θsim_save_all)],lw=3,legend=false)
    histogram!(fig_density_σr_prior,[θsim_save_all[i][2] for i=1:length(θsim_save_all)],lw=3,legend=false)
    histogram!(fig_density_μK_prior,[θsim_save_all[i][3] for i=1:length(θsim_save_all)],lw=3,legend=false)
    histogram!(fig_density_σK_prior,[θsim_save_all[i][4] for i=1:length(θsim_save_all)],lw=3,legend=false)
    display(fig_density_μr_prior)
    display(fig_density_σr_prior)
    display(fig_density_μK_prior)
    display(fig_density_σK_prior)
end

##############################################################
## 6 - Compute average ranking across low r, intermediate r, and high r
##############################################################

function rank_vector(v)
    # Sort the indices of the vector
    sorted_indices = sortperm(v,rev=true)
    # Map the indices back to ranks
    ranks = similar(v)
    for rank in 1:length(sorted_indices)
        ranks[sorted_indices[rank]] = rank
    end
    return ranks
end

function rank_vector_revfalse(v)
    # Sort the indices of the vector
    sorted_indices = sortperm(v,rev=false)
    # Map the indices back to ranks
    ranks = similar(v)
    for rank in 1:length(sorted_indices)
        ranks[sorted_indices[rank]] = rank
    end
    return ranks
end

# Rank the designs for low, intermediate, and high r 
utility_mean_D_lowr_rank = rank_vector(utility_mean_D_lowr);
utility_mean_D_intr_rank = rank_vector(utility_mean_D_intr);
utility_mean_D_highr_rank = rank_vector(utility_mean_D_highr);

# Combine the ranks into a matrix
ranks_matrix = hcat(utility_mean_D_lowr_rank, utility_mean_D_intr_rank, utility_mean_D_highr_rank)

# Compute an average rank (using the ranks from low, intermediate, and high r))
average_rank = mean(ranks_matrix, dims=2)[:];

# Compute an overall rank (by ranking the average ranks obtained across the low, intermediate, and high r designs)
overall_rank = rank_vector_revfalse(average_rank);

##############################################################
## 7 - Specify/identify times for selected designs 
# (early, middle, late, equalspace, expdata, bestforlowr, bestforintr, bestforhighr, overall)
##############################################################

# Specify times for selected designs (early, middle, late, equalspace,expdata)
t_early = [0.5;1.0;2.0;4.0;6.0;8.0];
t_middle =[6.0;8.0;10.0;12.0;14.0;16.0];
t_late = [14.0;16.0;18.0;20.0;22.0;24.0];
t_equalspace = [4.0;8.0;12.0;16.0;20.0;24.0];
t_expdata = [1.0,2.0,4.0,8.0,16.0,24.0];

# Identify index of selected designs (early, middle, late, equalspace,expdata) and check times are correct
T_design_early = 1;
T_design_middle = findfirst(T_design .== [t_middle]);
T_design_late = length(T_design);
T_design_equalspace = findfirst(T_design .== [t_equalspace]);
T_design_expdata = findfirst(T_design .== [t_expdata]);
T_design[T_design_early]
T_design[T_design_middle]
T_design[T_design_late]
T_design[T_design_equalspace]
T_design[T_design_expdata]

# Identify times for selected designs (bestforlowr, bestforintr, bestforhighr, overall)
T_design_bestforlowr = findfirst(maximum(utility_mean_D_lowr) .== utility_mean_D_lowr);  
T_design_bestforintr = findfirst(maximum(utility_mean_D_intr) .== utility_mean_D_intr); 
T_design_bestforhighr = findfirst(maximum(utility_mean_D_highr) .== utility_mean_D_highr)
T_design_overall = findfirst(overall_rank .== 1); 

# View time for selected designs (early, middle, late, equalspace,expdata) and check times are correct
T_design[T_design_bestforlowr]
T_design[T_design_bestforintr]
T_design[T_design_bestforhighr]
T_design[T_design_overall]

##############################################################
## 8 - Plot utility (within condition) - bar and box plots
##############################################################

# low_r
fig_utility_ABCerror_lowr=plot(layout=grid(1,1),xlab="Design",ylab=L"\hat{U}(w)",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlim=[0,11.5],xticks=false,size=fig_size_3across,frame=:box);
boxplot!(fig_utility_ABCerror_lowr,[10.5],utility_mean_D_lowr./maximum(utility_mean_D_lowr),msw=0,ylim=[-0.05,1.05],yticks=[0,0.5,1.0],color=:gray)
bar!(fig_utility_ABCerror_lowr,[1],[utility_mean_D_lowr[T_design_early]./maximum(utility_mean_D_lowr)],msw=0,color=color_early,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[2],[utility_mean_D_lowr[T_design_middle]./maximum(utility_mean_D_lowr)],msw=0,color=color_middle,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[3],[utility_mean_D_lowr[T_design_late]./maximum(utility_mean_D_lowr)],msw=0,color=color_late,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[4],[utility_mean_D_lowr[T_design_equalspace]./maximum(utility_mean_D_lowr)],msw=0,color=color_equalspace,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[5],[utility_mean_D_lowr[T_design_expdata]./maximum(utility_mean_D_lowr)],msw=0,color=color_expdata,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[6],[utility_mean_D_lowr[T_design_bestforlowr]./maximum(utility_mean_D_lowr)],msw=0,color=color_bestforlowr,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[7],[utility_mean_D_lowr[T_design_bestforintr]./maximum(utility_mean_D_lowr)],msw=0,color=color_bestforintr,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[8],[utility_mean_D_lowr[T_design_bestforhighr]./maximum(utility_mean_D_lowr)],msw=0,color=color_bestforhighr,markersize=7.5)
bar!(fig_utility_ABCerror_lowr,[9],[utility_mean_D_lowr[T_design_overall]./maximum(utility_mean_D_lowr)],msw=0,color=color_overall_rank,markersize=7.5)
display(fig_utility_ABCerror_lowr)
savefig(fig_utility_ABCerror_lowr,filepath_save * "fig_utility_ABCerror_lowr" * ".pdf")


# int_r
fig_utility_ABCerror_intr=plot(layout=grid(1,1),xlab="Design",ylab=L"\hat{U}(w)",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlim=[0,11.5],xticks=false,size=fig_size_3across,frame=:box);
boxplot!(fig_utility_ABCerror_intr,[10.5],utility_mean_D_intr./maximum(utility_mean_D_intr),msw=0,ylim=[-0.05,1.05],yticks=[0,0.5,1.0],color=:gray)
bar!(fig_utility_ABCerror_intr,[1],[utility_mean_D_intr[T_design_early]./maximum(utility_mean_D_intr)],msw=0,color=color_early,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[2],[utility_mean_D_intr[T_design_middle]./maximum(utility_mean_D_intr)],msw=0,color=color_middle,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[3],[utility_mean_D_intr[T_design_late]./maximum(utility_mean_D_intr)],msw=0,color=color_late,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[4],[utility_mean_D_intr[T_design_equalspace]./maximum(utility_mean_D_intr)],msw=0,color=color_equalspace,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[5],[utility_mean_D_intr[T_design_expdata]./maximum(utility_mean_D_intr)],msw=0,color=color_expdata,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[6],[utility_mean_D_intr[T_design_bestforlowr]./maximum(utility_mean_D_intr)],msw=0,color=color_bestforlowr,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[7],[utility_mean_D_intr[T_design_bestforintr]./maximum(utility_mean_D_intr)],msw=0,color=color_bestforintr,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[8],[utility_mean_D_intr[T_design_bestforhighr]./maximum(utility_mean_D_intr)],msw=0,color=color_bestforhighr,markersize=7.5)
bar!(fig_utility_ABCerror_intr,[9],[utility_mean_D_intr[T_design_overall]./maximum(utility_mean_D_intr)],msw=0,color=color_overall_rank,markersize=7.5)
display(fig_utility_ABCerror_intr)
savefig(fig_utility_ABCerror_intr,filepath_save * "fig_utility_ABCerror_intr" * ".pdf")


# high_r
fig_utility_ABCerror_highr=plot(layout=grid(1,1),xlab="Design",ylab=L"\hat{U}(w)",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlim=[0.0,11.5],xticks=false,size=fig_size_3across,frame=:box);
boxplot!(fig_utility_ABCerror_highr,[10.5],utility_mean_D_highr./maximum(utility_mean_D_highr),msw=0,ylim=[-0.05,1.05],yticks=[0,0.5,1.0],color=:gray)
bar!(fig_utility_ABCerror_highr,[1],[utility_mean_D_highr[T_design_early]./maximum(utility_mean_D_highr)],msw=0,color=color_early,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[2],[utility_mean_D_highr[T_design_middle]./maximum(utility_mean_D_highr)],msw=0,color=color_middle,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[3],[utility_mean_D_highr[T_design_late]./maximum(utility_mean_D_highr)],msw=0,color=color_late,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[4],[utility_mean_D_highr[T_design_equalspace]./maximum(utility_mean_D_highr)],msw=0,color=color_equalspace,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[5],[utility_mean_D_highr[T_design_expdata]./maximum(utility_mean_D_highr)],msw=0,color=color_expdata,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[6],[utility_mean_D_highr[T_design_bestforlowr]./maximum(utility_mean_D_highr)],msw=0,color=color_bestforlowr,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[7],[utility_mean_D_highr[T_design_bestforintr]./maximum(utility_mean_D_highr)],msw=0,color=color_bestforintr,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[8],[utility_mean_D_highr[T_design_bestforhighr]./maximum(utility_mean_D_highr)],msw=0,color=color_bestforhighr,markersize=7.5)
bar!(fig_utility_ABCerror_highr,[9],[utility_mean_D_highr[T_design_overall]./maximum(utility_mean_D_highr)],msw=0,color=color_overall_rank,markersize=7.5)
display(fig_utility_ABCerror_highr)
savefig(fig_utility_ABCerror_highr,filepath_save * "fig_utility_ABCerror_highr" * ".pdf")


##############################################################
## 9 - For the top 100 designs, for each of the time points plot a histogram
##############################################################

# Identify the top 100 design properties
D_topN = 100;

D_top100_idxs_lowr = partialsortperm(utility_mean_D_lowr, 1:D_topN, rev=true)
D_top100_idxs_intr = partialsortperm(utility_mean_D_intr, 1:D_topN, rev=true)
D_top100_idxs_highr = partialsortperm(utility_mean_D_highr, 1:D_topN, rev=true)
D_top100_idxs_overall_rank = partialsortperm(average_rank, 1:D_topN, rev=false)

# For each of the time points plot a histogram
for ta=1:length(T_design[1])
    println(ta)

    # compute best 100
    tmp_best100_t_lowr = zeros(D_topN);
    tmp_best100_t_intr = zeros(D_topN);
    tmp_best100_t_highr = zeros(D_topN);
    tmp_best100_t_overall_rank = zeros(D_topN);
    for i=1:D_topN
        tmp_best100_t_lowr[i] = T_design[D_top100_idxs_lowr[i]][ta];
        tmp_best100_t_intr[i] = T_design[D_top100_idxs_intr[i]][ta];
        tmp_best100_t_highr[i] = T_design[D_top100_idxs_highr[i]][ta];
        tmp_best100_t_overall_rank[i] = T_design[D_top100_idxs_overall_rank[i]][ta];
    end

    #  data
    data1 = tmp_best100_t_lowr;
    data2 = tmp_best100_t_intr;
    data3 = tmp_best100_t_highr;
    data4 = tmp_best100_t_overall_rank;

    #############
    # Determine the common bin edges for all datasets
    edges = 0.0:0.5:24.5;

    # Calculate histograms using the common bin edges
    h1 = fit(Histogram, data1, edges)
    h2 = fit(Histogram, data2, edges)
    h3 = fit(Histogram, data3, edges)
    h4 = fit(Histogram, data4, edges)

    # Shift the x positions slightly for side-by-side bars
    x_vals = h1.edges[1][1:end-1]  # The x values (bin centers)
    width = (x_vals[2] - x_vals[1]) / 4  # Bar width adjustment for side-by-side

    ############# Rescaled y - so all are equal size and less empty space
    t_show = [0.5;1.0;2.0:2.0:24];
    t_show_index_tmp = zeros(length(t_show))
    for i=1:length(t_show)
        t_show_index_tmp[i] = Int(findfirst(x -> x == t_show[i], x_vals));
    end
    t_show_index = Int.(t_show_index_tmp);

    x_vals_2 = 1:1:length(t_show)
    width2 = (x_vals_2[2] - x_vals_2[1]) / 5  # Bar width adjustment for side-by-side

    h1_weights_2 = h1.weights[t_show_index];
    h2_weights_2 = h2.weights[t_show_index];
    h3_weights_2 = h3.weights[t_show_index];
    h4_weights_2 = h4.weights[t_show_index];

    if ta== 1
        fig_hist_designtime = plot(layout=grid(1,1),ylab="",xlab="",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,14.5],yticks=([1:1:14;],t_show),size=(26.0*mm_to_pts_scaling,100*mm_to_pts_scaling),xticks=[0,100],xlim=[0,101],frame=:box)
        bar!(fig_hist_designtime, x_vals_2 .- 2*width2,h1_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforlowr)
        bar!(fig_hist_designtime,x_vals_2 .- 1*width2,h2_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforintr)
        bar!(fig_hist_designtime,x_vals_2 .+ 0*width2,h3_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforhighr)
        bar!(fig_hist_designtime,x_vals_2 .+ 1*width2,h4_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_overall_rank)
        yflip!(true)
        display(fig_hist_designtime)
        savefig(fig_hist_designtime,filepath_save  * "fig_hist_designtime_t" * string(ta) * "_tlabel" * ".pdf")
    end

    fig_hist_designtime = plot(layout=grid(1,1),ylab="",xlab="",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,ylim=[0,14.5],yticks=([1:1:14;],["","","","","","","","","","","","","",""]),size=(26.0*mm_to_pts_scaling,100*mm_to_pts_scaling),xticks=[0,100],xlim=[0,101],frame=:box)
    bar!(fig_hist_designtime, x_vals_2 .- 2*width2,h1_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforlowr)
    bar!(fig_hist_designtime,x_vals_2 .- 1*width2,h2_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforintr)
    bar!(fig_hist_designtime,x_vals_2 .+ 0*width2,h3_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_bestforhighr)
    bar!(fig_hist_designtime,x_vals_2 .+ 1*width2,h4_weights_2,bar_width=width2,lw=1, orientation=:h,color=color_overall_rank)
    yflip!(true)
    display(fig_hist_designtime)
    savefig(fig_hist_designtime,filepath_save * "fig_hist_designtime_t" * string(ta) * ".pdf")

end

##############################################################
## 10 -  Plot - Design times schematic
##############################################################
        
fig_designtimes = plot(layout=grid(1,1),xlab=L"t \ [\mathrm{hr}]",ylab="Design",legend=false,titlefont=fnt, guidefont=fnt, tickfont=fnt,xlim=[0,24.5],xticks=[0:2:24;],size=fig_size_1across,yticks = [],ylim=[-1.0,9.0],frame=:box)
scatter!(fig_designtimes,T_design[T_design_early],8.5*ones(6),color = color_early,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_middle],7.5*ones(6),color =color_middle,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_late],6.5*ones(6),color =color_late,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_equalspace],5.5*ones(6),color =color_equalspace,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_expdata],4.5*ones(6),color =color_expdata,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_bestforlowr],3*ones(6),color =color_bestforlowr,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_bestforintr],2*ones(6),color =color_bestforintr,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_bestforhighr],1*ones(6),color =color_bestforhighr,msw=1,markersize=10)
scatter!(fig_designtimes,T_design[T_design_overall],0*ones(6),color =color_overall_rank,msw=1,markersize=10)
hline!(fig_designtimes,[3.75],color=:black,lw=2)
display(fig_designtimes)
savefig(fig_designtimes,filepath_save * "fig_designtimes" * ".pdf")

##############################################################
## 11 - For each selected design view overall rank
##############################################################

# View overall rank of selected designs
overall_rank[T_design_early] # 2838
overall_rank[T_design_middle] # 2983
overall_rank[T_design_late] # 2530
overall_rank[T_design_equalspace] # 1585
overall_rank[T_design_expdata] # 291
overall_rank[T_design_bestforlowr] # 2530
overall_rank[T_design_bestforintr] # 86
overall_rank[T_design_bestforhighr] # 8
overall_rank[T_design_overall] # 1

##############################################################
## 12 - For each selected design view individual ranks for the three scenarios
##############################################################

# View individual ranks for the three scenarios
ranks_matrix[T_design_early,:]
# 3003.0
# 2528.0
# 1012.0
ranks_matrix[T_design_middle,:]
# 1635.0
# 2917.0
# 2908.0
ranks_matrix[T_design_late,:]
# 1.0
# 3000.0
# 2994.0
ranks_matrix[T_design_equalspace,:]
# 188.0
# 2495.0
# 1953.0
ranks_matrix[T_design_expdata,:]
# 2131.0
# 34.0
# 485.0
ranks_matrix[T_design_bestforlowr,:]
# 1.0
# 3000.0
# 2994.0
ranks_matrix[T_design_bestforintr,:]
# 814.0
# 1.0
# 1060.0
ranks_matrix[T_design_bestforhighr,:]
# 1009.0
# 82.0
#  1.0
ranks_matrix[T_design_overall,:]
# 389.0
# 214.0
#  12.0

##############################################################
## 13 - For each selected design view average of the three ranks
##############################################################

# View average rank
mean(ranks_matrix[T_design_early,:]) # 2181
mean(ranks_matrix[T_design_middle,:]) #2487
mean(ranks_matrix[T_design_late,:]) #1998
mean(ranks_matrix[T_design_equalspace,:]) #1545
mean(ranks_matrix[T_design_expdata,:]) #883
mean(ranks_matrix[T_design_bestforlowr,:]) #1998
mean(ranks_matrix[T_design_bestforintr,:]) #625
mean(ranks_matrix[T_design_bestforhighr,:]) #364
mean(ranks_matrix[T_design_overall,:]) #205

average_rank[T_design_overall]
average_rank[T_design_expdata]

##############################################################
## 14 - For each selected design view Normalised mean utilities
##############################################################

# low_r
round(utility_mean_D_lowr[T_design_early]./maximum(utility_mean_D_lowr),digits=3)# 0.018
round(utility_mean_D_lowr[T_design_middle]./maximum(utility_mean_D_lowr),digits=3) #0.282
round(utility_mean_D_lowr[T_design_late]./maximum(utility_mean_D_lowr),digits=3)# 1.0
round(utility_mean_D_lowr[T_design_equalspace]./maximum(utility_mean_D_lowr),digits=3) #0.653
round(utility_mean_D_lowr[T_design_expdata]./maximum(utility_mean_D_lowr),digits=3) #0.206
round(utility_mean_D_lowr[T_design_bestforlowr]./maximum(utility_mean_D_lowr),digits=3) #1.0
round(utility_mean_D_lowr[T_design_bestforintr]./maximum(utility_mean_D_lowr),digits=3) #0.434
round(utility_mean_D_lowr[T_design_bestforhighr]./maximum(utility_mean_D_lowr),digits=3) #0.39
round(utility_mean_D_lowr[T_design_overall]./maximum(utility_mean_D_lowr),digits=3) #0.553

# int_r
round(utility_mean_D_intr[T_design_early]./maximum(utility_mean_D_intr),digits=3) #0.241
round(utility_mean_D_intr[T_design_middle]./maximum(utility_mean_D_intr),digits=3) #0.034
round(utility_mean_D_intr[T_design_late]./maximum(utility_mean_D_intr),digits=3) #0.006
round(utility_mean_D_intr[T_design_equalspace]./maximum(utility_mean_D_intr),digits=3) #0.252
round(utility_mean_D_intr[T_design_expdata]./maximum(utility_mean_D_intr),digits=3) #0.812
round(utility_mean_D_intr[T_design_bestforlowr]./maximum(utility_mean_D_intr),digits=3) #0.006
round(utility_mean_D_intr[T_design_bestforintr]./maximum(utility_mean_D_intr),digits=3) #1
round(utility_mean_D_intr[T_design_bestforhighr]./maximum(utility_mean_D_intr),digits=3) #0.775
round(utility_mean_D_intr[T_design_overall]./maximum(utility_mean_D_intr),digits=3)# 0.696

# high_r
round(utility_mean_D_highr[T_design_early]./maximum(utility_mean_D_highr),digits=3) #0.753
round(utility_mean_D_highr[T_design_middle]./maximum(utility_mean_D_highr),digits=3) #0.516
round(utility_mean_D_highr[T_design_late]./maximum(utility_mean_D_highr),digits=3) #0.443
round(utility_mean_D_highr[T_design_equalspace]./maximum(utility_mean_D_highr),digits=3) #0.674
round(utility_mean_D_highr[T_design_expdata]./maximum(utility_mean_D_highr),digits=3)# 0.807
round(utility_mean_D_highr[T_design_bestforlowr]./maximum(utility_mean_D_highr),digits=3) #0.443
round(utility_mean_D_highr[T_design_bestforintr]./maximum(utility_mean_D_highr),digits=3) #0.749
round(utility_mean_D_highr[T_design_bestforhighr]./maximum(utility_mean_D_highr),digits=3) #1
round(utility_mean_D_highr[T_design_overall]./maximum(utility_mean_D_highr),digits=3) #0.946


# ##############################################################
# ## 15 - Minimum normalised mean utility for each scenario 
# ##############################################################

round(minimum(utility_mean_D_lowr./maximum(utility_mean_D_lowr)),digits=3) # 0.018
round(minimum(utility_mean_D_intr./maximum(utility_mean_D_intr)),digits=3) # 0.005
round(minimum(utility_mean_D_highr./maximum(utility_mean_D_highr)),digits=3) # 0.41
