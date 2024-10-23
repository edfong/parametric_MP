include("../src/normal_biv.jl")
using ProgressBars
using Turing
using StatsPlots
using LinearAlgebra
using DelimitedFiles

function run_sim_MCMC(n,rep)
    #Truth
    m1_true = -0.5
    m2_true = 1
    s1_true = 1
    s2_true = 0.5
    s12_true = 0.7
    theta_true = [m1_true,m2_true,s1_true,s2_true,s12_true]

    #Setup simulation
    coverage = zeros(rep,5)
    int_len = zeros(rep,5)
    sig = 0.05
    seed = 1533

    #Setup sampler
    B = 2000 #Number of posterior samples
    theta_B = zeros(5,B)

    #Setup MCMC
    @model function biv_norm(x, ::Type{T} = Float64) where {T}
        rows = size(x, 1)
        # priors
        μ = Vector{T}(undef, 2)
        μ[1] ~ Normal(0, 10)
        μ[2] ~ Normal(0, 10)
        σ₁ ~ truncated(Cauchy(0,5),lower = 0)
        σ₂ ~ truncated(Cauchy(0,5),lower = 0)
        r ~ Uniform(-1, 1)
        #r ~ truncated(Normal(0,1),lower = -0.99,upper = 0.99)

        # covariance matrix
        Σ = Symmetric([
            σ₁*σ₁ r*σ₁*σ₂
            r*σ₁*σ₂ σ₂*σ₂
        ])
        # return error if not positive definite
        if !isposdef(Σ)
            Turing.@addlogprob! -Inf
            return 
        end
        for i = 1:rows
            x[i, :] ~ MvNormal(μ, Σ)
        end
    end

    for i in ProgressBar(1:rep)
        #Generate data
        y = sim_data(theta_true,n,seed+i)

        #Run MCMC
        Random.seed!(382)
        p1 = sample(biv_norm(y'),NUTS(),B)
        theta_B[1,:] = p1["μ[1]"]
        theta_B[2,:] = p1["μ[2]"]
        theta_B[3,:] = p1["σ₁"].^2
        theta_B[4,:] = p1["σ₂"].^2
        theta_B[5,:] = p1["r"].*p1["σ₁"].*p1["σ₂"]


        #Calculate coverage
        for j in 1:5
            theta_sorted = sort(theta_B[j,:])
            lo = theta_sorted[Int(B*sig/2)]
            hi = theta_sorted[Int(B*(1-sig/2))]
            coverage[i,j]= (theta_true[j]>= lo) && (theta_true[j] <= hi)
            int_len[i,j]= hi - lo
        end

    end

    writedlm( "../results/MCMC_cov_$(n)_$(rep).csv",coverage, ',')
    writedlm( "../results/MCMC_len_$(n)_$(rep).csv",int_len, ',')
end

rep = 500
run_sim_MCMC(20,rep)
run_sim_MCMC(100,rep)
run_sim_MCMC(500,rep)

rep = 500
for i in 1:3
    n = [20,100,500][i]
    coverage = readdlm( "../results/MCMC_cov_$(n)_$(rep).csv",',')
    int_len = readdlm( "../results/MCMC_len_$(n)_$(rep).csv",',')
    print(mean(coverage[:,[1,3,5]],dims = 1))
    print('\n')
    print(maximum(sqrt.(var(coverage[:,[1,3,5]],dims = 1)/rep)))
    print('\n')

    print(round.(mean(int_len[:,[1,3,5]],dims = 1),digits = 2))
    print('\n')
    print(maximum(sqrt.(var(int_len[:,[1,3,5]],dims = 1)/rep)))
    print('\n')
end