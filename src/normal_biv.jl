using Distributions
using Plots,StatsPlots
using DelimitedFiles
using Random
using Statistics
using BenchmarkTools
using LinearAlgebra

# Sim Setup
function sim_data(theta_true,n,seed)
    Random.seed!(seed)
    #Truth
    mu_true = theta_true[1:2]
    s1_true = theta_true[3]
    s2_true = theta_true[4]
    s12_true = theta_true[5]
    Sigma_true = [s1_true s12_true; s12_true s2_true] #random covariance matrix
    y = rand(MvNormal(mu_true, Sigma_true), n)
    return y
end


# PR
@views begin
function pr_sample_B(theta,n,T,B,seed) #theta = (m1,m2,s1,s2,s12)
    Random.seed!(seed)
    #Initialize
    theta_B = theta .* ones(5,B)

    Y1 = zeros(B)
    Y2 = zeros(B)
    z1 = rand(Normal(0,1),(T,B))
    z2 = rand(Normal(0,1),(T,B))

    #Scale learning rate?
    #a = (1/sqrt(n))/sqrt(((pi^2/6)-sum(@. 1/(1:n)^2)))
    a = 1.

    @. begin
    for N in (n+1):(n+T)
        #Simulate MV gaussian
         Y2 = (theta_B[5,:]/sqrt(theta_B[3,:]))*z1[N-n,:] + sqrt(theta_B[4,:]-(theta_B[5,:]^2/theta_B[3,:]))*z2[N-n,:]
         Y1 = sqrt(theta_B[3,:])*z1[N-n,:] 
         Y1 += theta_B[1,:]
         Y2 += theta_B[2,:]

        # Update variances first
         theta_B[3,:] .+=  (a/N)*((Y1 - theta_B[1,:])^2- theta_B[3,:])   
         theta_B[4,:] .+= (a/N)*((Y2 - theta_B[2,:])^2- theta_B[4,:])  
         theta_B[5,:] .+=  (a/N)*((Y1 - theta_B[1,:])*(Y2- theta_B[2,:])- theta_B[5,:])    
        # Update means after variances!!
         theta_B[1,:] +=   (a/N)*(Y1 - theta_B[1,:]) 
         theta_B[2,:] +=  (a/N)*(Y2 - theta_B[2,:]) 
    end
    end
    return theta_B
end 
end

@views begin
function calc_FIM_inv_chol(s1,s2,s12)
    Sigma_chol = cholesky!(([s1 s12; s12 s2])).L
    J_chol = cholesky!((2*[s1^2 s12^2 s1*s12; s12^2 s2^2 s2*s12;s1*s12 s2*s12 0.5*(s12^2 + s1*s2)])).L
    return Sigma_chol,J_chol
end
end

@views begin #theta = (m1,m2,s1,s2,s12)
function add_tail(theta_B,n,T) #allocations less importance as just a single loop
    B = size(theta_B)[2]
    global p = 2
    global M = Int.(p*(p+1)/2)
    z1 = rand(Normal(0,1),(B,p))
    z2 = rand(Normal(0,1),(B,M))

    #Scaling
    r = sqrt(((pi^2/6)-sum(@. 1/(1:n+T)^2)))
    #r = 1/sqrt(n+T)

    for i in 1:B
        S,J =  calc_FIM_inv_chol(theta_B[3,i],theta_B[4,i],theta_B[5,i])
        lmul!(r,S)
        lmul!(r,J)
        theta_B[1:p,i] .+= S*z1[i,:]
        theta_B[p+1:p+M,i] .+= J*z2[i,:]
    end
end
end
