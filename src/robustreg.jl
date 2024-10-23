using Distributions
using Plots,StatsPlots
using DelimitedFiles
using Random
using Statistics
using BenchmarkTools
using LinearAlgebra
using StatsFuns

# PR
@views begin
    function pr_sample_B(theta,df,Xn,T,B,seed) #theta = (beta,tau^2)
        Random.seed!(seed)
        #Initialize
        d = size(theta)[1]
        n = size(Xn)[1]
        theta_B = theta .* ones(d,B)

        #Draw random samples
        R = rand(TDist(df),(B,T))  #Standard T variables
        ind_X = sample(1:n,(B,T)) #Choose random indices
    
        #Calculate covariate matrix terms
        #Z = inv((Xn'*Xn)/n)*(Xn') #This is Sigma^{-1}X_n, where each column is each index
        Z = ((Xn'*Xn)./n)\Xn' #This is Sigma^{-1}X_n, where each column is each index

        for N in (n+1):(n+T)
            # Update betas (how to do the transpose step without)
            for j in 1:B # 0.0055, 20 allocations
                theta_B[1:d-1,j] .+=  (1/N).*((sqrt(theta_B[d,j]) *(df + 3)*R[j,N-n])/(df + R[j,N-n]^2)).*(Z[:,ind_X[j,N-n]]) 
            end
         
            # Update variances
            theta_B[d,:] .+=  (1/N).*(theta_B[d,:] .*(df + 3).*(R[:,N-n].^2 .-1) )./(df .+ R[:,N-n].^2)
        end
        return theta_B 
        end
    end


@views begin #theta = (beta,tau)
    function add_tail(theta_B,df,Xn,T) #allocations less importance as just a single loop
        #Initialize
        B = size(theta_B)[2]
        d = size(theta_B)[1]
        n = size(Xn)[1]
        p = d-1
        z1 = rand(Normal(0,1),(p,B))
        z1_cov = zeros(p,B)
        z2 = rand(Normal(0,1),(B))
     
        #Scaling
        r = sqrt(((pi^2/6)-sum(@. 1/(1:n+T)^2)))
        Sigma_inv_chol = cholesky(Symmetric(diagm(ones(p))/((Xn'*Xn)./n))).L
        mul!(z1_cov,Sigma_inv_chol,z1) #generate normal r.v.s with covariance \Sigma_{n,x}^{-1/2}
        
        for i in 1:B
            theta_B[1:p,i] .+= r.*sqrt(((df+ 3)*(theta_B[p+1,i]))/(df + 1)).*z1_cov[:,i]
            theta_B[p+1,i] += r*sqrt(2*(theta_B[p+1,i]^2)*(df + 3)/df)*z2[i]
        end
    end
end

#MLE for student-t
@views begin 
function fit_treg(y,X,df,seed)
    #Optimzer setings
    max_iter = 500
    xrtol = 1e-5

    #Setup
    n = size(X)[1]
    p = size(X)[2]
    R = zeros(n)

    #Precompute terms
    Z = ((X'*X)./n)\X'
    
    #Random Initialize 
    Random.seed!(seed)
    theta = rand(Normal(0,1),(p+1))
    theta[p+1] = rand(Exponential(2))
    step = zeros(p+1)
    
    for j in 1:max_iter
        mul!(R,X,theta[1:p])
        R .= (y - R)/sqrt(theta[p+1])
        # Update betas (how to do the transpose step without)
        step .= 0
        for i in 1:n
            step[1:p] .+=  (1/n).*((sqrt(theta[p+1])*(df + 3)*R[i])/(df + R[i]^2)).*(Z[:,i]) 
            step[p+1] +=  (1/n)*((theta[p+1] *(df + 3)*(R[i]^2 -1) )/(df + R[i]^2))
        end
        
        # Update variances
        theta .+=  step
        theta[p+1] = max(theta[p+1],1e-6) #clip variance to be positive

        # Termination criterion
        if (norm(step) < norm(theta)*xrtol)
            break
        end
    end
    mul!(R,X,theta[1:p])
    R .= (y - R)/sqrt(theta[p+1])
    opt_loglik = mean(logpdf(TDist(df), R))
    return theta,opt_loglik
end
end

function fit_treg_restart(y,X,df,n_restart) #n_restart is number of repeats
seed = 38133
#Setup
p = size(X)[2]
theta_rest = zeros(p+1,n_restart)
opt_loglik_rest = zeros(n_restart)
    for i in 1:n_restart
        theta_rest[:,i],opt_loglik_rest[i] = fit_treg(y,X,df,seed + i)
    end
    return theta_rest[:,argmax(opt_loglik_rest)]
end



