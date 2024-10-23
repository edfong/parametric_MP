using Turing
using DelimitedFiles
using Random
using LinearAlgebra
using Distributions


#global TDist(μ, σ, ν) = μ + TDist(ν)*σ

#Setup MCMC
@model function t_regression(x,y,df, sigma0)
    # priors
    n = size(y)[1]
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(zeros(nfeatures), sigma0^2 * I)
    σ ~ truncated(Cauchy(0,5),lower = 1e-4)

    for i in 1:n
        v = (dot(x[i,:],coefficients))
        y[i] ~ σ*TDist(df) + v
        #y[i] ~ TDist(v,σ,df)
    end
end


#Load data
data =  readdlm( "../data/AIDS.csv",',',header = true)
colnames = data[2]
y = data[1][:,2]
X = data[1][:,3:end]

n = size(y)[1]
p = size(X)[2]
#Add intercept!
X = hcat(ones(n,1),X)


B = 10000
df = 5
Random.seed!(381)
@time begin
p_noninform = sample(t_regression(X,y,5.,10.),NUTS(),B)
end
print(mean(p_noninform["σ"]))
print(mean(p_noninform["coefficients[1]"]))

p = size(X)[2]
theta_Bayes = zeros(B,p+1)  
for i in 1:(p)
    theta_Bayes[:,i] = p_noninform["coefficients[$(i)]"]
end
theta_Bayes[:,p+1] = p_noninform["σ"].^2

writedlm( "../results/MCMC_aids.csv",theta_Bayes, ',') 