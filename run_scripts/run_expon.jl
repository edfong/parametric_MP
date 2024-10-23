using Distributions
using Plots,StatsPlots
using DelimitedFiles
using Random

Random.seed!(61222)

scale_truth = 1
n = 10
y = rand(Exponential(scale_truth), n)

scale_n = mean(y)

B = 50000
T = 20000 -n
theta_n = scale_n*ones(B)

@time begin
for N in n+1:n+T
    Y_N = @. rand(Exponential(theta_n))
    theta_n .= theta_n + (1/N)*(Y_N - theta_n)
end
end
density(theta_n,label ="Exact")

# Gaussian approx
std_n = sqrt((scale_n^2))*sqrt(((pi^2/6)-sum(@. 1/(1:n)^2)))

theta_n_normal = rand(Normal(scale_n, std_n),B)
density!(theta_n_normal,label = "Gaussian")


# Hybrid
T = 30-n
theta_n_trunc = scale_n*ones(B)
@time begin
for N in n+1:n+T
    Y_N = @. rand(Exponential(theta_n_trunc))
    theta_n_trunc .= theta_n_trunc + (1/N)*(Y_N - theta_n_trunc)
end
end
std_n_hybrid = sqrt.(((pi^2/6)-sum(@. 1/(1:n+T)^2)).*(theta_n_trunc.^2))

theta_n_hybrid = @. rand(Normal(theta_n_trunc, std_n_hybrid))
density!(theta_n_trunc, label = "Truncated")
density!(theta_n_hybrid,label = "Hybrid")


writedlm( "../results/theta_n_trunc.csv",  theta_n_trunc, ',')
writedlm( "../results/theta_n_hybrid.csv",  theta_n_hybrid, ',')
writedlm( "../results/theta_n_normal.csv",  theta_n_normal, ',')
writedlm( "../results/theta_n_exact.csv",  theta_n, ',')