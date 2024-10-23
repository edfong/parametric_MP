include("normal_biv.jl")
using ProgressBars

rep = 5000
for i in 1:3
    n = [20,100,500][i]
    coverage = readdlm( "MP_cov_$(n)_$(rep).csv",',')
    int_len = readdlm( "MP_len_$(n)_$(rep).csv",',')
    print(mean(coverage[:,[2,4]],dims = 1))
    print('\n')
    print(maximum(sqrt.(var(coverage[:,[2,4]],dims = 1)/rep)))
    print('\n')

    print(round.(mean(int_len[:,[2,4]],dims = 1),digits = 2))
    print('\n')
    print(maximum(sqrt.(var(int_len[:,[2,4]],dims = 1)/rep)))
    print('\n')
end



rep = 500
for i in 1:3
    n = [20,100,500][i]
    coverage = readdlm( "MCMC_cov_$(n)_$(rep).csv",',')
    int_len = readdlm( "MCMC_len_$(n)_$(rep).csv",',')
    print(mean(coverage[:,[2,4]],dims = 1))
    print('\n')
    print(maximum(sqrt.(var(coverage[:,[2,4]],dims = 1)/rep)))
    print('\n')

    print(round.(mean(int_len[:,[2,4]],dims = 1),digits = 2))
    print('\n')
    print(maximum(sqrt.(var(int_len[:,[2,4]],dims = 1)/rep)))
    print('\n')
end