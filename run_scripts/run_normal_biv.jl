include("../src/normal_biv.jl")
using ProgressBars


function run_sim_MP(n, rep)
    #Truth
    m1_true = -0.5
    m2_true = 1
    s1_true = 1
    s2_true = 0.5
    s12_true = 0.7
    theta_true = [m1_true,m2_true,s1_true,s2_true,s12_true]

    #Setup simulation
    B = 2000
    coverage = zeros(rep,5)
    int_len = zeros(rep,5)
    sig = 0.05
    seed = 1533

    for i in ProgressBar(1:rep)
        #Generate data
        y = sim_data(theta_true,n,seed+i)

        #Estimate initial parameters and put into theta array
        mu_n = mean(y,dims = 2)
        Sigma_n = cov(y,dims = 2)
        theta = [mu_n[1], mu_n[2],Sigma_n[1,1],Sigma_n[2,2],Sigma_n[1,2]]

        #Truncated sampling 
        T_trunc = 50
        theta_B_trunc = pr_sample_B(theta,n,T_trunc,B,382)
        #Hybrid
        theta_B_hybrid = copy(theta_B_trunc)
        add_tail(theta_B_hybrid,n,T_trunc) 

        #Calculate coverage
        for j in 1:5
            theta_sorted = sort(theta_B_hybrid[j,:])
            lo = theta_sorted[Int(B*sig/2)]
            hi = theta_sorted[Int(B*(1-sig/2))]
            coverage[i,j]= (theta_true[j]>= lo) && (theta_true[j] <= hi)
            int_len[i,j]= hi - lo
        end

    end

    writedlm( "../results/MP_cov_$(n)_$(rep).csv",coverage, ',')
    writedlm( "../results/MP_len_$(n)_$(rep).csv",int_len, ',')
end

rep = 5000
run_sim_MP(20,rep)
run_sim_MP(100,rep)
run_sim_MP(500,rep)

for i in 1:3
    n = [20,100,500][i]
    coverage = readdlm( "../results/MP_cov_$(n)_$(rep).csv",',')
    int_len = readdlm( "../results/MP_len_$(n)_$(rep).csv",',')
    print(mean(coverage[:,[1,3,5]],dims = 1))
    print('\n')
    print(maximum(sqrt.(var(coverage[:,[1,3,5]],dims = 1)/rep)))
    print('\n')

    print(round.(mean(int_len[:,[1,3,5]],dims = 1),digits = 2))
    print('\n')
    print(maximum(sqrt.(var(int_len[:,[1,3,5]],dims = 1)/rep)))
    print('\n')
end