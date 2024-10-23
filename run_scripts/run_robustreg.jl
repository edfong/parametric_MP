include("../src/robustreg.jl")
using ProgressBars


#Load data
data =  readdlm( "../data/AIDS.csv",',',header = true)
colnames = data[2]
y = data[1][:,2]
X = data[1][:,3:end]

n = size(y)[1]
p = size(X)[2]
#Add intercept!
X = hcat(ones(n,1),X)

#Robust estimation
df = 5
@time begin
    theta_mle = fit_treg_restart(y,X,df,10)
end


#Setup sampling
B = 10000
T = 50000
T_trunc = 100

@time begin
    theta_B = pr_sample_B(theta_mle,df,X,T,B,383)
end

@time begin
    theta_B_trunc = pr_sample_B(theta_mle,df,X,T_trunc,B,383)
end

@time begin
    theta_B_hybrid = copy(theta_B_trunc)
    add_tail(theta_B_hybrid,df,X,T_trunc) 
end


writedlm( "../results/MP_full_aids.csv",theta_B', ',') 
writedlm( "../results/MP_trunc_aids.csv",theta_B_trunc', ',') 
writedlm( "../results/MP_hybrid_aids.csv",theta_B_hybrid', ',') 