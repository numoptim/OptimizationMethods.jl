using MWE, OptimizationProblems, Random
Random.seed!(2026)
p=LogisticRegression(Float64;num_param=10,num_obs=100)
m=FixedStepGD(step_size=0.01)
s=allocate(p)
allocate!(s,m,p;x0=zeros(10),history=History())
optimize!(m,p,s;stop=StoppingCriteria(max_iterations=3))
for (k,x,g) in zip(s[:history][:iter],s[:history][:x],s[:history][:grad_norm])
    println("iteration=$k, x[1]=$(x[1]), gradient norm=$g")
end
println("stopped: ",s[:stop_reason])
