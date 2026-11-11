using MWE, OptimizationProblems, OptimizationModels, BenchmarkTools, LinearAlgebra, Random
struct NoBarrier <: AbstractMethod
    step_size::Float64
end
MWE.allocate!(s,::NoBarrier,p;kwargs...)=allocate_common!(s,p;kwargs...)
function MWE.step!(m::NoBarrier,p,s)
    s[:x] .-= m.step_size .* s[:grad]
    grad!(p;store=s,x=s[:x]); s[:grad_norm]=norm(s[:grad])
end
function run_case(p, mode, k)
    foreach(OptimizationModels.reset!,values(p.counters))
    s=allocate(p)
    m= mode==:no_barrier ? NoBarrier(1e-5) : FixedStepGD(step_size=1e-5)
    h=mode==:history ? History(keys=[:x,:grad_norm]) : nothing
    if mode==:handwritten
        x=zeros(p.num_param); grad!(p;store=s,x)
        for i in 1:k
            x .-= 1e-5 .* s[:grad]
            grad!(p;store=s,x); norm(s[:grad])
        end
        return x
    end
    allocate!(s,m,p;x0=zeros(p.num_param),history=h)
    return optimize!(m,p,s;stop=StoppingCriteria(grad_norm=0,max_iterations=k))
end
Random.seed!(2026)
println("Julia ",VERSION," | ",Sys.MACHINE," | CPU: ",Sys.cpu_info()[1].model," | Julia threads: ",Threads.nthreads()," | BLAS threads: ",BLAS.get_num_threads())
println("20 updates/run; setup, initialization and history included; minimum of 3 samples, evals=1. Per-update values amortize complete run.")
println("| parameters | observations | case | ms/run | μs/update | bytes/run | allocations/run |")
println("|---:|---:|:---|---:|---:|---:|---:|")
for (n,obs) in ((10,100),(1000,10000))
    p=LogisticRegression(Float64;num_param=n,num_obs=obs)
    for mode in (:handwritten,:driver,:history,:no_barrier)
        run_case(p,mode,20)
        trial=@benchmark run_case($p,$mode,20) samples=3 evals=1 seconds=120
        t=minimum(trial)
        println("| $n | $obs | $mode | $(t.time/1e6) | $(t.time/20/1e3) | $(t.memory) | $(t.allocs) |")
    end
end
