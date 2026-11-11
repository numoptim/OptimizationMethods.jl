using Test, MWE, OptimizationProblems, LinearAlgebra, Random
Random.seed!(2026)
function setup(m; T=Float64, history=nothing)
    p = LogisticRegression(T; num_param=10, num_obs=100)
    s = allocate(p; type=T)
    allocate!(s, m, p; x0=zeros(T,10), history)
    return p,s
end
@testset "MWE" begin
    @testset "References and history" begin
        for m in (FixedStepGD(step_size=0.01), NesterovAcceleratedGD(step_size=0.01), BacktrackingGD(step_size=1.0))
            for every in (1,3)
                p,s = setup(m; history=History(keys=[:x,:grad_norm,:counters], every=every))
                q = deepcopy(p); r = allocate(q); x=zeros(10); y=copy(x); z=copy(x); B=0.; D=1.
                grad!(q; store=r,x)
                expected=[copy(x)]
                for k in 1:5
                    if m isa NesterovAcceleratedGD
                        B += D; x = y - m.step_size*r[:grad]; z -= m.step_size*D*r[:grad]
                        D=(1+sqrt(4B+1))/2; y=x+D/(B+D+1/m.step_size)*(z-x)
                        grad!(q; store=r,x); grad!(q; store=r,x=y)
                    elseif m isa BacktrackingGD
                        obj!(q; store=r,x); f=r[:obj]; a=m.step_size; g=copy(r[:grad])
                        for j in 0:m.max_ls_iterations
                            trial=x-a*g; obj!(q;store=r,x=trial)
                            if r[:obj] <= f-m.rho*a*dot(g,g)
                                x=trial; break
                            end
                            a*=m.delta
                        end
                        grad!(q;store=r,x)
                    else
                        x -= m.step_size*r[:grad]; grad!(q;store=r,x)
                    end
                    push!(expected,copy(x))
                end
                optimize!(m,p,s;stop=StoppingCriteria(grad_norm=0,max_iterations=5))
                @test s[:x] ≈ x
                @test s[:stop_reason] == :max_iterations
                @test s[:iter] == 5
                @test s[:history][:iter] == (every==1 ? collect(0:5) : [0,3,5])
                for (i,k) in enumerate(s[:history][:iter])
                    @test s[:history][:x][i] ≈ expected[k+1]
                end
                @test p.counters[:grad].batch_equivalent == (m isa NesterovAcceleratedGD ? 11 : 6)
                @test p.counters[:grad].block_equivalent == p.counters[:grad].batch_equivalent
                @test s[:history][:counters][1][:grad].batch_equivalent == 1
                @test s[:counters] === p.counters
                @test s[:history][:x][end] !== s[:x]
                grad!(q;store=r,x=s[:x]); @test s[:grad_norm] ≈ norm(r[:grad])
                if m isa NesterovAcceleratedGD
                    @test s[:y] ≈ y
                    @test s[:z] ≈ z
                    @test s[:B] ≈ B
                    grad!(q;store=r,x=y); @test s[:grad] ≈ r[:grad]
                end
            end
        end
    end
    @testset "Stops, types, failure" begin
        m=FixedStepGD(step_size=0.01)
        for (stop,reason,k,count) in ((StoppingCriteria(max_iterations=0),:max_iterations,0,1),
            (StoppingCriteria(grad_norm=1e20),:grad_norm,0,1),
            (StoppingCriteria(max_grad_evals=3,grad_norm=0),:max_grad_evals,2,3),
            (StoppingCriteria(max_grad_evals=0),:max_grad_evals,0,0))
            p,s=setup(m); optimize!(m,p,s;stop)
            @test s[:stop_reason]==reason
            @test s[:iter]==k
            @test p.counters[:grad].batch_equivalent==count
            @test !haskey(s,:history)
            @test p.counters[:obj].batch_equivalent==0
        end
        for T in (Float32,Float64)
            p,s=setup(m;T); optimize!(m,p,s;stop=StoppingCriteria(max_iterations=2))
            @test eltype(s[:x])==T
        end
        m=NesterovAcceleratedGD(step_size=0.01); p,s=setup(m)
        optimize!(m,p,s;stop=StoppingCriteria(max_grad_evals=4,grad_norm=0))
        @test s[:iter]==1
        @test p.counters[:grad].batch_equivalent==3
        m=BacktrackingGD(step_size=1e6,max_ls_iterations=0)
        p,s=setup(m;history=History(every=3)); x=copy(s[:x])
        optimize!(m,p,s)
        @test s[:stop_reason]==:line_search_failed
        @test s[:x]==x
        @test s[:iter]==0
        @test s[:history][:iter]==[0]
        @test p.counters[:grad].batch_equivalent==1
        p,s=setup(m); optimize!(m,p,s;stop=StoppingCriteria(max_obj_evals=1))
        @test s[:stop_reason]==:max_obj_evals
        @test s[:x]==zeros(10)
        @test p.counters[:obj].batch_equivalent==1
        @test_throws ArgumentError History(every=0)
        @test_throws ArgumentError StoppingCriteria(max_iterations=-1)
        @test_throws ArgumentError FixedStepGD(step_size=-1.)
        @test_throws ArgumentError BacktrackingGD(delta=2.)
    end
end

@testset "History independence and allocation validation" begin
    for m in (FixedStepGD(step_size=0.01), BacktrackingGD(), NesterovAcceleratedGD(step_size=0.01))
        p,s=setup(m;history=History(keys=[:x,:counters],every=4))
        q=deepcopy(p); t=allocate(q); allocate!(t,m,q;x0=zeros(10))
        stop=StoppingCriteria(grad_norm=0,max_iterations=7)
        optimize!(m,p,s;stop); optimize!(m,q,t;stop)
        @test s[:x] ≈ t[:x]
        @test p.counters[:grad].batch_equivalent==q.counters[:grad].batch_equivalent
        @test p.counters[:obj].batch_equivalent==q.counters[:obj].batch_equivalent
        @test !haskey(t,:history)
        @test s[:history][:iter]==[0,4,7]
        saved=copy(s[:history][:x][end]); fill!(s[:x],99)
        @test s[:history][:x][end]==saved
        @test s[:history][:counters][1][:grad] !== p.counters[:grad]
        @test_throws ArgumentError allocate!(s,m,p;x0=zeros(10))
    end
    p=LogisticRegression(Float64;num_param=10,num_obs=100)
    @test_throws DimensionMismatch allocate!(allocate(p),FixedStepGD(step_size=.01),p;x0=zeros(9))
    @test_throws ArgumentError allocate!(allocate(p),FixedStepGD(step_size=.01),p;x0=zeros(Float32,10))
end
