
"
This is all test code and is in the works (Ella dissertation)
"

function MulitParamZygote_test(system, ref_sol, prob, params, opt, epoch)

    ground_sol = generate_groundtruth_system(ref_sol)
    tsteps = unique(ground_sol.t)

    p_array, params_idx, state_idx = get_parameters(prob, system, params)

    f_plain    = ODEFunction(prob.f.f)
    prob_plain = ODEProblem(f_plain, copy(prob.u0), prob.tspan, p_array)
    
    truth_vec = []
    for idx in state_idx
        push!(truth_vec, ground_sol(tsteps)[idx, :])
    end
    
    p0 = [p_array[x] for x in params_idx]

    optfn   = OptimizationFunction(loss_with_logging, Optimization.AutoZygote())
    optprob = OptimizationProblem(
        optfn, p0,
        (prob_plain, tsteps, truth_vec, params_idx, state_idx, params),
    )
    if opt=="ADAM"
        sol = solve(optprob, ADAM(0.1); epochs = epoch)
    end
    
    sol.u, sol
end


function loss_with_logging(x, p)
    prob_plain, tsteps, truth_vec, param_idx, state_idx, params = p

    p_new = [i in param_idx ? x[findfirst(==(i), param_idx)] : prob_plain.p[i] 
            for i in eachindex(prob_plain.p)]
    
    newprob = remake(prob_plain; p = p_new)

    sol = solve(
        newprob, Tsit5();
        saveat   = tsteps,
        sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP()))
    
    pred     = sol[state_idx[1], :]
    loss_val = mean(abs2, Array(pred) .- truth_vec[1])

    #test = Dict(zip(params, x))
    #println("loss = ", loss_val, " ", test)

    println("Loss: ",loss_val, " ", x)
    println(truth_vec[1], " ", pred)
   return loss_val
end

function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp;name=:IF1),
        MTKNeuralToolkit.build_LIF(;name=:IF2),
        MTKNeuralToolkit.build_LIF(;name=:IF3),
        MTKNeuralToolkit.build_LIF(;name=:IF4),
        MTKNeuralToolkit.build_LIF(;name=:IF5)
    ]
    connections = Dict(
    (1, 2) => [(type=:LIF, weight=3.0)],
    (1, 3) => [(type=:LIF, weight=3.0)],
    (1, 4) => [(type=:LIF, weight=3.5)],
    (2, 5) => [(type=:LIF, weight=10.0)],
    (3, 5) => [(type=:LIF, weight=10.0)],
    (4, 5) => [(type=:LIF, weight=10.0)]
)

    ground_sys = build_network(connections, neurons)

    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
end

function get_parameters(prob, system, params)
    param_syms = parameters(prob.f.sys)
    p_array    = Float64[prob.ps[s] for s in param_syms]
    
    params_idx = Int[]
    for p in params
        matches = findall(param_syms) do s
            sym_str = split(string(s), "(")[1]  # strip "(t)"
            contains(sym_str, p)                # contains not endswith
        end
        println("'$p' → matched: ", param_syms[matches])
        append!(params_idx, matches)
    end
    
    # state_idx is separate — define explicitly
    state_idx = [variable_index(prob, system.IF5.IF5.oneport.v)]  # or whichever output neuron
    
    println("Total optimizable params: ", length(params_idx))
    println("Symbols: ", param_syms[params_idx])
    
    return p_array, params_idx, state_idx
end

