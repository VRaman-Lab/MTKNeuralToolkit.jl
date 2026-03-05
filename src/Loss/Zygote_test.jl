
"
This is all test code and is in the works (Ella dissertation)
"

function Zygote_test(system, ref_sol, prob)

    ground_sol = generate_groundtruth_system(ref_sol)
    tsteps     = ground_sol.t

    param_syms = parameters(prob.f.sys)
    p_array    = Float64[prob.ps[s] for s in param_syms]

    g_max_sym  = system.s12_LIF1.g_max
    g_max_idx  = findfirst(s -> isequal(s, g_max_sym), param_syms)
    state_idx  = variable_index(prob, system.IF2.IF2.oneport.v)

    f_plain    = ODEFunction(prob.f.f)
    prob_plain = ODEProblem(f_plain, copy(prob.u0), prob.tspan, p_array)
    truth_vec  = ground_sol[state_idx, :]

    p0 = [p_array[g_max_idx]]

    optfn   = OptimizationFunction(loss_with_logging, Optimization.AutoZygote())
    optprob = OptimizationProblem(
        optfn, p0,
        (prob_plain, tsteps, truth_vec, g_max_idx, state_idx),
    )
    sol = solve(optprob, ADAM(0.1); epochs = 500)

    sol.u, sol
end


function loss_with_logging(x, p)
    prob_plain, tsteps, truth_vec, g_max_idx, state_idx = p

    p_new   = [i == g_max_idx ? x[1] : prob_plain.p[i] for i in eachindex(prob_plain.p)]
    newprob = remake(prob_plain; p = p_new)

    sol = solve(
        newprob, Tsit5();
        saveat   = tsteps,
        dense    = false,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())
    )

    pred     = sol[state_idx, :]
    loss_val = mean((truth_vec .- pred).^2)

    println("loss = ", loss_val, "  g_max = ", x[1])

    return loss_val
end

function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp;name=:IF1),
        MTKNeuralToolkit.build_LIF(name=:IF2) 
    ]
    connections = Dict(
        (1, 2) => [(type=:LIF, weight=1.0)]
    )

    ground_sys = build_network(connections, neurons)

    p0 = [ground_sys.s12_LIF1.g_max => 2.0]

    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0), p0)

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
end

function get_last_neurons_arrays(system, solver)
    neuron_array = solver[system.IF2.IF2.oneport.v]
    return neuron_array
end 

function get_weights(system, prob)
    weight = prob.ps[system.s12_LIF1.g_max]
    return [weight]
end 