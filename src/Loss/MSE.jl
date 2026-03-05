"
This is all test code and is in the works (Ella dissertation)
"

function membrane_mse(system, ref_sol, prob)
    
    p0 = get_weights(system, prob)
    
    for i in 1:4
        grad = ForwardDiff.gradient(p -> loss_mse(p, system, ref_sol, prob), p0)
        p0 = p0 - 0.01*grad
        print(get_weights(system, prob))
    end 

    @show p0
end

function loss_mse(p, system, ref_sol, prob)
    p_float = ForwardDiff.value.(p)
    pmap = Dict(zip(parameters(system), p_float))
   
    ground_sol = generate_groundtruth_system(ref_sol, pmap)
    
    prob = remake(prob; p = pmap)

    sol = solve(prob, Tsit5(); saveat = ref_sol.t)
    
    loss = mean((sol[system.IF2.IF2.oneport.v] .- ground_sol[system.IF2.IF2.oneport.v]).^2)
    
    println(loss)
    return loss
end


function loss_mse(p, grad)
    #grad[:] = ...
    return loss 
end

function generate_groundtruth_system(ref_sol, pmap)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp;name=:IF1),
        MTKNeuralToolkit.build_LIF(name=:IF2) 
    ]
    connections = Dict(
        (1, 2) => [(type=:LIF, weight=1.0)]
    )

    ground_sys = build_network(connections, neurons)

    ground_prob = ODEProblem(ground_sys, pmap, (0.0, 200.0))

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
end

function get_last_neurons_arrays(system, solver)
    neuron_array = solver[system.IF2.IF2.oneport.v]
    return neuron_array
end 

function get_weights(system, prob)
    weight = prob.p[ModelingToolkit.parameter_index(prob, system.s12_LIF1.g_max)]
    return [weight]
end 



