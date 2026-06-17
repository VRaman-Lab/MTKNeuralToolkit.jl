
using ModelingToolkit: renamespace




"""
build_neuron: Builder function that automatically compiles a parallel connection matrix
across a Soma and a hardcoded internal CurrentSource injector.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource()

    @named p = Pin()
    @named n = Pin()

    @variables begin
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, connect(capacitor.n, ground.g))
    
    # Destructure connection arrays sequentially to avoid splatting types
    p_connections = System[]
    push!(p_connections, capacitor.p)
    for ch in channels
        push!(p_connections, ch.gate.p)
    end
    push!(p_connections, injector.p)
    push!(eqs, connect(p_connections...))

    n_connections = System[]
    push!(n_connections, capacitor.n)
    for ch in channels
        push!(n_connections, ch.batt.n)
    end
    push!(n_connections, injector.n)
    push!(eqs, connect(n_connections...))
    
    push!(eqs, V ~ p.v)
    
    # Assemble systems cleanly
    all_systems = System[]
    push!(all_systems, p)
    push!(all_systems, n)
    push!(all_systems, capacitor)
    push!(all_systems, ground)
    push!(all_systems, injector)
    append!(all_systems, channels)
    
    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    else
        push!(eqs, injector.I.u ~ 0.0)
    end

    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = all_systems, 
        initial_conditions, 
        guesses, 
        name
    )
end



"""
build_channel: Factory function that wires a gating mechanism in series 
with an ionic reversal potential battery.
"""
function build_channel(gate, battery; name)
    eqs = Equation[]
    push!(eqs, connect(gate.n, battery.p))
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    subsystems = System[]
    push!(subsystems, gate)
    push!(subsystems, battery)
    
    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = subsystems, 
        initial_conditions, 
        guesses, 
        name
    )
end


function EventSynapseGate(; name, g_max = 0.5, τ = 5.0, v_th = -20.0, w = 0.1)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport
    
    @parameters begin
        g_max = g_max
        τ = τ
        v_th = v_th
        w = w
    end
    params = SymbolicT[]
    push!(params, g_max)
    push!(params, τ)
    push!(params, v_th)
    push!(params, w)
    
    @variables begin
        s(t)
    end
    vars = SymbolicT[]
    push!(vars, s)
    
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    initial_conditions[s] = 0.0
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    push!(eqs, i1 ~ 0.0)
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, i2 ~ v2 * s * g_max)
    
    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    
    events = root_eqs => affect
    
    syn_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        initial_conditions, 
        guesses, 
        continuous_events = events,
        name
    )
    return extend(syn_sys, twoport)
end



function build_synapse(gate, battery; name)
    @named p1 = Pin()
    @named n1 = Pin()
    @named p2 = Pin()
    @named n2 = Pin()
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    push!(eqs, connect(p1, gate.p1))
    push!(eqs, connect(p2, gate.p2))
    push!(eqs, connect(gate.n2, battery.p))
    
    # Pass the references straight through to the boundary pins
    # instead of sinking them to an internal ground
    push!(eqs, connect(n1, gate.n1))
    push!(eqs, connect(n2, battery.n))
    
    subsystems = System[]
    push!(subsystems, p1)
    push!(subsystems, n1)
    push!(subsystems, p2)
    push!(subsystems, n2)
    push!(subsystems, gate)
    push!(subsystems, battery)
    
    return System(eqs, t, vars, params; systems = subsystems, initial_conditions, guesses, name)
end
 
function neuron_connect(pre_compartment, post_compartment, synapse)
    eqs = Equation[]
    # Connect signal lines
    push!(eqs, connect(pre_compartment.p, synapse.p1))
    push!(eqs, connect(post_compartment.p, synapse.p2))
    
    # Safe reference routing: anchors the synapse ports to the neuron's ground reference
    push!(eqs, connect(pre_compartment.n, synapse.n1))
    push!(eqs, connect(post_compartment.n, synapse.n2))
    return eqs
end

"""
build_network: Automatically maps and connects an arbitrary list of neurons,
synaptic pairs, and external drivers into a unified system.

NEED TO MAKE PRECOMPILATION FRIENDLY
"""
function build_network(neurons, synapses, connections; drivers=[], name=:neural_network)
    eqs = Equation[]
    
    # 1. Initialize concretely-typed subsystems vector to avoid splatting types
    all_systems = System[]
    append!(all_systems, neurons)
    append!(all_systems, synapses)
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    # Automate synaptic wiring from a list of tuples: (pre_idx, post_idx, synapse_obj)
    for (pre_idx, post_idx, syn) in connections
        append!(eqs, neuron_connect(neurons[pre_idx], neurons[post_idx], syn))
    end
    
    # Automate driver loops (e.g., attaching stimulus blocks + current sources)
    for (neuron_idx, stimulus_block, source_block) in drivers
        push!(eqs, connect(stimulus_block.output, source_block.I))
        push!(eqs, connect(source_block.p, neurons[neuron_idx].p))
        push!(eqs, connect(source_block.n, neurons[neuron_idx].n))
        push!(all_systems, stimulus_block)
        push!(all_systems, source_block)
    end
    
    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = all_systems, 
        initial_conditions, 
        guesses, 
        name
    )
end

