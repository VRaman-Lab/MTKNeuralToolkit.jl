"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters begin
        C = C
    end
    params = SymbolicT[]
    push!(params, C)
    
    @variables begin
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    initial_conditions[V] = -65.0
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    push!(eqs, D(v) ~ i / C)
    push!(eqs, V ~ v)
    
    cap_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        initial_conditions, 
        guesses, 
        name
    )
    return extend(cap_sys, oneport)
end

"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    vars = SymbolicT[]
    params = SymbolicT[]
    eqs = Equation[]
    push!(eqs, i ~ -I.u)
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    # We cast 'I' into a Vector{System} instead of leaving it as an untyped literal array
    subsystems = System[]
    push!(subsystems, I)
    
    source_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = subsystems, 
        initial_conditions, 
        guesses, 
        name
    )
    return extend(source_sys, oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    @parameters begin
        E = E
    end
    params = SymbolicT[]
    push!(params, E)
    vars = SymbolicT[]
    eqs = Equation[]
    push!(eqs, v ~ E)
    
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    reversal_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        initial_conditions, 
        guesses, 
        name
    )
    return extend(reversal_sys, oneport)
end

"""
LIFCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function LIFCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = SymbolicT[]
    push!(params, C)
    push!(params, V_th)
    push!(params, V_reset)
    
    @variables begin
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    initial_conditions[V] = -65.0
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    push!(eqs, D(v) ~ i / C)
    push!(eqs, V ~ v)
    
    # continuous_events expects a Vector{Equation} => Vector{Equation} pair (or arrays of pairs)
    # Using push! ensures the condition and affect equations are cleanly typed
    root_eqs = Equation[]
    push!(root_eqs, v ~ V_th)
    
    affect = Equation[]
    push!(affect, v ~ V_reset)
    
    events = root_eqs => affect
    
    lif_sys = System(
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
    
    return extend(lif_sys, oneport)
end
