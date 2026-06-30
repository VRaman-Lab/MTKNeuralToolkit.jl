This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
src/
  BasicComponents.jl
  connections.jl
  MTKNeuralToolkit.jl
  tempgates.jl
```

# Files

## File: src/tempgates.jl
```julia
struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    # A function taking voltage `v` and returning a tuple: (alpha_expr, beta_expr)
    dynamics::F 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    if isempty(gates)
        # Pure leak channel (avoids broadcasting edge cases with empty gates)
        push!(eqs, i ~ g .* (v .- E_rev))
    else
        conductance_factor = true
        
        for gate in gates
            if isnothing(N)
                gate_var = only(@variables $(gate.name)(t))
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
                init_conds[gate_var] = gate.ic
            else
                gate_var = only(@variables $(gate.name)(t)[1:N])
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:N])
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:N])
                init_conds[gate_var] = fill(gate.ic, N)
            end
            
            push!(vars, gate_var, alpha_var, beta_var)
            
            alpha_expr, beta_expr = gate.dynamics(v)
            
            push!(eqs, alpha_var ~ alpha_expr)
            push!(eqs, beta_var ~ beta_expr)
            push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
            
            conductance_factor = conductance_factor .* (gate_var .^ gate.power)
        end
        
        push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    end
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end
```

## File: src/BasicComponents.jl
```julia
@component function Ground(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named g = Pin()
        eqs = [g.v ~ 0]
    else
        @named g = VectorizedPin(N=N)
        eqs = [g.v ~ zeros(Float64, N)]
    end
    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=[g], name=name)
end

@component function Capacitor(; name, C = 1.0, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=N)
    end
    @unpack v, i = oneport
    @parameters C=C
    # ./ works on both scalars and arrays natively in Symbolics
    eqs = Equation[D(v) ~ i ./ C]
    return extend(System(eqs, t, SymbolicT[], [C]; systems=System[], name=name), oneport)
end

@component function CurrentSource(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
        @named I = RealInput()
    else
        @named oneport = VectorizedOnePort(N=N)
        @named I = RealInputArray(nin=N)
    end
    @unpack i = oneport
    
    eqs = Equation[i ~ -I.u]
    return extend(System(eqs, t, SymbolicT[], SymbolicT[]; systems=[I], name=name), oneport)
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
    
    reversal_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(reversal_sys, oneport)
end

"""
SpikingCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function SpikingCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = SymbolicT[]
    push!(params, C, V_th, V_reset)
    
    @variables begin
        # Bind the incoming V_init default directly to the true differential state
        v(t) = V_init
        V(t)
    end
    # Include both v and V in the structural variables array
    vars = SymbolicT[]
    push!(vars, v, V)

    eqs = Equation[
        D(v) ~ i / C,
        V ~ v
    ]
    
    root_eqs = Equation[v ~ V_th]
    affect = Equation[v ~ V_reset]
    events = [root_eqs => affect]
    
    lif_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        continuous_events = events,
        name
    )
    
    return extend(lif_sys, oneport)
end

@component function GapJunction(; name, R = 1.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters R = R
    params = SymbolicT[]
    push!(params, R)

    vars = SymbolicT[]

    eqs = Equation[]
    push!(eqs, i1 ~ (v1 - v2) / R)
    push!(eqs, i2 ~ -i1)

    return extend(System(eqs, t, vars, params; systems=System[], name=name), twoport)
end

@component function ChemicalSynapse(; name, g_max=2.0, τ=5.0, v_th=-20.0, w=0.5, E_rev=0.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters E_rev=E_rev g_max=g_max τ=τ v_th=v_th w=w
    params = SymbolicT[]
    push!(params, E_rev, g_max, τ, v_th, w)

    @variables s(t) = 0.0
    vars = SymbolicT[]
    push!(vars, s)

    eqs = Equation[]
    push!(eqs, i1 ~ 0.0)
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, i2 ~ (v2 - E_rev) * s * g_max)

    # Events should also be built cleanly
    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    events = [root_eqs => affect]

    return extend(System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name), twoport)
end

function spike_affect!(mod, obs, ctx, integ)
    j = ctx.j
    W = ctx.W
    N = ctx.N

    S_new = copy(mod.S)
    for i in 1:N
        S_new[j, i] += W[j, i]
    end
    return (; S = S_new)
end

# ==========================================
# VECTORIZED ELECTRICAL COMPONENTS
# ==========================================

@connector function VectorizedPin(; name, N::Int, v = nothing, i = nothing)
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i, [connect = Flow]
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function VectorizedOnePort(; name, N::Int, v = nothing, i = nothing)
    pars = @parameters begin
    end
    systems = @named begin
        p = VectorizedPin(N=N)
        n = VectorizedPin(N=N)
    end
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i
    end
    equations = Equation[
        v ~ p.v - n.v,
        collect(p.i .+ n.i .~ 0.0)...,  # splat the collected equations
        i ~ p.i,
    ]

    return System(equations, t, vars, pars; name, systems)
end

@component function SynapsePort(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named p = Pin()
        @variables I_syn(t)
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    else
        @named p = VectorizedPin(N=N)
        @variables I_syn(t)[1:N]
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    end
    return System(eqs, t, vars, SymbolicT[]; systems=[p], name=name)
end

@component function ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    # Sigmoidal activation — smooth, no events needed
    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope]; 
                  systems=System[], name=name)
end

@component function AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s1(t)=0.0 s2(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s1) ~ -s1 / τ + σ(V_pre - V_th),
        D(s2) ~ -s2 / τ + s1,           # cascaded low-pass → alpha shape
        I_syn ~ g_max * s2 * (V_post - E_rev)
    ]
    return System(eqs, t, [s1, s2, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, slope]; systems=System[], name=name)
end

@component function NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, 
                                  Mg_conc=1.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th Mg_conc=Mg_conc slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    # Mg block is a function of V_post — still fully causal
    mg_block(V) = 1.0 / (1.0 + Mg_conc * exp(-0.062 * V))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * mg_block(V_post) * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, Mg_conc, slope]; systems=System[], name=name)
end

@component function VectorizedExpSynapse(; name, N_pre, N_post, W,
                                            g_max=1.0, τ=5.0, E_rev=0.0,
                                            V_th=-20.0, slope=2.0)
    @variables s(t)[1:N_pre] I_syn(t)[1:N_post] V_pre(t)[1:N_pre] V_post(t)[1:N_post]
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    # Native vectorized dynamics
    σ(V) = 1.0 ./ (1.0 .+ exp.(-(V .- V_th) ./ slope))
    synaptic_drive = W * s
    
    eqs = [
        D(s) ~ -s ./ τ .+ σ(V_pre),
        I_syn ~ g_max .* (V_post .- E_rev) .* synaptic_drive
    ]
    
    # Only provide initial conditions for the differential state variable
    init_conds = Dict(s => zeros(N_pre))
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope];
                  systems=System[], 
                  initial_conditions=init_conds, 
                  name=name)
end
```

## File: src/connections.jl
```julia
# connections.jl

using Symbolics: SymbolicT
using ModelingToolkit: t_nounits as t, connect, Equation, System, @named, getproperty, nameof

# =========================================================
# 1. STRUCT DEFINITIONS
# =========================================================

struct Compartment
    sys::System
    interfaces::NamedTuple
    V_init::Float64
    N::Union{Int, Nothing}  
end

struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

struct SynapseSpec
    pre_V
    post_V
    post_I_syn
    synapse
    post_comp::Union{Compartment, Nothing} 
end

# Backward-compatible constructor
SynapseSpec(pre_V, post_V, post_I_syn, synapse) = SynapseSpec(pre_V, post_V, post_I_syn, synapse, nothing)




struct CouplingSpec
    comp_i::Compartment
    comp_j::Compartment
    coupling::System
end


# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named injector  = CurrentSource()
        @named syn_injector = CurrentSource()
        @named p = Pin()
        @named n = Pin()
        init_v = V_init
    else
        @named injector  = CurrentSource(N=N)
        @named syn_injector = CurrentSource(N=N)
        @named p = VectorizedPin(N=N)
        @named n = VectorizedPin(N=N)
        init_v = fill(V_init, N)
    end

    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, syn_injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together
    p_connections = System[capacitor, injector, syn_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_injector, p, n]
    append!(all_systems, channels)

    sys = System(eqs, t, vars, SymbolicT[];
                 systems = all_systems,
                 initial_conditions = Dict(capacitor.v => init_v),
                 name)

    cap_name = nameof(capacitor)
    V_state  = getproperty(sys, cap_name).v

    interfaces = (
        V       = V_state,
        p_pin   = getproperty(sys, nameof(p)),
        n_pin   = getproperty(sys, nameof(n)),
        I_ext   = getproperty(sys, nameof(injector)).I.u,
        I_syn   = getproperty(sys, nameof(syn_injector)).I.u,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init, N)
end






# =========================================================
# 3. SYNAPSE WIRING
# =========================================================

"""
    wire_synapses!(eqs, systems, specs)

Wires a collection of SynapseSpecs into the network equations.
Pre-collects convergent synapses by target and writes one sum equation per target.
Returns the set of driven I_syn targets (for grounding the rest).
"""
function wire_synapses!(eqs::Vector{Equation}, systems::Vector{System},
                        specs::Vector{SynapseSpec})
    syn_by_target = Dict{SymbolicT, Vector{SymbolicT}}()
    driven_syn_targets = Set{SymbolicT}()
    block_driven_targets = Set{SymbolicT}()

    for spec in specs
        push!(systems, spec.synapse)
        
        if hasproperty(spec.synapse, :V_pre)
            push!(eqs, spec.synapse.V_pre ~ spec.pre_V)
        end
        if hasproperty(spec.synapse, :V_post)
            push!(eqs, spec.synapse.V_post ~ spec.post_V)
        end

        if spec.post_I_syn isa AbstractArray
            # Block synapse: equate arrays directly
            push!(eqs, spec.post_I_syn ~ spec.synapse.I_syn)
            push!(block_driven_targets, spec.post_I_syn) # Track the whole array!
            
            # Still add elements to driven_syn_targets for safety
            for i in 1:length(spec.post_I_syn)
                push!(driven_syn_targets, spec.post_I_syn[i])
            end
        else
            # Scalar synapse
            key = spec.post_I_syn
            haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
            push!(syn_by_target[key], spec.synapse.I_syn)
            push!(driven_syn_targets, key)
        end
    end

    for (target, currents) in syn_by_target
        push!(eqs, target ~ sum(currents))
    end

    return driven_syn_targets, block_driven_targets
end







# =========================================================
# 4. NETWORK BUILDER
# =========================================================

function build_acausal_network(compartments::Vector{Compartment};
                                coupling_specs=CouplingSpec[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network)

    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 1. Ground each compartment individually
    for (i, comp) in enumerate(compartments)
        if haskey(comp.interfaces, :n_pin)
            gnd_name = Symbol(:gnd_, i)
            if isnothing(comp.N)
                gnd = Ground(name=gnd_name)
            else
                gnd = Ground(N=comp.N, name=gnd_name)
            end
            push!(all_systems, gnd)
            push!(eqs, connect(gnd.g, comp.interfaces.n_pin))
        end
    end

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)
        comp = compartments[idx]

        if haskey(comp.interfaces, :I_ext)
            if stim isa System
                push!(all_systems, stim)
                push!(eqs, comp.interfaces.I_ext ~ stim.output.u)
            elseif stim isa AbstractVector
                push!(eqs, comp.interfaces.I_ext ~ stim)
            elseif stim isa Number
                if isnothing(comp.N)
                    push!(eqs, comp.interfaces.I_ext ~ stim)
                else
                    push!(eqs, comp.interfaces.I_ext ~ fill(stim, comp.N))
                end
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_ext) && !(i in driven_compartments)
            if isnothing(comp.N)
                push!(eqs, comp.interfaces.I_ext ~ 0.0)
            else
                push!(eqs, comp.interfaces.I_ext ~ zeros(Float64, comp.N))
            end
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, spec) in enumerate(coupling_specs)
        push!(all_systems, spec.coupling)
        
        # Defensive check for pins
        if haskey(spec.comp_i.interfaces, :p_pin) && hasproperty(spec.coupling, :p1)
            push!(eqs, connect(spec.comp_i.interfaces.p_pin, spec.coupling.p1))
            push!(eqs, connect(spec.coupling.n1, spec.comp_i.interfaces.n_pin))
        end
        
        if haskey(spec.comp_j.interfaces, :p_pin) && hasproperty(spec.coupling, :p2)
            push!(eqs, connect(spec.comp_j.interfaces.p_pin, spec.coupling.p2))
            push!(eqs, connect(spec.coupling.n2, spec.comp_j.interfaces.n_pin))
        end
        
        # Mark as gap junctioned so p_pin.i isn't grounded
        push!(gap_junctioned, findfirst(==(spec.comp_i), compartments))
        push!(gap_junctioned, findfirst(==(spec.comp_j), compartments))
    end

        # 5. Identify block-synapsed compartments by index using the Compartment object
    block_synapsed_compartments = Set{Int}()
    for spec in synapse_specs
        if spec.post_I_syn isa AbstractArray && spec.post_comp !== nothing
            idx = findfirst(==(spec.post_comp), compartments)
            if idx !== nothing
                push!(block_synapsed_compartments, idx)
            end
        end
    end

    # 6. Wire synapses
    driven_syn_targets, _ = wire_synapses!(eqs, all_systems, synapse_specs)

    # 7. Ground non-synapsed I_syn
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_syn)
            # Skip entirely if driven by a block synapse (reliable integer check)
            if i in block_synapsed_compartments
                continue
            end
            
            if isnothing(comp.N)
                if !(comp.interfaces.I_syn in driven_syn_targets)
                    push!(eqs, comp.interfaces.I_syn ~ 0.0)
                end
            else
                for j in 1:comp.N
                    i_syn_j = comp.interfaces.I_syn[j]
                    if !(i_syn_j in driven_syn_targets)
                        push!(eqs, i_syn_j ~ 0.0)
                    end
                end
            end
        end
    end


    # 8. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :p_pin) && !(i in gap_junctioned)
            if isnothing(comp.N)
                push!(eqs, comp.interfaces.p_pin.i ~ 0.0)
            else
                push!(eqs, comp.interfaces.p_pin.i ~ zeros(Float64, comp.N))
            end
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
    return Network(net_sys, DataFrame(), DataFrame(), SymbolicT[])
end








function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn, post_comp) # Pass post_comp here
end
```

## File: src/MTKNeuralToolkit.jl
```julia
module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT, ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num

include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, SpikingCapacitor, CurrentSource, FixedReversal 
export ChemicalSynapse, GapJunction, AlphaSynapse, SynapseSpec

export VectorizedPin, VectorizedOnePort

include("connections.jl")
export build_compartment, Compartment
export build_synapse
export build_acausal_network, build_synapse_block, CouplingSpec

include("tempgates.jl")
export GateSpec, GenericChannel

export ExpSynapse, VectorizedExpSynapse

end
```
