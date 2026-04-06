using ChainRulesCore
using SciMLStructures
using SymbolicIndexingInterface

@mtkmodel BasicSoma begin
    @parameters begin
        C, [description = "Capacitance"]
    end
    @variables begin
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        V ~ oneport.v
    end
end

"
Leaky Integrate-And-Fire soma where resetting dynamics are used 
This solves the Mass Matrix problem 
"

function make_spike_callback(prob, neurons_or_idx)
    param_syms   = parameters(prob.f.sys)
    p_tunable, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)

    V_th_pidx    = findfirst(s -> contains(string(s), "V_th"),    param_syms)
    V_reset_pidx = findfirst(s -> contains(string(s), "V_reset"), param_syms)

    # resolve indices vs neuron systems
    v_indices = if eltype(neurons_or_idx) <: Integer
        neurons_or_idx
    else
        state_syms = unknowns(prob.f.sys)
        map(neurons_or_idx) do n
            name = string(nameof(n))
            sym  = state_syms[findfirst(s -> contains(string(s), name * "₊" * name * "₊oneport₊v"), state_syms)]
            variable_index(prob, sym)
        end
    end

    spike_times = [Float64[] for _ in v_indices]

    callbacks = map(enumerate(v_indices)) do (i, v_idx)
        ContinuousCallback(
            # read V_th live from integrator so remake'd params are respected
            (u, t, integrator) -> begin
                V_th = SciMLStructures.canonicalize(SciMLStructures.Tunable(), integrator.p)[1][V_th_pidx]
                u[v_idx] - V_th
            end,
            (integrator) -> begin
                p   = SciMLStructures.canonicalize(SciMLStructures.Tunable(), integrator.p)[1]
                V_reset = p[V_reset_pidx]
                integrator.u[v_idx] = V_reset
                push!(spike_times[i], integrator.t)
            end
        )
    end

    return CallbackSet(callbacks...), spike_times
end


@mtkmodel LIFSoma begin
    @parameters begin
        C, [description = "Capacitance"]
        R
        V_reset = -70
        V_th = -55
        a = 1.0
    end
    @variables begin
        V(t) = -65, [description = "membrane voltage"]
        Spike_count(t) = 0
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u)/ C
        connect(ground.g, oneport.n)
        V ~ oneport.v
        D(Spike_count) ~ 0
    end
end 

"""
A battery: generates a constant potential difference across its terminals
"""
@mtkmodel fixed_reversal begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
    end
    @equations begin
        v ~ E
    end
end


FixedReversal(;name = :reversal, kwargs...) = fixed_reversal(;name, kwargs...)


 
