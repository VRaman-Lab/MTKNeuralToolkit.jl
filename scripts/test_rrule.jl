using ModelingToolkit, OrdinaryDiffEq, Zygote, ChainRulesCore
using ModelingToolkit: t_nounits as t, D_nounits as D
using SciMLSensitivity
using Plots
using SymbolicIndexingInterface: variable_index

sts = @variables x(t), v(t)
par = @parameters g = 9.8

bb_eqs = [D(x) ~ v
          D(v) ~ -g]

# ── surrogate gradient via rrule ─────────────────────────────────────────────

function bounce_affect(v::T) where {T}
    return -v
end

function ChainRulesCore.rrule(::typeof(bounce_affect), v)
    v_out = bounce_affect(v)
    function bounce_affect_pullback(ȳ)
        @info "here"
        return NoTangent(), -one(v) * ȳ
    end
    return v_out, bounce_affect_pullback
end

function bb_affect_ad!(mod, obs, integ, ctx)
    return (; v = bounce_affect(mod.v))
end

reflect_ad = [x ~ 0] => (bb_affect_ad!, (; v))

@mtkcompile bb_sys_ad = System(bb_eqs, t, sts, par,
    continuous_events = reflect_ad)

# ── build a NUMERIC u0 map once, outside the loss ────────────────────────────
# ODEProblem with symbolic u0 just to get the index ordering right
_prob0 = ODEProblem(bb_sys_ad, [v => 0.0, x => 1.0], (0.0, 5.0))

# find which index corresponds to v in the plain state vector
v_idx = variable_index(bb_sys_ad, v)   # e.g. 1 or 2
x_idx = variable_index(bb_sys_ad, x)

# ── loss: build problem from a plain Float64 vector, no remake/symbolic path ──
function loss(v0::T) where {T}
    # build u0 immutably using vcat of the two known slots
    # works for 2-state system [v_idx, x_idx]
    u0_num = if v_idx < x_idx
        T[v0, one(T)]
    else
        T[one(T), v0]
    end

    prob = ODEProblem(
        _prob0.f,
        u0_num,
        (zero(T), T(5.0)),
        _prob0.p,
    )
    sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8,
                sensealg=ForwardDiffSensitivity())
    return sum(abs2, sol.u[end])
end

# ── sanity checks ─────────────────────────────────────────────────────────────
val, pb = ChainRulesCore.rrule(bounce_affect, 3.0)
_, ∂v   = pb(1.0)
@assert ∂v ≈ -1.0

grad = Zygote.gradient(loss, 0.0)[1]
println("∂loss/∂v₀ = ", grad)