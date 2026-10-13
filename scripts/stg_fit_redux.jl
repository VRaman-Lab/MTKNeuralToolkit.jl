using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron: PrinzGeometry, na_gates, cas_gates, cat_gates, ka_gates, kca_gates, kdr_gates, h_gates
using ModelingToolkit: mtkcompile, @named, unknowns
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK: TRBDF2
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize
import SciMLStructures
using SymbolicIndexingInterface: parameter_values, setp, getu, variable_index
using PreallocationTools
using DataInterpolations
using SciMLBase
using PlotlyJS
using ForwardDiff
using LinearAlgebra: norm
using SciMLBase: terminate!, ReturnCode

println("Threads.nthreads() = ", Threads.nthreads(), "   MODE = ", MODE)

const ABORT = Threads.Atomic{Bool}(false)

@inline function diverged(u, bound)
    @inbounds for ui in u
        v = ForwardDiff.value(ui)
        (isfinite(v) && abs(v) < bound) || return true
    end
    return false
end

function flag_failures(flag::Threads.Atomic{Bool})
    function output_func(sol, i)
        SciMLBase.successful_retcode(sol.retcode) || (flag[] = true)
        return sol, false          # keep the full solution; no rerun
    end
    return output_func
end

function abort_callback(flag::Threads.Atomic{Bool}; max_steps=20_000, u_bound=1e3)
    condition(u, t, integrator) =
        flag[] || integrator.iter > max_steps || diverged(u, u_bound)
    function affect!(integrator)
        flag[] = true
        terminate!(integrator, ReturnCode.Unstable)
    end
    init!(c, u, t, integrator) = flag[] && terminate!(integrator, ReturnCode.Unstable)
    return DiscreteCallback(condition, affect!; initialize=init!, save_positions=(false, false))
end

const abort_cb = abort_callback(ABORT)

const LATEST_LOSS_COMPONENTS = Ref((obs=0.0, mse=0.0, cont=0.0, total=0.0, tot_raw=0.0, x=Float64[]))
const CB_PREV_U  = Ref{Union{Nothing,Vector{Float64}}}(nothing)
const EVAL_COUNT = Ref(0)

# ============================================================
# True STG network: geometry and channel constructors, AB/PY/LP
# compartments, synapses, and assembly
# ============================================================
geom = PrinzGeometry(area=0.0628, C_m=10.0)
const G_SCALE = geom.C_m / geom.area
tauCa = 200.0
Ca_inf = 0.05
nernst_factor = 500.0 * 8.6174e-5 * 283.15
prinz_ion_config = CalciumTracker(decay=ca -> (Ca_inf .- ca) ./ tauCa, Ca_init=Ca_inf)

NaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=50.0, gates=na_gates, geometry=geom)
CaSCh(g; name)  = CaVChannel(name=name, g=g, gates=cas_gates, Ca_out=3000.0, nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
CaTCh(g; name)  = CaVChannel(name=name, g=g, gates=cat_gates, Ca_out=3000.0, nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
HCh(g; name)    = GenericChannel(name=name, g=g, E_rev=-20.0, gates=h_gates, geometry=geom)
KaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=-80.0, gates=ka_gates, geometry=geom)
KCaCh(g; name)  = KCaChannel(name=name, g=g, E_rev=-80.0, gates=kca_gates, geometry=geom)
KdrCh(g; name)  = GenericChannel(name=name, g=g, E_rev=-80.0, gates=kdr_gates, geometry=geom)
LeakCh(g; name) = GenericChannel(name=name, g=g, E_rev=-50.0, gates=GateSpec[], geometry=geom)

function build_stg_true(V_inits=(-60.0, -55.0, -65.0); name=:stg_true)
    function build_AB()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(6.0);  @named cat = CaTCh(2.5)
        @named h    = HCh(0.01);   @named ka   = KaCh(50.0);  @named kca = KCaCh(5.0)
        @named kdr  = KdrCh(100.0)
        return build_compartment(cap, [na, cas, cat, h, ka, kca, kdr]; name=:AB, V_init=V_inits[1], ion_config=prinz_ion_config)
    end
    function build_PY()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(2.0);  @named cat = CaTCh(2.4)
        @named h    = HCh(0.05);   @named ka   = KaCh(50.0);  @named kdr = KdrCh(125.0)
        @named leak = LeakCh(0.01)
        return build_compartment(cap, [na, cas, cat, h, ka, kdr, leak]; name=:PY, V_init=V_inits[2], ion_config=prinz_ion_config)
    end
    function build_LP()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(4.0)
        @named h    = HCh(0.05);   @named ka   = KaCh(20.0);  @named kdr = KdrCh(25.0)
        @named leak = LeakCh(0.03)
        return build_compartment(cap, [na, cas, h, ka, kdr, leak]; name=:LP, V_init=V_inits[3], ion_config=prinz_ion_config)
    end

    AB = build_AB(); PY = build_PY(); LP = build_LP()
    neurons = [AB, PY, LP]

    @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

    synapse_specs = [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut),
    ]
    return build_acausal_network(neurons; synapse_specs=synapse_specs, name=name)
end

# ============================================================
# Ground-truth simulation and synthetic data
# ============================================================

simulation_length = 2000.0

println("Generating STG training data and interpolating...")
timesteps = 0.0:0.1:simulation_length
const NSAMP = length(timesteps)          # 20001

true_net = build_stg_true()
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, simulation_length), jac=true, sparse=true)
true_sol = solve(true_prob, TRBDF2(); saveat=timesteps)

V_data_AB = true_sol[true_sys.AB.cap.v]
V_data_LP = true_sol[true_sys.LP.cap.v]
V_data_PY = true_sol[true_sys.PY.cap.v]
v_data_true = [V_data_AB, V_data_PY, V_data_LP]          # order: AB, PY, LP
Ca_data_AB = true_sol[true_sys.AB.AB_ca_pool.Ca]
Ca_data_LP = true_sol[true_sys.LP.LP_ca_pool.Ca]
Ca_data_PY = true_sol[true_sys.PY.PY_ca_pool.Ca]

itp_AB   = LinearInterpolation(V_data_AB, timesteps)
itp_LP   = LinearInterpolation(V_data_LP, timesteps)
itp_PY   = LinearInterpolation(V_data_PY, timesteps)
itp_ABCa = LinearInterpolation(Ca_data_AB, timesteps)
itp_LPCa = LinearInterpolation(Ca_data_LP, timesteps)
itp_PYCa = LinearInterpolation(Ca_data_PY, timesteps)

true_idx = Dict(string(s) => k for (k, s) in enumerate(unknowns(true_sys)))
println("Done!")

# ============================================================
# Conductance bookkeeping
# ============================================================

true_g = Dict(
    :AB => Dict(:na=>100.0, :cas=>6.0, :cat=>2.5, :h=>0.01, :ka=>50.0, :kca=>5.0, :kdr=>100.0),
    :LP => Dict(:na=>100.0, :cas=>4.0, :h=>0.05, :ka=>20.0, :kdr=>25.0, :leak=>0.03),
    :PY => Dict(:na=>100.0, :cas=>2.0, :cat=>2.4, :h=>0.05, :ka=>50.0, :kdr=>125.0, :leak=>0.01),
)

# Starting guess (warm start for stage 1)
guess_g = Dict(
    :AB => Dict(:cas=>5.0, :cat=>8.0, :h=>0.8, :ka=>80.0, :kca=>8.0, :kdr=>80.0, :na=>80.0),
    :LP => Dict(:cas=>8.5913355150382917, :h=>1.066036858692129e-5, :ka=>8.9697593929373958, :kdr=>8.262058690827629, :leak=>0.80617713581606, :na=>80.1319251516899396),
    :PY => Dict(:cas=>8.0, :cat=>2.0352089502849813, :h=>1.2910544018343762e-5, :ka=>3.176764971395339, :kdr=>80.1992262537045413, :leak=>0.8953930254798708, :na=>80.13260191867413),
)

const NEURONS = (:AB, :PY, :LP)
const TUNABLE_SETS = Dict(
    :AB => [:na, :cas, :cat, :h, :ka, :kca, :kdr],
    :PY => [:na, :cas, :cat, :h, :ka, :kdr, :leak],
    :LP => [:na, :cas, :h, :ka, :kdr, :leak],
)

const TUNABLE_PAIRS = [(n, ch) for n in NEURONS for ch in TUNABLE_SETS[n]]
n_params = length(TUNABLE_PAIRS)

theta_from_g(g) = [log(g[n][ch]) for (n, ch) in TUNABLE_PAIRS]

function theta_to_g(θ, g_ref=guess_g)
    g = Dict(n => Dict{Symbol,Float64}(ch => Float64(v) for (ch, v) in d) for (n, d) in g_ref)
    for (k, (n, ch)) in enumerate(TUNABLE_PAIRS)
        g[n][ch] = exp(θ[k])
    end
    return g
end

function g_to_string(g)
    lines = ["guess_g = Dict("]
    for n in sort!(collect(keys(g)))
        body = join(("$(repr(ch))=>$(g[n][ch])" for ch in sort!(collect(keys(g[n])))), ", ")
        push!(lines, "    $(repr(n)) => Dict($body),")
    end
    push!(lines, ")")
    return join(lines, "\n")
end

function resolve_g(g_warm, neuron::Symbol)
    base = copy(true_g[neuron])
    for ch in TUNABLE_SETS[neuron]
        base[ch] = g_warm[neuron][ch]
    end
    return base
end

# ============================================================
# Fit-neuron models (PEM observation channel optional)
# ============================================================

function build_AB(g; includePem=false, itps=nothing, K=1.0)
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(g[:na]);  @named cas = CaSCh(g[:cas]); @named cat = CaTCh(g[:cat])
    @named h    = HCh(g[:h]);    @named ka  = KaCh(g[:ka]);   @named kca = KCaCh(g[:kca])
    @named kdr  = KdrCh(g[:kdr])
    comps = [na, cas, cat, h, ka, kca, kdr]
    if includePem
        @named pem = PEMObservationChannel(itps=itps, K_init=K, topology=Scalar())
        push!(comps, pem)
    end
    return build_compartment(cap, comps; name=:AB, V_init=-60.0, ion_config=prinz_ion_config)
end

function build_PY(g; includePem=false, itps=nothing, K=1.0)
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(g[:na]);  @named cas = CaSCh(g[:cas]); @named cat = CaTCh(g[:cat])
    @named h    = HCh(g[:h]);    @named ka  = KaCh(g[:ka]);   @named kdr = KdrCh(g[:kdr])
    @named leak = LeakCh(g[:leak])
    comps = [na, cas, cat, h, ka, kdr, leak]
    if includePem
        @named pem = PEMObservationChannel(itps=itps, K_init=K, topology=Scalar())
        push!(comps, pem)
    end
    return build_compartment(cap, comps; name=:PY, V_init=-55.0, ion_config=prinz_ion_config)
end

function build_LP(g; includePem=false, itps=nothing, K=1.0)
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(g[:na]);  @named cas = CaSCh(g[:cas])
    @named h    = HCh(g[:h]);    @named ka  = KaCh(g[:ka]);   @named kdr = KdrCh(g[:kdr])
    @named leak = LeakCh(g[:leak])
    comps = [na, cas, h, ka, kdr, leak]
    if includePem
        @named pem = PEMObservationChannel(itps=itps, K_init=K, topology=Scalar())
        push!(comps, pem)
    end
    return build_compartment(cap, comps; name=:LP, V_init=-65.0, ion_config=prinz_ion_config)
end

function build_stg_fit(k, g_warm; includePem::Bool, name=:stg_fit)
    AB = build_AB(resolve_g(g_warm, :AB); includePem=includePem, itps=[itp_AB], K=k)
    PY = build_PY(resolve_g(g_warm, :PY); includePem=includePem, itps=[itp_PY], K=k)
    LP = build_LP(resolve_g(g_warm, :LP); includePem=includePem, itps=[itp_LP], K=k)
    neurons = [AB, PY, LP]

    @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

    synapse_specs = [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut),
    ]
    return build_acausal_network(neurons; synapse_specs=synapse_specs, name=name)
end

# ============================================================
# Stage schedule (continuation)
#   n_shoots : number of shooting segments (multiples of 16, divisors of 20000)
#   k        : PEM gain (only used when w_obs > 0)
#   w_obs    : weight of observer-current loss  (0 => term not computed, PEM not built)
#   w_mse    : weight of voltage MSE vs data    (0 => not computed)
#   w_cont   : weight of boundary continuity    (0 => not computed)
#   y        : stage ends when total loss <= initial / y   (Inf => never)
# ============================================================

mkstage(; n_shoots, k, w_obs, w_mse, w_cont, y=4.0, max_iters=100, alternate::Union{Bool,NamedTuple}=false) =
    (n_shoots=n_shoots, k=Float64(k), w_obs=Float64(w_obs), w_mse=Float64(w_mse),
     w_cont=Float64(w_cont), y=Float64(y), max_iters=max_iters, alternate=alternate)

const CA_ALT = (groups   = (((:AB,:cas),(:AB,:cat),(:PY,:cas),(:PY,:cat),(:LP,:cas)),
                            ((:AB,:kca),)),
                patterns = ((1,0), (0,1)),
                divider  = 10)

const STAGES = [
    mkstage(n_shoots=16,   k=0.25,  w_obs=1, w_mse=0, w_cont=0, alternate=CA_ALT),
    mkstage(n_shoots=80,   k=0.15,  w_obs=1, w_mse=0, w_cont=0, alternate=CA_ALT),
    mkstage(n_shoots=160,  k=0.1, w_obs=1, w_mse=0, w_cont=0, alternate=CA_ALT),
    mkstage(n_shoots=400,  k=0.05, w_obs=1, w_mse=0, w_cont=0, alternate=CA_ALT),
    mkstage(n_shoots=800,  k=0.025,  w_obs=0.5, w_mse=0.1, w_cont=0.05, alternate=CA_ALT),
    mkstage(n_shoots=2000, k=0.0,  w_obs=0, w_mse=1, w_cont=0.5),
    mkstage(n_shoots=800,  k=0.0,  w_obs=0, w_mse=1, w_cont=0.5),
    mkstage(n_shoots=400,  k=0.0,  w_obs=0, w_mse=1, w_cont=0.25),
    mkstage(n_shoots=160,  k=0.0,  w_obs=0, w_mse=1, w_cont=0),
    mkstage(n_shoots=80,   k=0.0,  w_obs=0, w_mse=1, w_cont=0)
]

function split_shoots(N, n_shoots; overlap=1)
    (N - 1) % n_shoots == 0 || error("n_shoots=$n_shoots does not divide $(N-1)")
    stride = (N - 1) ÷ n_shoots
    group_size = stride + overlap
    @assert overlap * 4 <= group_size
    shoots = [s:min(s + group_size - 1, N) for s in 1:stride:(N - 1)]
    return shoots, length(shoots), stride, group_size
end

function build_stage(stage, g_warm; overlap=1)
    includePem = stage.w_obs > 0
    fit_net  = build_stg_fit(stage.k, g_warm; includePem=includePem)
    fit_sys  = mtkcompile(fit_net.sys)
    fit_prob = ODEProblem(fit_sys, [], (0.0, simulation_length), jac=true, sparse=true)

    tunable_syms = Any[getproperty(getproperty(fit_sys, n), ch).g for (n, ch) in TUNABLE_PAIRS]
    setter = setp(fit_prob, tunable_syms)

    i_pem_getter = includePem ?
        Any[getu(fit_prob, getproperty(getproperty(fit_sys, n), :pem).i) for n in NEURONS] : Any[]

    v_syms = [fit_sys.AB.cap.v, fit_sys.PY.cap.v, fit_sys.LP.cap.v]      # AB, PY, LP
    idx_v = [variable_index(fit_prob, s) for s in v_syms]
    v_cont_getter = getu(fit_prob, v_syms)

    diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

    state_syms = unknowns(fit_sys)
    map_idx = [get(true_idx, string(s), 0) for s in state_syms]          # 0 => PEM-only state

    shoots, nshoots, stride, group_size = split_shoots(NSAMP, stage.n_shoots; overlap=overlap)
    probs = [remake(fit_prob; tspan=(timesteps[first(r)], timesteps[last(r)])) for r in shoots]
    u0s = map(shoots) do r
        u = copy(fit_prob.u0)
        for (k, m) in enumerate(map_idx)
            m != 0 && (u[k] = true_sol.u[first(r)][m])
        end
        u
    end

    flags   = (obs=stage.w_obs > 0, mse=stage.w_mse > 0, cont=stage.w_cont > 0)
    weights = (obs=stage.w_obs, mse=stage.w_mse, cont=stage.w_cont)
    refs    = (obs=Ref{Union{Nothing,Float64}}(nothing),
               mse=Ref{Union{Nothing,Float64}}(nothing),
               cont=Ref{Union{Nothing,Float64}}(nothing))

    return (; sys=fit_sys, prob=fit_prob, setter, i_pem_getter, v_cont_getter, idx_v, diffcache,
            state_syms, map_idx, shoots, nshoots, stride, group_size, probs, u0s,
            flags, weights, refs, v_data_true, includePem, tunable_syms)
end

# ============================================================
# Loss
# ============================================================

function loss_terms(θ, p)
    g_factor = G_SCALE .* exp.(θ)
    ABORT[] = false
    (; prob, probs, shoots, nshoots, u0s, setter, diffcache, i_pem_getter,
       v_cont_getter, idx_v, v_data_true, flags) = p

    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, θ)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = SciMLStructures.replace(Tunable(), ps, buffer)
    setter(ps, g_factor)

    prob_func(_, ectx) = remake(probs[ectx.sim_id]; p=ps, u0=u0s[ectx.sim_id], build_initializeprob=false)
    eprob = EnsembleProblem(probs[1]; prob_func=prob_func, output_func=flag_failures(ABORT))
    sim = solve(eprob, TRBDF2(), EnsembleThreads(); trajectories=nshoots,
                saveat=timesteps, callback=abort_cb, abstol=1e-9, reltol=1e-9)

    if ABORT[] || any(s -> !SciMLBase.successful_retcode(s.retcode), sim.u)
        return nothing
    end

    T = eltype(θ)
    obs = mse = cont = zero(T)
    for (i, r) in enumerate(shoots)
        sol = sim.u[i]
        if flags.obs
            I_pem = reduce(vcat, [g(sol) for g in i_pem_getter])
            obs += sum(abs2, I_pem) / length(I_pem)
        end
        if flags.mse || flags.cont
            V = v_cont_getter(sol)                       # per time point: [AB, PY, LP]
            length(V) == length(r) || error("shoot $i: got $(length(V)) samples, expected $(length(r))")
            if flags.mse
                s = zero(T)
                for (j, ti) in enumerate(r), k in 1:3
                    d = V[j][k] - v_data_true[k][ti]
                    s += d * d
                end
                mse += s / (3 * length(r))
            end
            if flags.cont && i < nshoots
                jloc = first(shoots[i+1]) - first(r) + 1
                unext = u0s[i+1]
                s = zero(T)
                for (k, ix) in enumerate(idx_v)
                    d = V[jloc][k] - unext[ix]
                    s += d * d
                end
                cont += s / 3
            end
        end
    end
    return (obs = obs / nshoots, mse = mse / nshoots, cont = cont / max(nshoots - 1, 1))
end

function nterm(raw, ref::Ref)
    if ref[] === nothing
        v = ForwardDiff.value(raw)
        (isfinite(v) && v > 0) && (ref[] = Float64(v))
    end
    return (ref[] === nothing ? raw : raw / ref[], raw)
end

function shoot_loss_mt(x, p)
    EVAL_COUNT[] += 1
    t = loss_terms(x, p)
    t === nothing && return Inf
    fl, w, rf = p.flags, p.weights, p.refs

    o, o_raw = fl.obs  ? nterm(t.obs,  rf.obs)  : (0.0, 0.0)
    m, m_raw = fl.mse  ? nterm(t.mse,  rf.mse)  : (0.0, 0.0)
    c, c_raw = fl.cont ? nterm(t.cont, rf.cont) : (0.0, 0.0)
    tot = w.obs * o + w.mse * m + w.cont * c

    LATEST_LOSS_COMPONENTS[] = (obs   = w.obs  * Float64(ForwardDiff.value(o)),
                                mse   = w.mse  * Float64(ForwardDiff.value(m)),
                                cont  = w.cont * Float64(ForwardDiff.value(c)),
                                total = Float64(ForwardDiff.value(tot)),
                                tot_raw = Float64(ForwardDiff.value(o_raw + m_raw + c_raw)),
                                x     = Float64.(ForwardDiff.value.(x)))
    return tot
end

const PAIR_INDEX = Dict(pr => k for (k, pr) in enumerate(TUNABLE_PAIRS))

function resolve_alternate(stage)
    alt = stage.alternate
    if alt === false
        return (group_idx=Vector{Int}[], free=collect(1:n_params),
                segs=[(len=stage.max_iters, pidx=0, bits=(), active=collect(1:n_params))])
    end
    group_idx = Vector{Int}[]
    for grp in alt.groups
        ids = Int[]
        for pr in grp
            haskey(PAIR_INDEX, pr) || error("unknown (neuron, channel) pair $pr")
            push!(ids, PAIR_INDEX[pr])
        end
        push!(group_idx, ids)
    end
    all_ids = reduce(vcat, group_idx; init=Int[])
    @assert length(unique(all_ids)) == length(all_ids) "groups overlap"
    @assert all(length(b) == length(alt.groups) for b in alt.patterns)
    free = setdiff(1:n_params, all_ids)
    n_seg = min(alt.divider, stage.max_iters)
    base_len = stage.max_iters ÷ n_seg
    segs = map(1:n_seg) do s
        pidx = mod1(s, length(alt.patterns))
        bits = alt.patterns[pidx]
        active = sort!(reduce(vcat, (group_idx[i] for i in eachindex(bits) if bits[i] == 1); init=copy(free)))
        (len = s == n_seg ? stage.max_iters - (n_seg - 1) * base_len : base_len, pidx=pidx, bits=bits, active=active)
    end
    return (group_idx=group_idx, free=free, segs=segs)
end

function splice_theta(x, θ_base, active)
    T = eltype(x)
    θ = Vector{T}(undef, length(θ_base))
    @inbounds for k in eachindex(θ_base); θ[k] = T(θ_base[k]); end
    @inbounds for (j, k) in enumerate(active); θ[k] = x[j]; end
    return θ
end

shoot_loss_active(x, p) = shoot_loss_mt(splice_theta(x, p.θ_base, p.active), p)

# ============================================================
# Per-stage fit
# ============================================================

function make_callback(stage, stage_idx, nshoots, l0, x0, t0;
                       seg=1, nseg=1, active=nothing, θ_base=nothing,
                       seg_len=stage.max_iters, offset=0, shared_best=nothing, pidx=0, bits=())
    baseline = l0
    best   = shared_best === nothing ? Ref{Tuple{Float64,Vector{Float64}}}((l0, copy(x0))) : shared_best
    reason = Ref{Symbol}(:none)

    function cb(state, loss_val)
        u  = state.u
        Δu = CB_PREV_U[] === nothing ? fill(NaN, length(u)) : u .- CB_PREV_U[]
        CB_PREV_U[] = copy(u)
        ufull = active === nothing ? u : splice_theta(u, θ_base, active)
        elapsed = (time() - t0) / 60
        comps = LATEST_LOSS_COMPONENTS[]
        fresh = comps.x == ufull
        if fresh && isfinite(comps.total) && comps.total < best[][1]
            best[] = (comps.total, copy(ufull))
        end
        spe = (time() - t0) / max(EVAL_COUNT[], 1)
        done = false
        if fresh && isfinite(stage.y) && comps.total <= baseline / stage.y
            reason[] = :factor; done = true
        elseif state.iter >= seg_len
            reason[] = seg == nseg ? :max_iters : :segment; done = true
        end

        seginfo = active === nothing ? "" :
            " seg $seg/$nseg pattern $pidx bits=$bits nact=$(length(active)) stage-iter $(offset + state.iter)"
        msg = """
        "[iter $(state.iter): total(cb)=$(round(loss_val,digits=6)) total(raw)=$(round(comps.tot_raw, digits=6)) obs=$(round(comps.obs, digits=6)) mse=$(round(comps.mse, digits=6)) cont=$(round(comps.cont, digits=6))$(fresh ? "" : "  [components stale]")
        g = $(round.(exp.(ufull[1:n_params]); digits=5))
        ‖Δu‖ = $(norm(Δu))   elapsed = $(round(elapsed, digits=2)) min   $(round(spe, digits=3)) s/eval   evals = $(EVAL_COUNT[])
        """
        if done
            msg *= "Stopped: $(reason[]) (iter $(state.iter), best total $(best[][1]) vs start $baseline)\n"
        end
        print(msg)
        open(io -> write(io, msg), "output.txt", "a")
        return done
    end
    return (cb=cb, best=best, reason=reason)
end

function fit_stage(ctx, stage, θ0, stage_idx)
    lb = fill(log(1e-5), n_params)
    ub = fill(log(150.0), n_params)
    x0 = clamp.(θ0, lb .+ 1e-3, ub .- 1e-3)

    ABORT[] = false
    CB_PREV_U[] = nothing
    EVAL_COUNT[] = 0
    for r in values(ctx.refs); r[] = nothing; end

    t0 = time()
    l0 = shoot_loss_mt(x0, ctx)
    if !isfinite(l0)
        println("Stage $stage_idx: loss is not finite at x0, aborting stage.")
        return (res=nothing, θ=x0, stopped_by=:failed, n_switches=0)
    end

    use_active = stage.alternate !== false
    sch = resolve_alternate(stage)
    nseg = length(sch.segs)
    best = Ref{Tuple{Float64,Vector{Float64}}}((l0, copy(x0)))
    n_switches = 0
    offset = 0
    stopped_by = :max_iters
    res = nothing
    println("Stage $stage_idx: starting Fminbox(BFGS), nshoots=$(ctx.nshoots), group_size=$(ctx.group_size), l0=$l0, segments=$nseg")

    for (s, sg) in enumerate(sch.segs)
        CB_PREV_U[] = nothing
        ABORT[] = false
        θ_base = copy(best[][2])                    # hand-off: stage-wide best
        if use_active
            pseg  = merge(ctx, (θ_base=θ_base, active=sg.active))
            lbs, ubs = lb[sg.active], ub[sg.active]
            xs0 = clamp.(θ_base[sg.active], lbs .+ 1e-3, ubs .- 1e-3)
            f = shoot_loss_active
        else
            pseg, lbs, ubs, xs0, f = ctx, lb, ub, x0, shoot_loss_mt
        end
        cbs = make_callback(stage, stage_idx, ctx.nshoots, l0, x0, t0;
                            seg=s, nseg=nseg, active=use_active ? sg.active : nothing, θ_base=θ_base,
                            seg_len=sg.len, offset=offset, shared_best=best, pidx=sg.pidx, bits=sg.bits)
        optfn   = OptimizationFunction(f, AutoForwardDiff())
        optprob = OptimizationProblem(optfn, xs0, pseg; lb=lbs, ub=ubs)
        res = solve(optprob, Fminbox(BFGS()); maxiters=sg.len, callback=cbs.cb)

        r = cbs.reason[]
        if r == :factor
            stopped_by = :factor; break
        elseif r == :none && !isfinite(res.objective)
            stopped_by = :failed; break
        end
        s < nseg && (n_switches += 1)
        offset += sg.len
    end

    θ_final = best[][2]
    println("Stage $stage_idx finished: stopped_by=$stopped_by obj=$(res.objective) retcode=$(res.retcode) switches=$n_switches")
    return (res=res, θ=θ_final, stopped_by=stopped_by, n_switches=n_switches)
end

# ============================================================
# Outer loop
# ============================================================

function run_schedule(stages, g_start)
    g_warm = g_start
    ctx = nothing
    θ = theta_from_g(g_start)
    for (i, st) in enumerate(stages)
        println("\n=== Stage $i / $(length(stages)) ===  ", st)
        ctx = build_stage(st, g_warm)
        println("flags=$(ctx.flags) includePem=$(ctx.includePem) nshoots=$(ctx.nshoots) stride=$(ctx.stride) group_size=$(ctx.group_size) n_states=$(length(ctx.prob.u0))")
        out = fit_stage(ctx, st, theta_from_g(g_warm), i)
        if out.stopped_by == :failed
            println("Stage $i FAILED; stopping the schedule (no silent continuation with a broken warm start).")
            break
        end
        θ = out.θ
        g_warm = theta_to_g(θ)
        s = g_to_string(g_warm)
        write("stage_$(i)_guess.txt", s * "\n")
        open(io -> write(io, "\n--- stage $i ($(out.stopped_by)) ---\n" * s * "\n"), "output.txt", "a")

        try
            display(plot_ms(ctx, θ, title="Stage $i: $(out.stopped_by)"))
            display(plot_continuous(θ))
        catch e
            @warn "plotting after stage $i failed" exception=(e, catch_backtrace())
        end
    end
    return (ctx=ctx, θ=θ, g=g_warm)
end

# ============================================================
# Validation plots
# ============================================================

function plot_ms(ctx, θ; abstol=1e-9, reltol=1e-9, title="")
    (; shoots, probs, u0s, setter, v_cont_getter, prob) = ctx
    overlap = 1
    ps = deepcopy(parameter_values(prob)) 
    setter(ps, G_SCALE .* exp.(θ[1:n_params]))

    t_sp = Float64[]
    V_sp = [Float64[] for _ in 1:3] 
    gaps = Float64[]
    prev_end = nothing

    for (i, r) in enumerate(shoots)
        pr  = remake(probs[i]; p=ps, u0=u0s[i], build_initializeprob=false)
        sol = solve(pr, TRBDF2(); saveat=timesteps[r], abstol=abstol, reltol=reltol)
        SciMLBase.successful_retcode(sol.retcode) || (@warn "shoot $i failed"; continue)

        V = v_cont_getter(sol)
        keep = i == 1 ? (1:length(V)) : (overlap+1:length(V))
        append!(t_sp, collect(timesteps[r])[keep])
        for k in 1:3
            append!(V_sp[k], [v[k] for v in V[keep]])
        end
        prev_end === nothing || push!(gaps, norm(prev_end .- V[1]))
        prev_end = V[end]
    end
    isempty(gaps) || println("max/mean endpoint gap: ", maximum(gaps), " / ", sum(gaps) / length(gaps))

    names  = ["AB", "PY", "LP"]
    cols   = ["black", "blue", "red"]
    traces = GenericTrace[]
    for k in 1:3
        push!(traces, scatter(x=timesteps, y=v_data_true[k], mode="lines", name="True $(names[k])",
                              line=attr(color=cols[k], width=2)))
        push!(traces, scatter(x=t_sp, y=V_sp[k], mode="lines", name="Spliced $(names[k])",
                              line=attr(color=cols[k], width=2, dash="dot")))
    end
    plot_title = isempty(title) ? "Spliced multiple-shooting segments (nshoots=$(ctx.nshoots))" : title
    layout = Layout(title=plot_title,
                    xaxis_title="Time (ms)", yaxis_title="V (mV)")
    return PlotlyJS.plot(traces, layout)
end

function plot_continuous(θ)
    val_prob = ODEProblem(true_sys, [], (0.0, simulation_length), jac=true, sparse=true)
    true_syms = Any[getproperty(getproperty(true_sys, n), ch).g for (n, ch) in TUNABLE_PAIRS]
    val_setter = setp(val_prob, true_syms)
    val_ps = deepcopy(parameter_values(val_prob))
    val_setter(val_ps, G_SCALE .* exp.(θ[1:n_params]))
    val_sol = solve(remake(val_prob; p=val_ps), TRBDF2(); saveat=timesteps)

    names = ["AB", "PY", "LP"]
    cols  = ["black", "blue", "red"]
    syms  = [true_sys.AB.cap.v, true_sys.PY.cap.v, true_sys.LP.cap.v]
    traces = GenericTrace[]
    for k in 1:3
        push!(traces, scatter(x=timesteps, y=v_data_true[k], mode="lines", name="True $(names[k])",
                              line=attr(color=cols[k], width=2)))
        push!(traces, scatter(x=timesteps, y=val_sol[syms[k]], mode="lines", name="Recovered $(names[k])",
                              line=attr(color=cols[k], width=2, dash="dash")))
    end
    layout = Layout(title="Single continuous run, recovered conductances (no PEM)",
                    xaxis_title="Time (ms)", yaxis_title="V (mV)")
    return PlotlyJS.plot(traces, layout)
end

# ============================================================
# Main
# ============================================================

trace_AB = scatter(x=timesteps, y=V_data_AB, mode="lines", name="True AB", line=attr(color="black", width=2))
trace_PY = scatter(x=timesteps, y=V_data_PY, mode="lines", name="True PY", line=attr(color="blue", width=2))
trace_LP = scatter(x=timesteps, y=V_data_LP, mode="lines", name="True LP", line=attr(color="red", width=2))
display(PlotlyJS.plot([trace_AB, trace_PY, trace_LP],
                      Layout(title="Ground truth", xaxis_title="Time (ms)", yaxis_title="V (mV)")))

stages = [merge(s, (max_iters=2,)) for s in STAGES] : STAGES
result = run_schedule(stages, guess_g)
if result.ctx !== nothing
    display(plot_ms(result.ctx, result.θ))
    display(plot_continuous(result.θ))
end