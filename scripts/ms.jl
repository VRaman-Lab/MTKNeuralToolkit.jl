using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron: PrinzGeometry, na_gates, cas_gates, cat_gates, ka_gates, kca_gates, kdr_gates, h_gates
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named, unknowns
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
import OrdinaryDiffEqSDIRK: TRBDF2
using Optimization
using OptimizationOptimJL
using MultistartOptimization
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp, getu, variable_index
using PreallocationTools
using DataInterpolations
using SciMLBase
using Plots
using Markdown
using Infiltrator
using Random
using DifferentiationInterface
using ForwardDiff
using HTTP
using Statistics


top = Scalar()

function build_hh_neuron(name::Symbol; gNa=120.0, gK=36.0, gleak=0.3, pem = false, itps=nothing, K=1.0)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top, g=gNa)
    @named k    = PotassiumChannel(topology=top, g=gK)
    @named leak = LeakChannel(topology=top, g=gleak)
    channels = [na, k, leak]
    if pem
        @named pem_ch = PEMObservationChannel(itps=itps, K_init=K, topology=top)
        push!(channels, pem_ch)
    end
    return build_compartment(cap, channels; name=name, V_init=-65.0, topology=top)
end

true_gNa = 120.0
true_gK  = 36.0
true_gleak = 0.3

true_neuron = build_hh_neuron(:true_neuron; gNa=true_gNa, gK=true_gK, gleak=true_gleak)

drivers = [(1, 8.0)]
true_net = build_acausal_network([true_neuron]; drivers=drivers, name=:true_net)
END_TIME= 120.0
GROUP_SIZE = 80
OVERLAP = 0
PRINT_SUMMARY = true
global READLINE_PAUSE = false
intra_shoot_losses, continuity_losses, obs_losses = Float64[0], Float64[0], Float64[0]

function pause(i=0, otherprint="")
    println(i == 0 ? "Here's a pause $otherprint" : "Here's a big pause")
    re = readline()
    if re == "exit"
        error("Exiting optimization loop.")
    elseif re == "pause"
        @infiltrate
    end
end
function split_shoots(timesteps, group_size=nothing; overlap=0)
    N = length(timesteps)
    nshoots(gs) = cld(N - overlap, gs - overlap)
    if isnothing(group_size)
        group_size = 10
        if nshoots(group_size) > 100
            group_size = overlap + cld(N - overlap, 100)
        end
    end
    @assert overlap * 4 <= group_size
    stride = group_size - overlap
    intermediate = [s:min(s + group_size - 1, N) for s in 1:stride:N]
    if length(intermediate[end]) < group_size && length(intermediate) > 1
        intermediate[end-1] = first(intermediate[end-1]):last(intermediate[end])
        pop!(intermediate)
    end
    return intermediate, length(intermediate)
end

function split_solution(sol, timesteps, group_size=nothing; overlap=0)
    shoots = split_shoots(timesteps, group_size, overlap=overlap)
    return ([(t=sol.t[r], u=sol.u[r], u0=sol.u[first(r)]) for r in shoots], shoots)
end

function shoot_endpoint_diffs(shoot_ids, Vs, shoots, V_data_clean, v_getter)
    ids = isempty(shoot_ids) ? keys(shoots) : shoot_ids
    Dict(i => begin
        d = ForwardDiff.value.(v_getter(Vs[i])[end] .- V_data_clean[shoots[i]][end])
        (diff=round.(d, digits=5), std=round(std(ForwardDiff.value.(v_getter(Vs[i]))), digits=5))
    end for i in ids)
end

function shoot_loss(x, p, readline_pause=false)
    prob, timesteps, g_getter, setter, diffcache, v_getter, group_size, overlap, n_params, i_pem_getter, loss_weights = p
    n_states = length(prob.u0)
    # SPLIT shoots
    shoots, nshoots = split_shoots(timesteps, group_size; overlap=overlap)
    len = length(timesteps)
    #GROUNDKEEPING
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    setter(ps, x)
    vals = ForwardDiff.value.(g_getter(ps))
    u0s = [x[n_params+(i-1)*n_states+1:n_params+i*n_states] for i in 1:length(shoots)]
    
    #pause(1)
    
    #LOOP SHOOTS
    intra_loss = 0
    cont_loss = 0
    obs_loss = 0
    Vs = [] #This and above useful for infiltrating and debugging
    for (i, r) in enumerate(shoots)
        ts = timesteps[r]
        sol = solve(remake(prob; p=ps, u0=u0s[i], tspan=(ts[1], ts[end])), Rodas5(); saveat=ts)
        push!(Vs, sol)
        if !SciMLBase.successful_retcode(sol.retcode)
            return Inf
        end
        V_fit = v_getter(sol)
        intra_loss += sum(abs2, V_fit .- V_data_clean[r])
        #I_pem = i_pem_getter(sol)

        i < length(shoots) && (cont_loss+=sum(abs2, sol.u[end] .- u0s[i+1]))
        #observer_effort = sum(abs2, I_pem) / length(I_pem)
        #obs_loss += observer_effort
    end
    adj_intra_loss, adj_cont_loss, obs_loss = intra_loss, cont_loss * GROUP_SIZE, obs_loss 
    tot_loss_value = loss_weights[1] * adj_intra_loss + loss_weights[2] * adj_cont_loss + loss_weights[3] * obs_loss

    println("gNa=$(vals[1]), gK=$(vals[2]), gleak=$(vals[3])")
    println(shoot_endpoint_diffs([1,6,10,15], Vs, shoots, V_data_clean, v_getter))
    println("Total loss: $(tot_loss_value.value), intra-shoot: $(adj_intra_loss.value), continuity: $(adj_cont_loss.value), observer: $(obs_loss)")
    return tot_loss_value
end

true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, END_TIME))
timesteps = 0.0:0.1:END_TIME
println(timesteps)
true_sol = solve(true_prob, Rodas5(); saveat=timesteps)
V_data_clean = true_sol[true_sys.true_neuron.cap.v];

_, nshoots = split_shoots(timesteps, GROUP_SIZE, overlap = OVERLAP)
print(nshoots)
# Add 3 mV Gaussian noise to the data
Random.seed!(42)
noise_level = 1.0 
V_data_noisy = V_data_clean .+ noise_level .* randn(length(V_data_clean))
itp_V = LinearInterpolation(V_data_noisy, timesteps);

guess_gNa = 10.0
guess_gK  = 10.0
guess_gleak = 10.0
toOptimize = [guess_gNa, guess_gK, guess_gleak]

fit_neuron = build_hh_neuron(:fit_neuron; gNa=guess_gNa, gK=guess_gK, gleak=guess_gleak, pem=false, itps=[itp_V], K=2.0)
fit_net = build_acausal_network([fit_neuron]; drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, END_TIME))

gNa_sym = fit_sys.fit_neuron.na.g
gK_sym  = fit_sys.fit_neuron.k.g
gleak_sym = fit_sys.fit_neuron.leak.g

setter = setp(fit_prob, [gNa_sym, gK_sym, gleak_sym])

v_getter = getu(fit_prob, fit_sys.fit_neuron.cap.v)
#i_pem_getter = getu(fit_prob, fit_sys.fit_neuron.pem_ch.i);
i_pem_getter = 0
g_getter = getu(fit_prob, [gNa_sym, gK_sym, gleak_sym])
state_syms = unknowns(fit_sys)

guess_ps = parameter_values(fit_prob)
setter(guess_ps, toOptimize)
guess_prob = remake(fit_prob; p=guess_ps)
guess_sol = solve(guess_prob, Rodas5(); saveat=timesteps)
shoots, nshoots = split_shoots(timesteps, GROUP_SIZE, overlap = OVERLAP)
u0_guesses = [guess_sol[state_syms, first(r)] for r in shoots]

x0 = vcat([guess_gNa, guess_gK, guess_gleak], reduce(vcat, u0_guesses))
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))
loss_weights = [0.1, 0.5, 1.0]  # Weights for intra-shoot, continuity, and observer losses

opt_params = (fit_prob, timesteps, g_getter, setter, diffcache, v_getter,GROUP_SIZE, 0, length(toOptimize), i_pem_getter, loss_weights)
adtype = AutoForwardDiff()

lb, ub = vcat([5.0, 5.0, 0.0], fill(-Inf, length(x0) - length(toOptimize))), vcat([200.0, 200.0, 20.0], fill(Inf, length(x0) - length(toOptimize))) 

optfn = OptimizationFunction(shoot_loss, adtype)
optprob = OptimizationProblem(optfn, x0, opt_params, lb=lb, ub=ub)

println("Starting optimization...")
res = solve(optprob, Fminbox(BFGS()); maxiters=5, show_every=1)

if PRINT_SUMMARY
    grad = DifferentiationInterface.gradient(x -> shoot_loss(x, opt_params), AutoForwardDiff(), x0)

    _, nshoots_final = split_shoots(timesteps, GROUP_SIZE; overlap=OVERLAP)
    println("""
    Final parameter estimates: $(res.u[1:3])
    Training summary:
    retcode, restat: $(res.retcode), $(res.stats)
    shoots: $nshoots_final (size $GROUP_SIZE, overlap $OVERLAP)
    iterations: $(res.stats.iterations)
    final intra-shoot loss: $(round(last(intra_shoot_losses)))
    final continuity loss:  $(round(last(continuity_losses)))
    final observer loss: $(round(last(obs_losses)))
    final objective: $(round(res.objective; digits=4))
    """)
end

fit_gNa, fit_gK, fit_gleak = res.u[1:3]
val_neuron = build_hh_neuron(:fit_neuron; gNa=fit_gNa, gK=fit_gK, gleak=fit_gleak)
val_net = build_acausal_network([val_neuron]; drivers=drivers, name=:val_net)
val_sys = mtkcompile(val_net.sys)
val_prob = ODEProblem(val_sys, [], (0.0, END_TIME))
val_prob = solve(val_prob, Rodas5(); saveat=timesteps)

init_ps = parameter_values(fit_prob)
setter(init_ps, [guess_gNa, guess_gK, guess_gleak])
fit_init_prob = remake(fit_prob; p=init_ps)
fit_init_sol = solve(fit_init_prob, Rodas5(); saveat=timesteps)

p1 = plot(timesteps, V_data_noisy, label="Noisy Target", color=:gray, lw=1, alpha=0.8)
plot!(p1, timesteps, V_data_clean, label="True Clean", color=:black, lw=2)
plot!(p1, timesteps, val_prob[val_sys.fit_neuron.cap.v], label="Fit", ls=:dash, lw=2, color=:red)
plot!(p1, timesteps, fit_init_sol[fit_sys.fit_neuron.cap.v], label="Init", ls=:solid, lw=2, color=:blue)
title!("Voltage Trace Recovery (Noisy Data)")

p = plot(p1, size=(800, 700), legend=:outertop)
xlabel!(p, "Time (ms)")
ylabel!(p, "V (mV)")
HTTP.post("https://ntfy.sh/elouanmsjltest"; body="Optimization finished: loss=$(res.objective), gNa=$(fit_gNa), gK=$(fit_gK), gleak=$(fit_gleak), intra-shoot loss=$(last(intra_shoot_losses)), continuity loss=$(last(continuity_losses)), observer loss=$(last(obs_losses))")