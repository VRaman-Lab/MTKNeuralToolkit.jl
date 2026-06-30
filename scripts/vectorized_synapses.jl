using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

N = 30

# === Build vectorized HH compartment ===
@named soma = Capacitor(N=N, C=1.0)

hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)
sodium_gates = [GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 0.0, hh_na_h)]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)
potassium_gates = [GateSpec(:n, 4, 0.0, hh_k_n)]

@named sodium_channel = GenericChannel(N=N, g=120.0, E_rev=50.0, gates=sodium_gates)
@named potassium_channel = GenericChannel(N=N, g=36.0, E_rev=-77.0, gates=potassium_gates)
@named leak_channel = GenericChannel(N=N, g=0.3, E_rev=-54.4, gates=GateSpec[])

hh = build_compartment(soma, [sodium_channel, potassium_channel, leak_channel];
                        name=:hh, V_init=-65.0, N=N)

# === Create Vectorized Synapses ===
# We store the weights directly in the matrix W. g_max defaults to 1.0 in the synapse.
W_exc = zeros(N, N)
W_exc[2, 1] = 2.0  # 1->2
W_exc[2, 4] = 1.5  # 4->2
W_exc[3, 1] = 1.0  # 1->3
W_exc[7, 3] = 2.0  # 3->7
W_exc[8, 2] = 1.5  # 2->8

W_inh = zeros(N, N)
W_inh[3, 5] = 1.2  # 5->3

# Use build_synapse_block to create a single vectorized component for each type
syn_exc = build_synapse_block(hh, hh, W_exc; name=:syn_exc, E_rev=0.0)
syn_inh = build_synapse_block(hh, hh, W_inh; name=:syn_inh, E_rev=-80.0)

synapse_specs = [syn_exc, syn_inh]

# === Drivers: graded current ===
drivers = [(1, collect(Float64, 1:N))]

# === Build via refactored build_acausal_network ===
net = build_acausal_network([hh]; synapse_specs=synapse_specs, drivers=drivers)

@time net_compiled = mtkcompile(net.sys)
prob = ODEProblem(net_compiled, [], (0.0, 50.0))
@time sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=[net_compiled.hh.soma.v...],
          title="Vectorized Synapse API", xlabel="Time", ylabel="V (mV)")

p2 = plot(sol, idxs=[net_compiled.hh.soma.v[1], net_compiled.hh.soma.v[4], net_compiled.hh.soma.v[2]],
          label=["Pre 1" "Pre 4" "Post 2"],
          title="Convergent 1→2 + 4→2", xlabel="Time", ylabel="V (mV)")

plot(p1, p2, layout=(2,1), size=(800,500))
