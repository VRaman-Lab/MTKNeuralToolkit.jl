using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface: getu
using Plots
using PlotlyJS


top = Scalar()

function build_hh_neuron(name::Symbol; gNa=120.0, gK=36.0, gleak=0.3)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top, g=gNa)
    @named k    = PotassiumChannel(topology=top, g=gK)
    @named leak = LeakChannel(topology=top, g=gleak)
    channels = [na, k, leak]
    return build_compartment(cap, channels; name=name, V_init=-65.0, topology=top)
end

# Canonical true parameters
true_gNa = 120.0
true_gK  = 36.0
true_gleak = 0.3
test_gNa=200
test_gK=128
test_gleak=7.5#, with shoot 600 loss_weights = [0.0, 0.4, 0.8] 
# Manually entered test parameters
#test_gNa = 53.44897075594032
#test_gK  = 48
#test_gleak = 0.00293
#[199.99999999999835, 128.06415813564846, 7.5241200825026215] WITHOUT PEM, group size 80, loss_weights =  [0.1, 0.5, 1.0]

# Build both systems
drivers = [(1, 8.0)]
END_TIME = 100.0
timesteps = 0.0:0.1:END_TIME

# True system
true_neuron = build_hh_neuron(:true_neuron; gNa=true_gNa, gK=true_gK, gleak=true_gleak)
true_net = build_acausal_network([true_neuron]; drivers=drivers, name=:true_net)
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, END_TIME))
true_sol = solve(true_prob, Rodas5(); saveat=timesteps, reltol=1e-8, abstol=1e-8)
V_true = true_sol[true_sys.true_neuron.cap.v]

# Test system
test_neuron = build_hh_neuron(:test_neuron; gNa=test_gNa, gK=test_gK, gleak=test_gleak)
test_net = build_acausal_network([test_neuron]; drivers=drivers, name=:test_net)
test_sys = mtkcompile(test_net.sys)
test_prob = ODEProblem(test_sys, [], (0.0, END_TIME))
test_sol = solve(test_prob, Rodas5(); saveat=timesteps, reltol=1e-8, abstol=1e-8)
V_test = test_sol[test_sys.test_neuron.cap.v]

trace_true = scatter(x=timesteps, y=V_true, mode="lines",
    name="True (gNa=$true_gNa, gK=$true_gK, gleak=$true_gleak)",
    line=attr(color="black", width=2.5))

trace_test = scatter(x=timesteps, y=V_test, mode="lines",
    name="Test (gNa=$test_gNa, gK=$test_gK, gleak=$test_gleak)",
    line=attr(color="red", width=2.5, dash="dash"))

layout = Layout(
    title="Membrane Potential: True vs Test Parameters",
    xaxis_title="Time (ms)",
    yaxis_title="V (mV)",
    legend=attr(orientation="h", y=1.1)
)

p = Plot([trace_true, trace_test], layout)
display(p)