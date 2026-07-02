# ==============================================================================
# Example 8: Spike-Timing-Dependent Plasticity (STDP)
# ==============================================================================
# 
# This example introduces learning and plasticity. We connect two neurons using 
# the `STDPSynapse`, a continuous, smooth approximation of STDP. 
# 
# We drive the presynaptic neuron with a constant current, and the postsynaptic 
# neuron with a slightly different constant current so they spike at different 
# frequencies. The STDP rule will continuously strengthen or weaken the synaptic 
# weight based on their relative spike times. We will plot the dynamic weight 
# variable `w` to watch learning happen in real-time!

using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using Plots

# ------------------------------------------------------------------------------
# 1. Build Two Neurons
# ------------------------------------------------------------------------------
top = Scalar()

function build_neuron(name::Symbol)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top)
    @named k    = PotassiumChannel(topology=top)
    @named leak = LeakChannel(topology=top)
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

pre_neuron  = build_neuron(:pre_neuron)
post_neuron = build_neuron(:post_neuron)

# ------------------------------------------------------------------------------
# 2. Define the STDP Synapse
# ------------------------------------------------------------------------------
# The STDPSynapse continuously updates its weight `w` based on the traces `x` (pre) 
# and `y` (post). 
# - If pre spikes before post, x is high when y spikes -> LTP (weight increases)
# - If post spikes before pre, y is high when x spikes -> LTD (weight decreases)

@named stdp_syn = STDPSynapse(g_max=2.0, E_rev=0.0, V_th=0.0, slope=2.0,
                              τ_s=5.0, τ_plus=20.0, τ_minus=20.0, 
                              A_plus=0.1, A_minus=0.1, w_init=0.5, w_max=1.0, w_min=0.0)

synapse_specs = [
    SynapseSpec(
        pre_neuron.interfaces.V, 
        post_neuron.interfaces.V, 
        post_neuron.interfaces.I_syn, 
        stdp_syn
    )
]

# ------------------------------------------------------------------------------
# 3. Driving Stimuli
# ------------------------------------------------------------------------------
# Drive them at different rates to force out-of-phase spikes
drivers = [
    (1, 10.0), # Pre gets 10.0
    (2, 15.0)  # Post gets 15.0
]

# ------------------------------------------------------------------------------
# 4. Build and Simulate
# ------------------------------------------------------------------------------
net = build_acausal_network([pre_neuron, post_neuron]; 
                            synapse_specs=synapse_specs, 
                            drivers=drivers, 
                            name=:stdp_net)

println("Compiling STDP network...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 500.0), jac=true, sparse=true)

println("Solving...")
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

# ------------------------------------------------------------------------------
# 5. Plot the Results
# ------------------------------------------------------------------------------
p1 = plot(sol, idxs=[sys.pre_neuron.cap.v, sys.post_neuron.cap.v], 
          title="Neuron Voltages", ylabel="V (mV)", 
          labels=["Pre" "Post"], legend=:right)

# Plot the plastic weight variable! We reach into the synapse namespace to get `w`.
p2 = plot(sol, idxs=[sys.stdp_syn.w], 
          title="STDP Synaptic Weight (w)", 
          ylabel="Weight", xlabel="Time (ms)", legend=false)

# Plot the pre and post synaptic traces (x and y) which drive the weight update
p3 = plot(sol, idxs=[sys.stdp_syn.x, sys.stdp_syn.y], 
          title="STDP Activity Traces", 
          labels=["x (Pre)" "y (Post)"], legend=:right)

plot(p1, p2, p3, layout=(3,1), size=(800, 800))
