# ==============================================================================
# Example 1: Building a Single-Compartment Hodgkin-Huxley Neuron from Scratch
# ==============================================================================
# 
# This example introduces the core acausal electrical primitives of 
# MTKNeuralToolkit. It demonstrates how to define standard ion channel 
# gating dynamics from first principles and assemble them into a single 
# point-neuron compartment.
#
# To run this, you will need to activate the examples environment: launch julia from the terminal (in root directory of package) as `julia --project=examples`


using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, t_nounits as t
using OrdinaryDiffEq
using Plots




# ------------------------------------------------------------------------------
# 1. Define Ion Channel Gating Dynamics
# ------------------------------------------------------------------------------
# In the HH formalism, gating variables (m, h, n) evolve according to 
# transition rates: alpha (opening) and beta (closing).
# We define these as standard Julia functions. Note the use of broadcasting (.+)
# so these exact same functions work for both scalar and vectorized topologies.

hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),  # alpha_m
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))   # beta_m
)

hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),                          # alpha_h
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0) # beta_h
)

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),   # alpha_n
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))   # beta_n
)

# ------------------------------------------------------------------------------
# NOTE: Using Infinity/Tau Formulations
# ------------------------------------------------------------------------------
# Many modern neuroscience papers define gating dynamics using steady-state 
# (infinity) functions and time constants, rather than alpha/beta rates.
# MTKNeuralToolkit provides the `InfTau` helper to automatically convert these 
# into the alpha/beta formulation used internally:
#   alpha = inf / tau
#   beta  = (1 - inf) / tau
#
# For example, if a paper defined a K+ n-gate like this:
#   n_inf(v) = 1.0 / (1.0 + exp((v + 50.0) / -5.0))
#   n_tau(v) = 5.0
#
# You could define it  using `InfTau` instead of writing alpha/beta:
#
#   hh_k_n_alt = InfTau(v -> 1.0 ./ (1.0 .+ exp.((v .+ 50.0) ./ -5.0)), 
#                       v -> 5.0)
#   potassium_gates_alt = [GateSpec(:n, 4, 0.0, hh_k_n_alt)]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 2. Define Gate Specifications
# ------------------------------------------------------------------------------
# A GateSpec bundles the gating dynamics with its symbolic name, the power it 
# is raised to in the conductance equation, and its initial condition (ic).
# For HH: Na+ is m^3 * h, and K+ is n^4.

sodium_gates = [
    GateSpec(:m, 3, 0.0, hh_na_m), 
    GateSpec(:h, 1, 0.0, hh_na_h)
]

potassium_gates = [
    GateSpec(:n, 4, 0.0, hh_k_n)
]

# ------------------------------------------------------------------------------
# 3. Build the Electrical Components
# ------------------------------------------------------------------------------
# We instantiate the capacitor and our ion channels.
# `GenericChannel` automatically hooks up the acausal OnePort and generates 
# the correct ODEs for the gating variables under the hood.

top = Scalar() # Define a scalar (single-neuron) topology

@named soma_cap = Capacitor(topology=top, C=1.0)

@named na_ch = GenericChannel(topology=top, g=120.0, E_rev=50.0,  gates=sodium_gates)
@named k_ch  = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
@named leak  = GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[]) # No gates = pure leak

# ------------------------------------------------------------------------------
# 4. Assemble the Compartment
# ------------------------------------------------------------------------------
# `build_compartment` connects all the positive pins to the membrane voltage,
# grounds the negative pins, and injects the currents. It returns a `Compartment`
# struct containing the MTK System and exposed interfaces.

channels = [na_ch, k_ch, leak]

soma = build_compartment(soma_cap, channels; 
                         name=:soma, 
                         V_init=-65.0, 
                         topology=top)

# ------------------------------------------------------------------------------
# 5. Build and Simulate the Network
# ------------------------------------------------------------------------------
# A single neuron is just a network with one node. We drive it with a constant
# 10.0 mA current injection and solve.


drivers = [(1, 20.0)] # (Index 1, Current 20.0)

# Or make a more complex input with library blocks, which you can add multiply etc. If lazy, you can @register your own julia function for an input.
# using ModelingToolkitStandardLibrary.Blocks: Sine
# @named current_driver = Sine(amplitude=5.0, frequency=0.5, offset=10.0)

# Pass the sine block block to the drivers list
# drivers = [(1, current_driver)]

net = build_acausal_network([soma]; drivers=drivers, name=:single_neuron)

println("Compiling single compartment...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 100.0))

println("Solving...")
sol = solve(prob, Rosenbrock23())

# ------------------------------------------------------------------------------
# 6. Plot the Results
# ------------------------------------------------------------------------------
plot(sol, idxs=[sys.soma.soma_cap.v], 
     title="Example 1: Single-Compartment HH Neuron", 
     ylabel="Membrane Potential (mV)", 
     xlabel="Time (ms)", 
     legend=false)
