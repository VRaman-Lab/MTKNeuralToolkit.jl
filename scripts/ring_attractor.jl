using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
using Plots

# ==========================================
# 1. Network Size & Topology Configuration
# ==========================================
const NUM_NEURONS = 3

function angular_distance(i, j, total)
    θ_i = (i - 1) * (2π / total)
    θ_j = (j - 1) * (2π / total)
    Δθ = abs(θ_i - θ_j)
    return Δθ > π ? 2π - Δθ : Δθ
end

# ==========================================
# 2. Eager Instantiation of Components
# ==========================================
println("Pre-allocating network components explicitly...")

# Constrain the comprehension type to System to bypass type-inference checks
neurons = System[build_compartment(LIFCapacitor(C = 1.0; name=:soma), []; name = Symbol(:neuron_, i)) for i in 1:NUM_NEURONS]

connections = Tuple{Int, Int, System}[]
synapses = System[]

for i in 1:NUM_NEURONS, j in 1:NUM_NEURONS
    if i != j
        dist = angular_distance(i, j, NUM_NEURONS)
        
        local_syn = if dist < (π / 4)
            g_max = 3.0 * cos(2 * dist) 
            gate = EventSynapseGate(g_max = g_max, τ = 5.0, v_th = -55.0, w = 1.0; name = Symbol(:gate_, i, :_to_, j))
            batt = FixedReversal(E = 0.0; name = Symbol(:batt_, i, :_to_, j))
            build_synapse(gate, batt; name = Symbol(:syn_, i, :_to_, j))
        else
            g_max = 0.5 * sin(dist)
            gate = EventSynapseGate(g_max = g_max, τ = 10.0, v_th = -55.0, w = 0.5; name = Symbol(:gate_, i, :_to_, j))
            batt = FixedReversal(E = -70.0; name = Symbol(:batt_, i, :_to_, j))
            build_synapse(gate, batt; name = Symbol(:syn_, i, :_to_, j))
        end
        
        push!(synapses, local_syn)
        push!(connections, (i, j, local_syn))
    end
end

stim1 = Blocks.Constant(k = 80.0; name = :kick_stim)
src1  = CurrentSource(; name = :kick_source)

# Type the drivers tuple vector explicitly to prevent any Vector{Any} fallback
drivers = Tuple{Int, System, System}[(1, stim1, src1)]

# ==========================================
# 4. Assembly & Compilation
# ==========================================
println("Assembling and compiling the explicit neural network system...")

@named ring_system = build_network(neurons, synapses, connections; drivers=drivers)
ring_compiled = mtkcompile(ring_system)

# ==========================================
# 5. Simulation & Seamless Plot Unpacking
# ==========================================
prob = ODEProblem(ring_compiled, [], (0.0, 50.0); warn_initialize_determined = false)
sol = solve(prob, Tsit5())

time_steps = 0.0:0.5:50.0
voltage_matrix = zeros(NUM_NEURONS, length(time_steps))

for (t_idx, t_val) in enumerate(time_steps)
    for n_idx in 1:NUM_NEURONS
        voltage_matrix[n_idx, t_idx] = sol(t_val, idxs=neurons[n_idx].V)
    end
end

heatmap(
    time_steps, 
    1:NUM_NEURONS, 
    voltage_matrix, 
    xlabel="Time (ms)", 
    ylabel="Neuron Index around Ring", 
    title="Eager Ring Attractor Dynamics",
    c=:viridis
)
