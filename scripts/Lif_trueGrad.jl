
using ModelingToolkit
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit 
import MTKNeuralToolkit.IntegrateAndFire as IaF
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit
#using script_utils.jl
using Plots

@named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
@named inp2 = TimeVaryingFunction(f = t ->  ifelse((t > 20) & (t < 30),20.0, 0.0))
neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(inp2;name=:IF2),
    build_LIF(;name=:IF3),
    build_LIF(;name=:IF4),
    build_LIF(;name=:IF5),
    build_LIF(;name=:IF6),
    build_LIF(;name=:IF7),
    build_LIF(;name=:IF8),
    build_LIF(;name=:IF9),
    build_LIF(;name=:IF10)

    
]
connections = Dict(
    (1, 2) => [(type=:LIF, weight=3.0)],
    (2, 3) => [(type=:LIF, weight=10.0)],
    (3, 4) => [(type=:LIF, weight=2.0)],
    (4, 5) => [(type=:LIF, weight=1.0)],
    (5, 6) => [(type=:LIF, weight=5.0)],
    (6, 7) => [(type=:LIF, weight=5.0)],
    (7, 8) => [(type=:LIF, weight=5.0)],
    (8, 9) => [(type=:LIF, weight=5.0)],
    (9, 10) => [(type=:LIF, weight=5.0)]
)


sys = build_network(connections, neurons)

prob = ODEProblem(sys, Pair[], (0.0, 200.0))

sol = solve(prob, Tsit5());





