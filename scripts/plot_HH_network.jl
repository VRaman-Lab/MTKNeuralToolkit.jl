import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.Synapse as Syn
import MTKNeuralToolkit.HodgkinHuxley as HH
#using script_utils.jl
using Plots

function build_HH(input=TimeVaryingFunction(f=t -> 0)) begin

    Na = build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
    K = build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
    Leak = build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)

    fn=BasicSoma(; C=1, name = :soma)

    neur = build_neuron(fn, channels = [Na, K, Leak], input = input)
    return(neur) 
end
end

@named inp = TimeVaryingFunction(f=t -> sin(t))

pre_neur = build_HH(inp)
post_neur = build_HH()

syn_channel = build_channel(Syn.E_syn_gates(;g=0.1, E = 0.0), FixedReversal(;E=0.0); name = :Syn)
synapse = build_synapse(syn_channel, pre_neur, post_neur)
#TO FINISH