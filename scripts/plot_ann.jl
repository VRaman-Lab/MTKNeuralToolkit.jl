import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum, RealInputArray, RealOutputArray
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

using ModelingToolkit
using OrdinaryDiffEq
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.RMM as RMM
import MTKNeuralToolkit
using Plots

@mtkmodel ANNFred begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        τ = 1e-3
    end
    @components begin
        nn_in = RealInputArray(nin = 1)
        nn_out = RealOutputArray(nout = 1)
        nn = NeuralNetworkBlock(n_input = 1, n_output = 1; 
                                chain = multi_layer_feed_forward(1, 1, width=5))
    end
    @equations begin
        v ~ nn_in.u[1]
        
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * nn_out.u[1] * v
    end
end
ANNGates(;name=:conductance, kwargs...) = ANNFred(;name, kwargs...)

Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)
@named ann_gates = ANNGates(g=0.3, E=-65;name=:conductance)
ann_channel = build_channel(ann_gates, FixedReversal(E=-65); name =:Ann)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)
println("________________")
neur = build_neuron(fn, inp; channels = [ann_channel, Na, K, Leak])
neur_c = structural_simplify(neur) 
prob = ODEProblem(neur_c, Pair[], (0.0, 200.0) )
sol = solve(prob, Rodas5());

plot(sol)
#=p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(5,1), subplot=1)
plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
plot!(p, sol, idxs=[neur.Ann.conductance.n_gate], subplot=3)
plot!(p, sol, idxs=[neur.soma.v], subplot=4)=#
