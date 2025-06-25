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
using LinearAlgebra

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

@mtkmodel RMMBob begin
    @extend v, i = oneport = OnePort()
    @variables begin
        #lti_v[1:8](t) = zeros(8)
        lti_v(t) = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    end
    @parameters begin
        #A_diag[1:8] = [0.6065, 0.8465, 0.9048, 0.9310, 0.9512, 0.9834, 0.9900, 0.9929]
        #B[1:8] = [0.3935, 0.1535, 0.0952, 0.0690, 0.0488, 0.0166, 0.0100, 0.0071]
    end
    @equations begin
    #[D(lti_v[j]) ~ A_diag[j] * lti_v[j] + B[j] * v for j in 1:8]...    
    D(lti_v) ~ 0.5*lti_v + v
    i ~ lti_v    
    end
end

@mtkmodel RMMTed begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
    end
    @components begin
        nn_in = RealInputArray(nin = 8)
        nn_out = RealOutputArray(nout = 8)
        nn = NeuralNetworkBlock(n_input = 8, n_output = 8; 
                                chain = multi_layer_feed_forward(8, 8, width=5))
    end
    @equations begin        
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * sum(nn_out.u) * v
    end
end

@mtkmodel RMMGertha begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        lti_v(t) = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        A_diag[1:8] = diagm([0.6065, 0.8465, 0.9048, 0.9310, 0.9512, 0.9834, 0.9900, 0.9929])
        B[1:8] = [0.3935, 0.1535, 0.0952, 0.0690, 0.0488, 0.0166, 0.0100, 0.0071]
    end
    @components begin
        nn_in = RealInputArray(nin = 8)
        nn_out = RealOutputArray(nout = 8)
        nn = NeuralNetworkBlock(n_input = 8, n_output = 8; 
                                chain = multi_layer_feed_forward(8, 8, width=5))
    end
    @equations begin        
        [D(lti_v[j]) ~ A_diag[j] * lti_v[j] + B[j] * v for j in 1:8]...   
        nn_in ~ lti_v   
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * sum(nn_out.u)
    end
end
#LTI(;name=:LTI, kwargs...) = RMMBob(;name, kwargs...)
#ANN(;name=:ANN, kwargs...) = RMMTed(;name, kwargs...)
#lti = LTI()
#ann = ANN()
#RMM = build_RMM(lti, ann; name=:RMM)
#ANNGates(;name=:conductance, kwargs...) = ANNFred(;name, kwargs...)
RMMGates(;name=:condutance, kwargs...) = RMMGertha(;name, kwargs...)
Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)
@named RMM = RMMGates(g=0.3, E=-65;name=:conductance)
RMM_channel = build_channel(RMM, FixedReversal(E=-65); name =:RMM)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)
println("________________")
neur = build_neuron(fn, inp; channels = [RMM_channel, Na, K, Leak])
neur_c = structural_simplify(neur) 
prob = ODEProblem(neur_c, Pair[], (0.0, 200.0) )
sol = solve(prob, Rodas5());

plot(sol)
#=p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(5,1), subplot=1)
plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
plot!(p, sol, idxs=[neur.Ann.conductance.n_gate], subplot=3)
plot!(p, sol, idxs=[neur.soma.v], subplot=4)=#
