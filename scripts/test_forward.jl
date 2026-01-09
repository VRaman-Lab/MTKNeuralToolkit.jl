import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit
using Plots
using ForwardDiff

f(p) = p[1]^2+p[2]^3

function loss(p)
    return  2f(p) - f(p)^2
end

p0 = [40,40]
global _p = p0
for i in 1:10
    gr = ForwardDiff.gradient(loss, _p)
    global _p = _p - 0.01*gr
end

print(_p)

