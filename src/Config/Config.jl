module Config
import ..MTKNeuralToolkit: IonicPort, IonicPin, IonicGround, IonicTerminal


using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D

include("neuron_defaults.jl")

end
