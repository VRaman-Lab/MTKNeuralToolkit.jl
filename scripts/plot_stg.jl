import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkit
using OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Blocks: Constant, TimeVaryingFunction 
using ModelingToolkit: t_nounits as t, D_nounits as D
import MTKNeuralToolkit.Synapse as Synapse
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.Liu as Liu
import MTKNeuralToolkit.Prinz as Prinz
import MTKNeuralToolkit.Config as cfg
import MTKNeuralToolkit.Types: SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams
using MTKNeuralToolkit
include("script_utils.jl")
using Plots

@named inp2 = TimeVaryingFunction(f=t -> sin(t))
syn_cf = 0.254
prinz_cf = 159.2
neurons = Dict(
    "AB" => build_Prinz(inp2; name=:AB, config=cfg.PrinzConfig(
        V0=-60.0, Na_g=100.0*prinz_cf, CaS_g=6.0*prinz_cf, CaT_g=2.5*prinz_cf, 
        H_g=0.01*prinz_cf, K_g=50.0*prinz_cf, KCa_g=5.0*prinz_cf, DRK_g=100.0*prinz_cf, Leak_g=0.0*prinz_cf)),
    "PY" => build_Prinz(;name=:PY, config=cfg.PrinzConfig(
        V0=-55.0, Na_g=100.0*prinz_cf, CaS_g=2.0*prinz_cf, CaT_g=2.4*prinz_cf, 
        H_g=0.05*prinz_cf, K_g=50.0*prinz_cf, KCa_g=0.0*prinz_cf, DRK_g=125.0*prinz_cf, Leak_g=0.01*prinz_cf)),
    "LP" => build_Prinz(;name=:LP, config=cfg.PrinzConfig(
        V0=-65.0, Na_g=100.0*prinz_cf, CaS_g=4.0*prinz_cf, CaT_g=0.0*prinz_cf, 
        H_g=0.05*prinz_cf, K_g=20.0*prinz_cf, KCa_g=0.0*prinz_cf, DRK_g=25.0*prinz_cf, Leak_g=0.03*prinz_cf))
)

connections = Dict(
    ("AB", "LP") => [(type=:Chol, weight=30.0*syn_cf), (type=:Glut, weight=30.0*syn_cf)],
    ("AB", "PY") => [(type=:Chol, weight=3.0*syn_cf), (type=:Glut, weight=10.0*syn_cf)],
    ("LP", "AB") => [(type=:Glut, weight=30.0*syn_cf)],
    ("LP", "PY") => [(type=:Glut, weight=1.0*syn_cf)],
    ("PY", "LP") => [(type=:Glut, weight=30.0*syn_cf)],
)

@time network = build_network(connections, neurons)

@time prob = ODEProblem(network, Pair[], (0.0, 300.0) )
@time sol = solve(prob, TRBDF2());

p = plot(sol, idxs=parse_sol_for_membrane_voltages(sol), layout=(4,1), subplot=1,size=(1000, 800))
plot!(p, sol, idxs=[network.s_ABLP_Chol_1.AB.AB.Ca], subplot=2)
plot!(p, sol, idxs=[network.s_ABLP_Chol_1.LP.LP.Ca], subplot=3)
plot!(p, sol, idxs=[network.s_ABPY_Chol_1.PY.PY.Ca], subplot=4)
gui(p)