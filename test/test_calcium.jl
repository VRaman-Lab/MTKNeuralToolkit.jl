using MTKNeuralToolkit
using ModelingToolkit
using OrdinaryDiffEq
using Test

# 1. Define Gating Dynamics
hh_na_m = v -> (
    0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),
    4.0 .* exp.(-(v .+ 65.0) ./ 18.0)
)
hh_na_h = v -> (
    0.07 .* exp.(-(v .+ 65.0) ./ 20.0),
    1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))
)
sodium_gates = [GateSpec(:m, 3, 0.052, hh_na_m), GateSpec(:h, 1, 0.596, hh_na_h)]

hh_k_n = v -> (
    0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)),
    0.125 .* exp.(-(v .+ 65.0) ./ 80.0)
)
potassium_gates = [GateSpec(:n, 4, 0.317, hh_k_n)]

CaV_m_inf(v) = 1.0 ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
CaV_tau_m(v) = 5.0 .+ 10.0 ./ (1.0 .+ exp.((v .+ 20.0) ./ 10.0))
CaV_dynamics = InfTau(CaV_m_inf, CaV_tau_m)
cav_gates = [GateSpec(:mCaV, 3, 0.0, CaV_dynamics)]

KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
KCa_tau_m(v) = 20
KCa_dynamics = InfTauCa(KCa_m_inf, KCa_tau_m)
kca_gates = [GateSpec(:mKCa, 4, 0.0, KCa_dynamics)]

top = Scalar()

@testset "Calcium Neuron Compartment" begin
    @named cap = Capacitor(topology=top, C=1.0)
    @named na  = GenericChannel(topology=top, g=100.0, E_rev=50.0,  gates=sodium_gates)
    @named k   = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
    @named leak= GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[])

    @named cav = CaVChannel(topology=top, g=2.0, gates=cav_gates, Ca_out=3000.0, 
                            nernst_factor=13.0, conversion_factor=0.047)

    @named kca = KCaChannel(topology=top, g=5.0, E_rev=-80.0, gates=kca_gates)

    ion_config = CalciumTracker(decay=200.0, Ca_init=0.05)

    neuron = build_compartment(cap, [na, k, leak, cav, kca]; 
                               name=:neuron, 
                               V_init=-65.0, 
                               topology=top, 
                               ion_config=ion_config)

    drivers = [(1, 15.0)] 
    net = build_acausal_network([neuron]; drivers=drivers, name=:ca_neuron)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 800.0), jac=true, sparse=true)
    sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

    @test sol.retcode == ReturnCode.Success

    V = sol[sys.neuron.cap.v]
    Ca = sol[sys.neuron.neuron_ca_pool.Ca]
    I_kca = sol[sys.neuron.kca.i]

    @test all(!isnan, V)
    @test all(!isnan, Ca)
    @test all(!isnan, I_kca)

    # Neuron should spike
    @test maximum(V) > 0.0
    
    # Calcium influx should cause the internal Ca concentration to rise above baseline
    @test maximum(Ca) > 0.05
    
    # KCa current should be active when Calcium is high
    @test maximum(abs.(I_kca)) > 0.0
end
