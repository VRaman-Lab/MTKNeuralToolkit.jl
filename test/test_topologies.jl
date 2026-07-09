using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit
using OrdinaryDiffEq
using SparseArrays
using Test

# Shared gating dynamics for vectorized populations
hh_na_m = v -> (
    0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
    -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
)
hh_na_h = v -> (
    0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
    0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
)
sodium_gates = [GateSpec(:m, 3, 0.0, hh_na_m), GateSpec(:h, 1, 0.0, hh_na_h)]

hh_k_n = v -> (
    0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
    -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
)
potassium_gates = [GateSpec(:n, 4, 0.0, hh_k_n)]

function build_vector_pop(name::Symbol, top)
    gNa_heterogeneous = collect(range(119.0, 121.0, length=top.N)) 
    @named cap = Capacitor(topology=top, C=1.0)
    @named na  = GenericChannel(topology=top, g=gNa_heterogeneous, E_rev=50.0,  gates=sodium_gates)
    @named k   = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
    @named leak= GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[])
    
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

@testset "Vectorized E/I Network" begin
    N_E = 5 
    N_I = 3  
    top_E = Vectorized(N_E)
    top_I = Vectorized(N_I)

    pop_E = build_vector_pop(:pop_E, top_E)
    pop_I = build_vector_pop(:pop_I, top_I)

    W_EE = 0.5 .* rand(N_E, N_E)
    W_EI = 1.0 .* rand(N_I, N_E)
    W_IE = 2.0 .* rand(N_E, N_I)
    W_II = 1.0 .* rand(N_I, N_I)

    syn_EE = build_synapse_block(pop_E, pop_E, W_EE; name=:syn_EE, E_rev=0.0)
    syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)
    syn_IE = build_synapse_block(pop_I, pop_E, W_IE; name=:syn_IE, E_rev=-80.0)
    syn_II = build_synapse_block(pop_I, pop_I, W_II; name=:syn_II, E_rev=-80.0)

    synapse_specs = [syn_EE, syn_EI, syn_IE, syn_II]
    drivers = [(1, 25.0)]

    net = build_acausal_network([pop_E, pop_I]; synapse_specs=synapse_specs, drivers=drivers)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 100.0), jac=true, sparse=true)
    sol = solve(prob, Rosenbrock23())
    
    @test sol.retcode == ReturnCode.Success
    
    V_E = sol[sys.pop_E.cap.v]
    V_I = sol[sys.pop_I.cap.v]
    
    @test length(V_E[1]) == N_E
    @test length(V_I[1]) == N_I
    @test all(x -> all(!isnan, x), V_E)
    @test all(x -> all(!isnan, x), V_I)
end


@testset "Mixed Topologies (Scalar Hub & Vector Pop)" begin
    N_E = 3
    N_I = 2
    top_scalar = Scalar()
    top_E = Vectorized(N_E)
    top_I = Vectorized(N_I)

    function build_scalar_neuron(name::Symbol)
        @named cap  = Capacitor(topology=top_scalar, C=1.0)
        @named na   = SodiumChannel(topology=top_scalar)
        @named k    = PotassiumChannel(topology=top_scalar)
        @named leak = LeakChannel(topology=top_scalar)
        return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top_scalar)
    end

    hub_neuron = build_scalar_neuron(:hub_neuron)
    pop_E = build_vector_pop(:pop_E, top_E)
    pop_I = build_vector_pop(:pop_I, top_I)

    @named syn_hub_to_E = ExpSynapse(g_max=3.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @named syn_E_to_hub = ExpSynapse(g_max=5.0, τ=5.0, E_rev=-80.0, V_th=-20.0, slope=2.0)

    W_EI = 0.5 .* rand(N_I, N_E)
    syn_EI = build_synapse_block(pop_E, pop_I, W_EI; name=:syn_EI, E_rev=0.0)

    synapse_specs = [
        SynapseSpec(hub_neuron.interfaces.V, pop_E.interfaces.V[1], pop_E.interfaces.I_syn[1], syn_hub_to_E),
        SynapseSpec(pop_E.interfaces.V[1], hub_neuron.interfaces.V, hub_neuron.interfaces.I_syn, syn_E_to_hub),
        syn_EI
    ]

    drivers = [(1, 10.0)] # Constant drive to hub

    net = build_acausal_network([hub_neuron, pop_E, pop_I]; 
                                synapse_specs=synapse_specs, 
                                drivers=drivers, 
                                name=:mixed_net)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 100.0), jac=true, sparse=true)
    sol = solve(prob, Rosenbrock23())

    @test sol.retcode == ReturnCode.Success

    V_hub = sol[sys.hub_neuron.cap.v]
    V_E = sol[sys.pop_E.cap.v]
    V_I = sol[sys.pop_I.cap.v]

    @test all(!isnan, V_hub)
    @test all(x -> all(!isnan, x), V_E)
    @test all(x -> all(!isnan, x), V_I)
    
    @test maximum(V_hub) > 0.0
end
