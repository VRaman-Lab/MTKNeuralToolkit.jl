using SafeTestsets

using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Test

top = Scalar()

function build_hh_neuron(name::Symbol; gNa=120.0, ENa=50.0, gK=36.0, EK=-77.0, gleak=0.3, Eleak=-54.4)
    @named cap  = Capacitor(topology=top, C=1.0) 
    @named na   = SodiumChannel(topology=top, g=gNa, E_rev=ENa)
    @named k    = PotassiumChannel(topology=top, g=gK, E_rev=EK)
    @named leak = LeakChannel(topology=top, g=gleak, E_rev=Eleak)
    
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

@testset "Directed Exponential Synapse" begin
    pre_neuron  = build_hh_neuron(:pre_neuron)
    post_neuron = build_hh_neuron(:post_neuron; gNa=80.0, ENa=45.0, gleak=0.5)

    @named excitatory_synapse = ExpSynapse(g_max=2.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

    synapse_specs = [
        SynapseSpec(
            pre_neuron.interfaces.V,
            post_neuron.interfaces.V,
            post_neuron.interfaces.I_syn,
            excitatory_synapse
        )
    ]

    @named current_driver = Sine(amplitude=8.0, frequency=0.05, offset=8.0)
    drivers = [(1, current_driver)] 

    net = build_acausal_network([pre_neuron, post_neuron]; 
                                synapse_specs=synapse_specs, 
                                drivers=drivers, 
                                name=:two_neuron_net)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 200.0))
    sol = solve(prob, Rosenbrock23())

    @test sol.retcode == ReturnCode.Success
    
    pre_V = sol[sys.pre_neuron.cap.v]
    post_V = sol[sys.post_neuron.cap.v]
    I_syn = sol[sys.excitatory_synapse.I_syn]

    @test all(!isnan, pre_V)
    @test all(!isnan, post_V)
    @test all(!isnan, I_syn)

    # Pre neuron should spike due to sine drive
    @test maximum(pre_V) > 0.0
    # Synaptic current should be active at some point
    @test maximum(abs.(I_syn)) > 0.0
    # Post neuron should spike due to synaptic input
    @test maximum(post_V) > 0.0
end

@testset "STDP Synapse Plasticity" begin
    pre_neuron  = build_hh_neuron(:pre_neuron_stdp)
    post_neuron = build_hh_neuron(:post_neuron_stdp)

    @named stdp_syn = STDPSynapse(g_max=2.0, E_rev=0.0, V_th=0.0, slope=2.0,
                                  τ_s=5.0, τ_plus=20.0, τ_minus=20.0, 
                                  A_plus=0.1, A_minus=0.1, w_init=0.5, w_max=1.0, w_min=0.0)

    synapse_specs = [
        SynapseSpec(
            pre_neuron.interfaces.V, 
            post_neuron.interfaces.V, 
            post_neuron.interfaces.I_syn, 
            stdp_syn
        )
    ]

    @named pre_driver  = Sine(amplitude=8.0, frequency=0.2, offset=8.0, phase=0.0)
    @named post_driver = Sine(amplitude=8.0, frequency=0.2, offset=8.0, phase=1.0)

    drivers = [
        (1, pre_driver),
        (2, post_driver)
    ]

    net = build_acausal_network([pre_neuron, post_neuron]; 
                                synapse_specs=synapse_specs, 
                                drivers=drivers, 
                                name=:stdp_net)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 500.0), jac=true, sparse=true)
    sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

    @test sol.retcode == ReturnCode.Success
    
    w = sol[sys.stdp_syn.w]
    @test all(!isnan, w)
    
    # Since post fires before pre, we expect LTD (weight decreases)
    @test w[end] < w[1]
end

