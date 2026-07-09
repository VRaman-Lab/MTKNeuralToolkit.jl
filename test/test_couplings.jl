using SafeTestsets

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Test

top = Scalar()
# Define a tapering geometry: the soma is large, and distal dendrites are small.
areas = [0.0628, 0.0314, 0.0157, 0.0078, 0.0039] #cm^2

function build_passive_compartment(name::Symbol, area::Float64)
    geom = Geometry(area=area, C_m=1.0) #Geometry struct handles biophysical scaling
    @named cap  = Capacitor(topology=top, C=1.0, geometry=geom)
    @named leak = GenericChannel(topology=top, g=0.3, E_rev=-65.0, gates=GateSpec[], geometry=geom)
    
    return build_compartment(cap, [leak]; name=name, V_init=-65.0, topology=top)
end

@testset "Tapered Multi-Compartment Cable" begin
    N = 5
    cable = [build_passive_compartment(Symbol(:comp, i), areas[i]) for i in 1:N]

    coupling_specs = CouplingSpec[]
    for i in 1:(N-1)
        avg_area = (areas[i] + areas[i+1]) / 2.0
        R_axial = 1.0 / avg_area 
        
        # Note: you must give unique names to systems created in a loop.
        gj = GapJunction(R=R_axial; name=Symbol(:gj_, i))  
        push!(coupling_specs, CouplingSpec(cable[i], cable[i+1], gj))
    end

    # Inject a slow sinusoidal current only into the first compartment (the soma)
    @named current_driver = Sine(amplitude=5.0, frequency=0.05, offset=5.0)
    drivers = [(1, current_driver)] 

    net = build_acausal_network(cable; 
                                coupling_specs=coupling_specs, 
                                drivers=drivers, 
                                name=:cable_net)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 200.0))
    sol = solve(prob, Rosenbrock23())

    @test sol.retcode == ReturnCode.Success

    # Extract voltages for first and last compartments
    comp1_sys = getproperty(sys, :comp1)
    comp5_sys = getproperty(sys, :comp5)
    V1 = sol[getproperty(comp1_sys, :cap).v]
    V5 = sol[getproperty(comp5_sys, :cap).v]

    @test all(!isnan, V1)
    @test all(!isnan, V5)

    # Check for voltage attenuation: the driven compartment should have 
    # a larger voltage peak than the distal compartment.
    @test maximum(V1) > maximum(V5)
end
