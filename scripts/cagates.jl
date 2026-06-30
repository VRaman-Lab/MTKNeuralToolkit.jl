using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

top = Scalar()

# Define a simple CaV gate (e.g., T-type calcium)
ca_t_m = (v) -> (1.0 / (1.0 + exp(-(v + 50.0) / 3.0)), 0.0)
ca_gates = [GateSpec(:m, 2, 0.0, ca_t_m)]

# Define a KCa gate (e.g., BK channel) that senses Ca
# Note the two-argument signature: (v, Ca)
kca_m = (v, Ca) -> (Ca / (Ca + 1.0), 0.1) 
kca_gates = [GateSpec(:m, 1, 0.0, kca_m)]

@named cap = Capacitor(topology=top, C=1.0)
@named cav = CaVChannel(topology=top, g=2.0, E_rev=120.0, gates=ca_gates)
@named kca = KCaChannel(topology=top, g=10.0, E_rev=-80.0, gates=kca_gates)
@named leak = GenericChannel(topology=top, g=0.3, E_rev=-54.4, gates=GateSpec[])

# Pass the CalciumTracker config!
cell = build_compartment(cap, [cav, kca, leak]; 
                         name=:cell, V_init=-65.0, topology=top, 
                         ion_config=CalciumTracker(tau_Ca=50.0))

drivers = [(1, 10.0)]
net = build_acausal_network([cell]; drivers=drivers)
net_compiled = mtkcompile(net.sys)

prob = ODEProblem(net_compiled, [], (0.0, 100.0))
sol = solve(prob, Rosenbrock23())

p1 = plot(sol, idxs=net_compiled.cell.cap.v, title="Membrane Potential")
p2 = plot(sol, idxs=net_compiled.cell.cell_ca_pool.Ca, title="Calcium Concentration")

plot(p1, p2, layout=(2,1), size=(800,500))
