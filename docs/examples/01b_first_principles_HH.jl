# # Building Custom Ion Channels from Scratch
# 
# In Example 1, we used `GateSpec` and `GenericChannel` to build a Hodgkin-Huxley 
# neuron in just a few lines of code. But what if you need a channel that doesn't 
# fit the standard alpha/beta gating paradigm? What if you want to implement a 
# complex Markov chain model, or non-standard biophysics?
# 
# `GenericChannel` isn't a black box—it's just a convenience wrapper around standard ModelingToolkit components. 
# 
# The only rule for an ion channel in this toolkit is that it must be an MTK 
# system that extends a standard electrical `OnePort`. Let's prove it by 
# building a custom leak channel and a custom persistent sodium channel (NaP) 
# from first principles.
# 
# !!! note "Extending Synapses"
#     Synapses are very similar. You can look at the source code for e.g., exponential synapses and extend them similarly.

# ---

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, @variables, @parameters, t_nounits as t, D_nounits as D, extend, @component, @unpack, System
using ModelingToolkitStandardLibrary.Electrical: OnePort
using OrdinaryDiffEq
using Plots

# ## 1. A Custom Leak Channel
# 
# A leak channel has no states; it simply follows Ohm's law:
# ```math
# I = g(V - E)
# ```
# We define it as an MTK `@component` that wraps an `OnePort`. The steps are:
# 1. Instantiate the OnePort to get the voltage `v` and current `i`.
# 2. Define parameters `g` and `E_rev`.
# 3. Write the governing equation.
# 4. Extend the OnePort with our equations. This tells MTK that our component behaves as an electrical OnePort, which is exactly what `build_compartment` expects.

@component function CustomLeakChannel(; name, g=0.3, E_rev=-54.4)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters g=g E_rev=E_rev
    eqs = [i ~ g * (v - E_rev)]
    return extend(System(eqs, t, [], [g, E_rev]; name=name), oneport)
end

# ---

# ## 2. A Custom Persistent Sodium Channel (NaP)
# 
# Let's build a channel with a single gating variable, `m`, that does not 
# inactivate (persistent). We'll use a custom steady-state function and a 
# time constant, writing out the ODE manually.
#
# The gating ODE is:
# ```math
# \frac{dm}{dt} = \frac{m_{\infty} - m}{\tau}
# ```
# 
# And the current equation is:
# ```math
# I = g \cdot m^p \cdot (V - E)
# ```
# Let's use $p=1$ for NaP. We initialize the state variable `m` based on `V_init`.

@component function CustomNaPChannel(; name, g=10.0, E_rev=50.0, V_init=-65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    @variables m(t) = 1.0 / (1.0 + exp(-(V_init + 50.0) / 5.0))
    
    m_inf(V) = 1.0 / (1.0 + exp(-(V + 50.0) / 5.0))
    tau_m = 5.0
    
    eqs = [
        D(m) ~ (m_inf(v) - m) / tau_m,
        i ~ g * m * (v - E_rev)
    ]
    
    return extend(System(eqs, t, [m], [g, E_rev]; name=name), oneport)
end

# ---

# ## 3. Assemble the Compartment
# We instantiate our custom channels and a standard Capacitor.
# Note that `build_compartment` doesn't know or care that these are custom; 
# it just connects their positive pins to the membrane voltage and grounds the 
# negative pins.

top = Scalar() # Define a scalar topology

@named cap = Capacitor(topology=top, C=1.0)
@named leak = CustomLeakChannel()
@named nap = CustomNaPChannel()

# `build_compartment` handles the wiring just like in Example 1.
channels = [leak, nap]
soma = build_compartment(cap, channels; name=:soma, V_init=-65.0, topology=top)

# ---

# ## 4. Build and Simulate
# We'll inject a small current to see how the persistent sodium channel interacts 
# with the leak channel to create a steady depolarized state.

drivers = [(1, 2.0)]
net = build_acausal_network([soma]; drivers=drivers, name=:custom_neuron)

println("Compiling custom channel neuron...")
sys = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, 100.0))

println("Solving...")
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

# ---

# ## 5. Plot the Results
# We can access the variables of our custom components exactly as we would 
# with the pre-built channels, reaching into the compiled system's namespace.

p1 = plot(sol, idxs=[sys.soma.cap.v], 
          title="Example 0: Custom Channel Voltage", 
          ylabel="V (mV)", legend=false)

# Look at the gating variable `m` of our custom NaP channel!
p2 = plot(sol, idxs=[sys.soma.nap.m], 
          title="Custom NaP Gating Variable (m)", 
          ylabel="Fraction open", xlabel="Time (ms)", legend=false)

plot(p1, p2, layout=(2,1), size=(800, 600))

# > OK... nobody said it was going to be an interesting channel!
