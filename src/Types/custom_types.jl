@enum SynapseType Exc Inh Custom

struct CustomSynapseParams
    E::Float64
    Vth::Float64
    k_::Float64
    sigma::Float64
    
    function CustomSynapseParams(E, Vth, k_, sigma)

        k_ > 0 || throw(ArgumentError("k_ must be positive"))
        sigma > 0 || throw(ArgumentError("sigma must be positive"))
        new(E, Vth, k_, sigma)
    end

end