@mtkmodel E_syn_gates begin
    @extend v,i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E = 0.0
        Vth = -35.0
        k_ = 0.025
        sigma = 5.0
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t)
        v_post(t)
    end
    @equations begin
        s_hat(v) ~ 1.0 / (1.0 + exp((Vth - v) / sigma))
        tau_s(v) ~ (1.0 - s_hat(v)) / k_
        D(s) ~ (1/tau_s(v)) * (s_hat(v) - s)
        i ~ g * s * (v_post - E)
    end
end

@mtkmodel I_syn_gates begin
    @extend v,i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E = -70.0
        Vth = -35.0
        k_ = 0.01
        sigma = 5.0
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t)
        v_post(t)
    end
    @equations begin
        s_hat(v) ~ 1.0 / (1.0 + exp((Vth - v) / sigma))
        tau_s(v) ~ (1.0 - s_hat(v)) / k_
        D(s) ~ (1/tau_s(v)) * (s_hat(v) - s)
        i ~ g * s * (v_post - E)
    end
end
