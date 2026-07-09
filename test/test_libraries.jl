using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: mtkcompile
using OrdinaryDiffEq
using Test

@testset "STG Network Integration Smoke Test" begin
    # build_stg() compiles the full 3-neuron Prinz network
    net = PrinzNeuron.build_stg()
    sys = mtkcompile(net.sys)
    
    # We don't even need to solve it for long, just check it compiles and initiates
    prob = ODEProblem(sys, [], (0.0, 10.0), jac=true, sparse=true)
    sol = solve(prob, Rosenbrock23())
    
    @test sol.retcode == ReturnCode.Success
    @test all(!isnan, sol[sys.AB.cap.v])
    @test all(!isnan, sol[sys.LP.cap.v])
    @test all(!isnan, sol[sys.PY.cap.v])
end
