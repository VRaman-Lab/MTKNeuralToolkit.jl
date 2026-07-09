using MTKNeuralToolkit
using Test
using ExplicitImports
using ModelingToolkit
using Symbolics
using OrdinaryDiffEq

@testset "Explicit Imports" begin
    # @test check_no_implicit_imports(MTKNeuralToolkit; skip=(ModelingToolkit, Symbolics, OrdinaryDiffEq)) === nothing
    @test check_no_stale_explicit_imports(MTKNeuralToolkit) === nothing
    @test check_all_explicit_imports_via_owners(MTKNeuralToolkit) === nothing
end
