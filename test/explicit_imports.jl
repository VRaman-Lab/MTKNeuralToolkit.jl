using MTKNeuralToolkit
using Test
using ExplicitImports

@testset "Explicit Imports" begin
    # Checks that all names used in the package are explicitly imported
    @test check_no_implicit_imports(MTKNeuralToolkit) === nothing
    @test check_no_stale_explicit_imports(MTKNeuralToolkit) === nothing
    @test check_all_explicit_imports_via_owners(MTKNeuralToolkit) === nothing
end
