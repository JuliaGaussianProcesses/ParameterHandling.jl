@testset "parameters_meta.jl" begin
    @testset "fixed" begin
        @testset "plain" begin
            val = (a=5.0, b=4.0)
            p = fixed(val)
            test_parameter_interface(p)
            @test value(p) == val
        end

        @testset "constrained" begin
            val = 1.234
            constrained_val = positive(val)
            p = fixed(constrained_val)
            test_parameter_interface(p)
            @test value(p) ≈ val
        end
    end

    @testset "deferred" begin
        mvnormal(args...) = MvNormal(args...)
        pdiagmat(args...) = PDiagMat(args...)

        test_parameter_interface(deferred(sin, 0.5); check_inferred=tuple_infers)
        test_parameter_interface(deferred(sin, positive(0.5)); check_inferred=tuple_infers)
        test_parameter_interface(
            deferred(
                mvnormal, fixed(randn(5)), deferred(pdiagmat, positive.(rand(5) .+ 1e-1))
            );
            check_inferred=tuple_infers,
        )
    end
end
