@testset "parameters" begin

	@testset "Postive" begin
   		test_parameter_interface(Positive(5.0))
   		test_parameter_interface(Positive(5.0))
   	end

   	@testset "Fixed" begin
   		test_parameter_interface(Fixed((a=5.0, b=4.0)))
   	end

	function objective_function(unflatten, flat_θ::Vector{<:Real})
		θ = unflatten(flat_θ)
		return abs2(value(θ.a)) + abs2(value(θ.b))
	end

   	# This is more of a worked example. Will be properly split up / tidied up.
   	@testset "Integration" begin

		θ0 = (a=5.0, b=4.0)
		flat_parameters, unflatten = flatten(θ0)

		results = Optim.optimize(
		    θ -> objective_function(unflatten, θ),
		    θ->only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
		    flat_parameters,
		    BFGS(),
		    Optim.Options();
		    inplace=false,
		)

		# Check that it's successfully optimised.
		@test objective_function(unflatten, results.minimizer) < 1e-12
   	end

   	@testset "Other Integration" begin

		θ0 = (a=5.0, b=Fixed(4.0))
		flat_parameters, unflatten = flatten(θ0)

		results = Optim.optimize(
		    θ -> objective_function(unflatten, θ),
		    θ->only(Zygote.gradient(θ -> objective_function(unflatten, θ), θ)),
		    flat_parameters,
		    BFGS(),
		    Optim.Options();
		    inplace=false,
		)

		# Check that it's successfully optimised.
		@test value(unflatten(results.minimizer).b) == 4.0
   	end
end
