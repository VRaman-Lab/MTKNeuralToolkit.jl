using Optimization
using OptimizationOptimJL
using ForwardDiff
using Plots
using DifferentiationInterface
using Random
using SciMLStructures: Tunable, canonicalize, replace
using PreallocationTools

# Generate synthetic data from true function
function true_exp_func(x, a=2.0)
    return a * exp(x)
end

# Create noisy observations
x_data = collect(0.0:0.1:3.0)
true_a = 2.0
y_data_clean = true_exp_func.(x_data, true_a)

Random.seed!(42)
noise_level = 0.5
y_data_noisy = y_data_clean .+ noise_level .* randn(length(y_data_clean))

# Shooting loss function
function exp_shoot_loss(params, p)
    x_data, y_data, group_size = p
    a = params[1]
    
    # Split data into shoots
    N = length(x_data)
    stride = group_size
    shoots = [i:min(i + group_size - 1, N) for i in 1:stride:N]
    if length(shoots) > 1 && length(shoots[end]) < group_size
        shoots[end-1] = first(shoots[end-1]):last(shoots[end])
        pop!(shoots)
    end
    
    total_loss = 0.0
    
    for (i, r) in enumerate(shoots)
        x_shoot = x_data[r]
        y_shoot = y_data[r]
        y_pred = a .* exp.(x_shoot)
        
        # Intra-shoot loss
        intra_loss = sum(abs2, y_pred .- y_shoot)
        total_loss += intra_loss
        
        # Continuity loss (ensure smooth connection between shoots)
        if i < length(shoots)
            y_end = a * exp(x_shoot[end])
            y_next_start = a * exp(x_data[first(shoots[i+1])])
            total_loss += 0.1 * abs2(y_end - y_next_start)
        end
    end
    
    return total_loss / length(x_data)
end

# Initial guess and optimization
guess_a = 0.5
x0 = [guess_a]
group_size = 5

opt_params = (x_data, y_data_noisy, group_size)
adtype = AutoForwardDiff()

optfn = OptimizationFunction(exp_shoot_loss, adtype)
optprob = OptimizationProblem(optfn, x0, opt_params, lb=[0.1], ub=[5.0])

println("Starting exponential function fitting...")
res = solve(optprob, Fminbox(BFGS()); maxiters=100, show_every=10)

fitted_a = res.u[1]
println("True a: $true_a, Fitted a: $fitted_a")
println("Final loss: $(res.objective)")

# Plot results
y_fitted = fitted_a .* exp.(x_data)
y_init = guess_a .* exp.(x_data)

p = plot(x_data, y_data_noisy, seriestype=:scatter, label="Noisy Data", color=:gray, ms=5)
plot!(p, x_data, y_data_clean, label="True Function (a=$true_a)", color=:black, lw=2)
plot!(p, x_data, y_fitted, label="Fitted Function (a=$(round(fitted_a; digits=3)))", ls=:dash, lw=2, color=:red)
plot!(p, x_data, y_init, label="Initial Guess (a=$guess_a)", ls=:dot, lw=2, color=:blue)
title!("Exponential Function Fitting with Shooting Method")
xlabel!("x")
ylabel!("y = a*exp(x)")
savefig(p, "exp_fit_result.png")
display(p)