using Optimization
using OptimizationOptimJL
using ForwardDiff
using Plots
using DifferentiationInterface
using Random
using SciMLStructures: Tunable, canonicalize, replace
using PreallocationTools
using DataInterpolations

# Generate synthetic data from true function
function true_exp_func(x, a=2.0, b=0.5)
    return a * exp(b * x)
end

# Create noisy observations
x_data = collect(0.0:0.05:5.0)
true_a = 2.0
true_b = 0.5
y_data_clean = true_exp_func.(x_data, true_a, true_b)

Random.seed!(42)
noise_level = 0.3
y_data_noisy = y_data_clean .+ noise_level .* randn(length(y_data_clean))
itp_y = LinearInterpolation(y_data_noisy, x_data)

# Parameters
END_TIME = x_data[end]
GROUP_SIZE = 8
OVERLAP = 2
PRINT_SUMMARY = true

# Tracking losses
intra_shoot_losses, continuity_losses, observer_losses = Float64[], Float64[], Float64[]

function split_shoots(timesteps, group_size=nothing; overlap=0)
    N = length(timesteps)
    nshoots(gs) = cld(N - overlap, gs - overlap)
    if isnothing(group_size)
        group_size = 10
        if nshoots(group_size) > 100
            group_size = overlap + cld(N - overlap, 100)
        end
    end
    @assert overlap * 4 <= group_size
    stride = group_size - overlap
    intermediate = [s:min(s + group_size - 1, N) for s in 1:stride:N]
    if length(intermediate[end]) < group_size && length(intermediate) > 1
        intermediate[end-1] = first(intermediate[end-1]):last(intermediate[end])
        pop!(intermediate)
    end
    return intermediate, nshoots(group_size)
end

function split_solution(sol, timesteps, group_size=nothing; overlap=0)
    shoots, nshoots = split_shoots(timesteps, group_size, overlap=overlap)
    return ([(t=sol.t[r], u=sol.u[r], u0=sol.u[first(r)]) for r in shoots], shoots)
end

function shoot_endpoint_diffs(shoot_ids, Vs, shoots, y_data_clean)
    Dict(i => ForwardDiff.value.(Vs[i][end] .- y_data_clean[shoots[i]][end])
         for i in shoot_ids)
end

# Shooting loss function with multiple states
function exp_shoot_loss(x, p)
    x_data, y_data_noisy, setter, diffcache, group_size, overlap, n_params, observer_weight = p
    n_states = 1  # Just tracking y value
    
    # SPLIT shoots
    shoots, nshoots = split_shoots(x_data, group_size; overlap=overlap)
    
    # Extract parameters and states
    ps = x[1:n_params]  # [a, b]
    a, b = ps[1], ps[2]
    u0s = [x[n_params+(i-1)*n_states+1:n_params+i*n_states] for i in 1:length(shoots)]
    
    # LOOP SHOOTS
    intra_loss = 0.0
    cont_loss = 0.0
    obs_loss = 0.0
    Ys = []
    
    for (i, r) in enumerate(shoots)
        xs = x_data[r]
        # Solve: y = a * exp(b * x) with initial condition
        y_pred = a .* exp.(b .* xs)
        
        # If we have initial condition from previous shoot, blend it
        if i > 1
            y_pred[1] = u0s[i][1]
        end
        
        push!(Ys, y_pred)
        
        # Data fitting loss
        y_actual = y_data_noisy[r]
        intra_loss += sum(abs2, y_pred .- y_actual)
        
        # Observer/regularization loss (penalize deviation from true model)
        observer_error = sum(abs2, a .* exp.(b .* xs) .- y_pred) / length(xs)
        obs_loss += observer_error
        
        # Continuity loss with next shoot
        if i < length(shoots)
            cont_loss += sum(abs2, y_pred[end] .- u0s[i+1])
        end
    end
    
    # Weighted total loss
    rho = 0.5
    adj_intra_loss = intra_loss / length(x_data)
    adj_cont_loss = (rho * cont_loss) / nshoots
    adj_obs_loss = (observer_weight * obs_loss) / length(x_data)
    
    push!(intra_shoot_losses, adj_intra_loss)
    push!(continuity_losses, adj_cont_loss)
    push!(observer_losses, adj_obs_loss)
    
    tot_loss = 0.5 * adj_intra_loss + adj_cont_loss + adj_obs_loss
    
    if length(intra_shoot_losses) % 10 == 0
        println("Iter $(length(intra_shoot_losses)): a=$(round(a; digits=3)), b=$(round(b; digits=3))")
        println("  Intra: $(round(adj_intra_loss; digits=4)), Cont: $(round(adj_cont_loss; digits=4)), Obs: $(round(adj_obs_loss; digits=4))")
        println("  Diffs: $(shoot_endpoint_diffs([1, min(2, length(shoots))], Ys, shoots, y_data_noisy))")
    end
    
    return tot_loss
end

# Initial guess
guess_a = 0.5
guess_b = 0.2
toOptimize = [guess_a, guess_b]

# Setup optimization
x0 = vcat(toOptimize, fill(y_data_noisy[1], GROUP_SIZE))  # params + initial conditions per shoot
n_params = length(toOptimize)
diffcache = DiffCache(copy(x0))

shoots, nshoots = split_shoots(x_data, GROUP_SIZE, overlap=OVERLAP)
println("Splits: $nshoots shoots of size $GROUP_SIZE with overlap $OVERLAP")

opt_params = (x_data, y_data_noisy, nothing, diffcache, GROUP_SIZE, OVERLAP, n_params, 0.1)
adtype = AutoForwardDiff()

lb = vcat([0.1, 0.01], fill(-Inf, length(x0) - n_params))
ub = vcat([5.0, 2.0], fill(Inf, length(x0) - n_params))

optfn = OptimizationFunction(exp_shoot_loss, adtype)
optprob = OptimizationProblem(optfn, x0, opt_params, lb=lb, ub=ub)

println("Starting exponential function fitting with shooting method...")
res = solve(optprob, Fminbox(BFGS()); maxiters=100, show_every=1, reltol=1e-6, abstol=1e-6)

if PRINT_SUMMARY
    fitted_a, fitted_b = res.u[1:2]
    println("""
    Final parameter estimates: a=$(round(fitted_a; digits=4)), b=$(round(fitted_b; digits=4))
    True values: a=$true_a, b=$true_b
    Training summary:
    retcode: $(res.retcode)
    iterations: $(res.stats.iterations)
    final intra-shoot loss: $(round(last(intra_shoot_losses); digits=4))
    final continuity loss:  $(round(last(continuity_losses); digits=4))
    final observer loss:    $(round(last(observer_losses); digits=4))
    final objective: $(round(res.objective; digits=4))
    """)
end

# Plotting
fitted_a, fitted_b = res.u[1:2]
y_fitted = fitted_a .* exp.(fitted_b .* x_data)
y_init = guess_a .* exp.(guess_b .* x_data)

p1 = plot(x_data, y_data_noisy, seriestype=:scatter, label="Noisy Data", color=:gray, ms=4, alpha=0.7)
plot!(p1, x_data, y_data_clean, label="True (a=$true_a, b=$true_b)", color=:black, lw=2.5)
plot!(p1, x_data, y_fitted, label="Fitted (a=$(round(fitted_a; digits=3)), b=$(round(fitted_b; digits=3)))", ls=:dash, lw=2.5, color=:red)
plot!(p1, x_data, y_init, label="Init (a=$guess_a, b=$guess_b)", ls=:dot, lw=2, color=:blue)
title!("Exponential Function Fitting with Shooting Method")
xlabel!("x")
ylabel!("y")

p2 = plot(intra_shoot_losses, label="Intra-shoot", lw=2)
plot!(p2, continuity_losses, label="Continuity", lw=2)
plot!(p2, observer_losses, label="Observer", lw=2)
xlabel!("Iteration")
ylabel!("Loss")
title!("Loss Components Over Time")

p = plot(p1, p2, size=(1000, 500), legend=:outertop)
savefig(p, "exp_fit_advanced.png")
display(p)