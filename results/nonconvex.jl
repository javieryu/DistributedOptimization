using DistributedOptimization, Plots

function test_nonconvex()
    # Problem parameters
    N = 30 # Number of sensors
    T = 3 # Number of targets
    mvar = 0.001 # Measurement variance

    # Desired outputs
    NEXT_RESULTS = true

    # Generate a random problem formulation
    (comm_graph, locs) = generate_disk_graph(N)
    paper_targs = [0.03 0.86 0.6; 0.85 0.5 0.01]
    prob = SqDistTargetLoc(T, locs, mvar, comm_graph, targ_poses=paper_targs)

    # Generate intial estimates
    x_inits = random_feasible_inits(prob) 
    x_cent = prob.grnd_tru

    # Solve with different methods and visualize results
    # Plot using the PyPlot backend
    pyplot()
    error_plt = plot(yaxis=(:log), grid=false, tick_direction=:out)
    traj_plt = plot(legend=false, reuse=false)
    stat_plt = plot(reuse=false)

    if NEXT_RESULTS
        next_errors, next_xhist = next(prob, x_inits; recordx=true,
                                       MAX_CYCLES=100, x_cent=x_cent,
                                       verbose=true)
        next_stat = compute_stationarity_series(prob, next_xhist)
        plot_bounded_errors!(error_plt, next_errors, :red, "NEXT")
        plot_sq_dis_traj!(traj_plt, prob, next_xhist)
        plot!(stat_plt, next_stat, c=:red, lab="NEXT")
    end

    title!(error_plt, "Error to centralized")
    xaxis!(error_plt, "Iterations")
    yaxis!(error_plt, "Euc. Dist. to centralized")

    title!(traj_plt, "Solution trajectories")
    xaxis!(traj_plt, "x")
    yaxis!(traj_plt, "y")

    title!(stat_plt, "Stationarity")
    xaxis!(stat_plt, "Iterations")
    yaxis!(stat_plt, "J")

    gui(error_plt)
    gui(traj_plt)
    gui(stat_plt)

    return
end
