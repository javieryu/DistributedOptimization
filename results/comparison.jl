using DistOpt

function convergence_check()
    # Initialization
    N = 20
    n = 4

    PS_RESULTS = false
    CADMM_RESULTS = true
    DDA_RESULTS = false
    EXTRA_RESULTS = true
    DIG_RESULTS = true
    SAVE_FIG = true

    # Create a random problem
    #comm_graph, tess = generate_random_delaunay_graph(N)
    (comm_graph, pts) = generate_disk_graph(N)
    prob = SeperableQuadratic(n, comm_graph)
    println("Seperable Quadratic problem statistics:")
    println(" - max condition number: ", get_max_cond(prob))
    println(" - comm_graph spectral radius: ", 
            maximum(abs.(laplacian_spectrum(comm_graph))))

    # Solve the centralized problem
    x_cent = solve_centralized(prob)

    # Generate initial node values
    x_inits = solve_local(prob)

    # Solve with each algorithm and plot values
    error_plt = plot(yaxis=(:log), grid=false, tick_direction=:out)

    if PS_RESULTS
        @time ps_errors, ps_xhist = push_sum(prob, x_inits; x_cent=x_cent)
        plot_bounded_errors!(error_plt, ps_errors, :orange, "Push Sum GD")
    end

    if CADMM_RESULTS
        @time cadmm_errors, cadmm_xhist = cadmm(prob, x_inits; x_cent=x_cent)
        plot_bounded_errors!(error_plt, cadmm_errors, :blue, "C-ADMM")
    end

    if DDA_RESULTS
        @time dda_errors, dda_xhist = dda(prob, x_inits; x_cent=x_cent)
        plot_bounded_errors!(error_plt, dda_errors, :green, "DDA")
    end

    if EXTRA_RESULTS
        @time ex_errors, ex_xhist = extra(prob, x_inits; x_cent=x_cent)
        plot_bounded_errors!(error_plt, ex_errors, :red, "Extra")
    end

    if DIG_RESULTS
        @time dig_errors, dig_xhist = dig(prob, x_inits; x_cent=x_cent)
        plot_bounded_errors!(error_plt, dig_errors, :purple, "DIG")
    end

    title!(error_plt, "Error to centralized")
    xaxis!(error_plt, "Iterations")
    yaxis!(error_plt, "Norm diff to centralized")

    if SAVE_FIG
        png(error_plt, "../images/convergence_comparisions")
    end

    display(error_plt)
end

function xhist_comparison()
    # Initialization
    N = 10
    n = 2
    K = 1500 #Number of iterations

    # Create a random problem
    #comm_graph, tess = generate_random_delaunay_graph(N)
    (comm_graph, pts) = generate_disk_graph(N)
    prob = SeperableQuadratic(n, comm_graph)

    # Solve the centralized problem
    x_cent = solve_centralized(prob)

    # Generate initial node values
    x_inits = solve_local(prob)

    # Solve with different methods
    #error, xhist = push_sum(prob, x_inits; MAX_CYCLES=K, x_cent=x_cent, recordx=true)
    error, xhist  = cadmm(prob, x_inits; MAX_CYCLES=K, x_cent=x_cent, recordx=true)
    #error, xhist = dda(prob, x_inits; MAX_CYCLES=K, x_cent=x_cent, recordx=true)   

    #grad = 2.0 * prob.A[1] * xhist[1][:, 5] + prob.b[1]
    #display(grad)

    #plt = plot(legend=false, xlims=(-0.5, 0.5), ylims=(-0.5, 0.5)) 
    plt = plot(legend=false) 
    plot_xhist!(plt, xhist)
    plot_xcent!(plt, x_cent)
    plot_xinits!(plt, x_inits)

    title!(plt, "X Trajectory")
    xaxis!(plt, "X1")
    yaxis!(plt, "X2")

    #png(plt, "../pics/ps_xhist_comparisions")
    png(plt, "../pics/bad_ps_xhist_comparisions")
    display(plt)
    return
end

function topology_comparision()
    n = 2

    grid_graph = LightGraphs.grid([4, 4])
    disk_graph = lollipop_graph(4, 12)
    barb_graph = barbell_graph(8, 8)

    grid_prob = SeperableQuadratic(n, grid_graph)
    disk_prob = deepcopy(grid_prob)
    barb_prob = deepcopy(grid_prob)

    disk_prob.graph = disk_graph
    barb_prob.graph = barb_graph

    x_cent = solve_centralized(grid_prob)
    x_inits = solve_local(grid_prob)
    grid_error, xh  = cadmm(grid_prob, x_inits; MAX_CYCLES=1000, x_cent=x_cent, recordx=false)
    barb_error, xh  = cadmm(barb_prob, x_inits; MAX_CYCLES=1000, x_cent=x_cent, recordx=false)
    disk_error, xh  = cadmm(disk_prob, x_inits; MAX_CYCLES=1000, x_cent=x_cent, recordx=false)

    error_plt = plot(yaxis=(:log), grid=false, tick_direction=:out)
    plot_mean_error!(error_plt, grid_error, :red, "Grid") 
    plot_mean_error!(error_plt, barb_error, :green, "Barbell") 
    plot_mean_error!(error_plt, disk_error, :blue, "Cycle") 

    title!(error_plt, "Error to centralized")
    xaxis!(error_plt, "Iterations")
    yaxis!(error_plt, "Norm diff to centralized")

    png(error_plt, "../pics/topology_comparision")

    display(error_plt)

    return
end
