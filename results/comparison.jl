using DistributedOptimization, Plots, LightGraphs

function convergence_check()
    # Initialization
    N = 20
    n = 4

    # Create a random problem
    #comm_graph, tess = generate_random_delaunay_graph(N)
    (comm_graph, pts) = generate_disk_graph(N)
    prob = SeperableQuadratic(n, comm_graph)
    println("Seperable Quadratic problem statistics:")
    println(" - max condition number: ", get_max_cond(prob))
    println(" - comm_graph spectral radius: ", 
            maximum(abs.(laplacian_spectrum(comm_graph))))

    # Solve the centralized problem
    xcent = solve_centralized(prob)

    # Generate initial node values
    x_inits = solve_local(prob)

    # Solve with each algorithm and plot values
    error_plt = plot(yaxis=(:log), grid=false, tick_direction=:out)

    @time fvals, cadmm_xhist, chist = cadmm(prob, x_inits; recordx=true)
    cadmm_errors = norm2_error(cadmm_xhist, xcent)
    plot_bounded_errors!(error_plt, cadmm_errors, :blue, "C-ADMM"; chist=chist)

    @time fvals, ex_xhist = extra(prob, x_inits; recordx=true)
    ex_errors = norm2_error(ex_xhist, xcent)
    plot_bounded_errors!(error_plt, ex_errors, :red, "Extra")
#
#    @time fvals, dig_xhist = dig(prob, x_inits; recordx=true)
#    dig_errors = norm2_error(dig_xhist, xcent)
#    plot_bounded_errors!(error_plt, dig_errors, :purple, "DIG")

#    ps_errors = norm2_error(ps_xhist, xcent)
#    plot_bounded_errors!(error_plt, ps_errors, :orange, "Push Sum GD")
#
#    @time fvals, dda_xhist = dda(prob, x_inits; recordx=true)
#    dda_errors = norm2_error(dda_xhist, xcent)
#    plot_bounded_errors!(error_plt, dda_errors, :green, "DDA")

    title!(error_plt, "Error to centralized")
    xaxis!(error_plt, "Communication")
    yaxis!(error_plt, "Norm diff to centralized")

    display(error_plt)
end
