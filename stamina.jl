module SMoN

using Graphs, Distributions, Symbolics, POMDPTools, NLsolve

using PlotlyJS
import GraphPlot  # for spring_layout

struct Ensemble
    n::Integer # Number of vertices in the graph.
    constraints::Vector{Function} # f_i(G)=c in form [g_i] with g_i = f_i - c.
end

"""Constrain a 2-vertex digraph to be acyclic."""
function f₁(A)
    # Constraint is that a_12 * a_21 = 0.
    return A[1, 2] * A[2, 1]
end

"""Constrain a 2-vertex digraph to not have an arc 1 -> 2."""
function f₂(A)
    # Constraint is that a_12 = 0.
    return A[1, 2]
end

"""Constrain a 2-vertex digraph to have an arc 2 -> 1."""
function f₃(A)
    # Constraint is that a_21 = 1.
    return A[2, 1] - 1
end

"""Return the <fᵢ> as a symbolic vector function of the λᵢ."""
function conjugates(Z)
    # Obtain the variables from the expression for the partition function.
    λs = Symbolics.get_variables(Z)
    # Make vector taking appropriate derivatives, implicitly return it.
    [-Symbolics.derivative(Z, λ) / Z for λ in λs]
end

function conjugates_to_nlsolve(conjugates, expectations=nothing)
    if !isnothing(expectations)
        conjugates -= expectations
    end
    # Obtain the variables from the first conjugate expression.
    λs = Symbolics.get_variables(conjugates[1])
    # Make a function builder that can be mapped over a vector of expressions.
    builder = ex -> Symbolics.build_function(ex, λs..., expression=Val{false})

    # Produce vector of functions from the conjugate expressions.
    f_functions = map(builder, conjugates)
    # Prepare function computing residuals in-place, the way NLsolve likes it.
    function f!(F, λs)
        for i ∈ 1:length(λs)
            F[i] = f_functions[i](λs...)
        end
    end

    # Compute symbolic Jacobian from symbolic conjugates.
    jacobian = Symbolics.jacobian(conjugates, λs)
    # Produce matrix of functions from the Jacobian expressions.
    j_functions = map(builder, jacobian)
    # Prepare function computing Jacobian in-place, the way NLsolve likes it.
    function j!(J, λs)
        for i ∈ 1:length(λs), j ∈ 1:length(λs)
            J[i, j] = j_functions[i, j](λs...)
        end
    end

    # Return the f! and j! functions.
    return f!, j!
end

"""Solve for and return values of the lambdas, given a partition function."""
function lambdas( Z, expectations=nothing
                , initial_values=nothing, method=:newton, ftol=1e-999 )
    # Obtain the variables from the expression for the partition function.
    λs = Symbolics.get_variables(Z)

    # Initialise the initial values if not provided.
    nλs = length(λs)
    if isnothing(initial_values)
        initial_values = rand(nλs) # .1 * ones(nλs)
    end

    # Do the actual solving for the lambdas using NLsolve.
    f!, j! = conjugates_to_nlsolve(conjugates(Z), expectations)
    # solution = nlsolve(f!, j!, initial_values, method=method, ftol=ftol)
    solution = nlsolve(f!, initial_values, method=:trust_region, ftol=ftol)
    # solution = nlsolve(f!, initial_values, method=:anderson, ftol=ftol)

    # Do as promised.
    return solution.zero
end

"""Solve the entire problem. Return the Z, H(G), p(G) and λs."""
function solve(n, constraints, expectations=nothing)
    # Build the partition function.
    Z = partitionsum(n, constraints)

    # Obtain the variables from the expression for the partition function.
    λs_symbolic = Symbolics.get_variables(Z) # Symbolic λs.
    # Solve for the λs.
    λs_numeric = lambdas(Z, expectations) # Numerical values of each λᵢ.
    # Make a dictionary mapping the λs to their numerical values.
    nλs = length(λs_symbolic) # Number of λs.
    λs_sym_num = Dict(λs_symbolic[i] => λs_numeric[i] for i in 1:nλs)

    # Plug λs into Z to obtain numerical value for the partition function.
    Z = Symbolics.value(substitute(Z, λs_sym_num))

    # Plug λ into H to obtain numerical value for the Hamiltonian.
    H = Symbolics.value(hamiltonian(constraints, λs_numeric))

    # Create the probability distribution function.
    p(G) = exp(-H(G)) / Z

    # Wrap the probability distribution in an expected value functional...
    E = expectation(p, n)

    # ... and a variance functional.
    var(f) = E(G->f(G)^2) - E(G->f(G))^2

    # Return Z, H(G), p(G), E, var and the dictionary of λs.
    return Z, H, p, E, var, λs_sym_num
end

"""Produce a Distribution object from a discrete probability distribution."""
function distribution(p, n)
    ngraphs = 2^(n*(n-1))
    values = []
    probabilities = []
    for i in 1:ngraphs
        G = graph(i, n)
        push!(values, G)
        push!(probabilities, p(G))
    end
    return values, probabilities # SparseCat(values, probabilities)
end

"""Produce an expection functional based on a distribution."""
function expectation(d)
    function E(f)
        e = 0
        for (i, A) in enumerate(d.vals)
            e += f(A) * d.probs[i]
        end
        return e
    end
    return E
end

"""Produce an expectation functional based on a probability distribution."""
function expectation(p, n)
    ngraphs = BigInt(2)^(n*(n-1))
    function E(f)
        e = 0
        for i in 1:ngraphs
            G = graph(i, n)
            e += p(G) * f(G)
        end
        return e
    end
end

function partitionsum(n, constraints::Vector)
    Z = 0
    ngraphs = BigInt(2)^(n*(n-1))
    H = hamiltonian(constraints)
    for i ∈ 1:ngraphs
        G = graph(i, n)
        Z += exp(-H(G))
    end
    return Z
end

function hamiltonian(constraints)
    λs = []
    for i in 1:length(constraints)
        λ = Symbol("λ", gosub(i))
        λs   = [@variables $λ; λs]
    end

    function H(G)
        h = 0
        for i in 1:length(constraints)
            h += λs[i] * constraints[i](G)
        end
        return h
    end

    return H
end

function hamiltonian(constraints, lambdas)
    function H(G)
        h = 0
        for i in 1:length(constraints)
            if !isnan(lambdas[i] * constraints[i](G))
                h += lambdas[i] * constraints[i](G)
            end
        end
        return h
    end

    return H
end


function gosub(n::Integer)::String
    String(map(d->Char(d+8320), reverse(digits(n))))
end

"""Produce the adjacency matrix of the i-th graph on n vertices (1-based)."""
function graph(i::Integer, n::Integer)
    nentries = n*(n-1)
    bitstring = digits(i-1, base=2, pad=nentries)
    for j ∈ 1:n
        k = (j-1)*n + j # Zeroes on diagonal, thus k = 0n+1, 1n+2, 2n+3, etc.
        insert!(bitstring, k, 0)
    end
    A = reshape(bitstring, (n, n))
    return A
end

"""Using graphplot with pleasant defaults."""
gplot(G) = graphplot( G # Graph or adjacency matrix.
                    , nodeshape= :circle
                    , markersize = .05
                    , markercolour = :darkgrey
                    , linecolour = :darkgrey
                    , linealpha = .5
                    , edgewidth = (s, d, w) -> 5 * A[s, d]
                    )

"""Adapted from Julia Plotly documentation."""
function ambigplot(A)
    # Generate a random layout
    G = Graphs.DiGraph(A)
    # G =  Graphs.euclidean_graph(200, 2, cutoff=0.125)[1]
    # Position nodes
    pos_x, pos_y = GraphPlot.spring_layout(G)

    # Create plot points
    edge_x = []
    edge_y = []

    for edge in edges(G)
        push!(edge_x, pos_x[src(edge)])
        push!(edge_x, pos_x[dst(edge)])
        push!(edge_y, pos_y[src(edge)])
        push!(edge_y, pos_y[dst(edge)])
    end

    #  Color node points by the number of connections.
    # color_map = [size(neighbors(G, node))[1] for edge in 1:200]
    color_map = [size(neighbors(G, node))[1] for node in Graphs.nv(G)]

    # Create edges
    edges_trace = scatter(
        mode="lines",
        x=edge_x,
        y=edge_y,
        line=attr(
            width=0.5,
            color="#888"
        ),
    )

    # Create nodes
    nodes_trace = scatter(
        x=pos_x,
        y=pos_y,
        mode="markers",
        text = [string("# of connections: ", connection) for connection in color_map],
        marker=attr(
            showscale=true,
            colorscale=colors.imola,
            color=color_map,
            size=10,
            colorbar=attr(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right"
        )
        )
    )

    # Create Plot
    plot(
        [edges_trace, nodes_trace],
        Layout(
            hovermode="closest",
            title="Network Graph made with Julia",
            titlefont_size=16,
            showlegend=false,
            showarrow=true,
            xaxis=attr(showgrid=false, zeroline=false, showticklabels=false),
            yaxis=attr(showgrid=false, zeroline=false, showticklabels=false)
        )
    )
end

# End of module.
end
