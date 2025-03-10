"""
Robot dance single location

Implements an automatic control framework to design efficient mitigation strategies
for Covid-19 based on the control of a SEIR model.

Copyright: Paulo J. S. Silva <pjssilva@unicamp.br>, 2021.
"""

using JuMP
using Ipopt
using Printf
using LinearAlgebra
using SparseArrays
using Distributions
import Statistics.mean
using CSV
using DataFrames
include("simple_arts.jl")

"""
    struct SEIR_Parameters

Parameters to define a SEIR model:

- tinc: incubation time (5.2 for Covid-19).
- tinf: time of infection (2.9 for Covid-19).
- rep: The natural reproduction rate of the disease, for examploe for Covid-19
  you might want to use 2.5 or a similar value.
- ndays: simulation duration.
- ncities: number of (interconnected) cities in the model.
- s1, e1, i1, r1: start proportion of the population in each SEIR class.
- alternate: the weight to give to alternation in the solution.
- availICU: vector with the ratio of avaible ICU.
- time_icu: mean stay in ICU
- rho_icu_ts: information on the time series to estimate the ratio of ICU needed
    for each infected.
- window: time window to keep rt constant.
- out: vector that represents the proportion of the population that leave each city during
    the day.
- M: Matrix that has at position (i, j) how much population goes from city i to city j, the
    proportion with respect to the population of the destiny (j). It should have 0 on the
    diagonal.
- Mt: Matrix that has at position (i, j) how much population goes from city j to city i, the
    proportion with respect to the population of the origin (j). It should have 0 on the
    diagonal.
"""
struct SEIR_Parameters
    # Basic epidemiological constants that define the SEIR model
    tinc::Float64
    tinf::Float64
    rep::Float64
    ndays::Int64
    ncities::Int64
    s1::Vector{Float64}
    e1::Vector{Float64}
    i1::Vector{Float64}
    r1::Vector{Float64}
    alternate::Float64
    availICU::Vector{Float64}
    time_icu::Int64
    rho_icu_ts::Matrix{Float64}
    window::Int64
    out::Vector{Float64}
    M::SparseMatrixCSC{Float64,Int64}
    Mt::SparseMatrixCSC{Float64,Int64}
    vstates::Int64
    r_atten::Vector{Float64}
    # TODO: Should this be population dependent?
    icu_atten::Vector{Float64}
    effect_window::Vector{Int64}
    doses_min_window::Vector{Int64}
    doses_max_window::Vector{Int64}
    max_doses::Vector{Float64}
    npops::Int64                         # Number of subpopulations
    subpop::Vector{Float64}              # Size of each subpopulation in (0, 1), sum = 1
    r0pop::Vector{Float64}               # Factor to adjust R0 for each subpopulation
    icupop::Vector{Float64}              # Factor to adjust the need for ICU of each subpopulation
    contact::Matrix{Float64}             # Contact matrix

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate,
        availICU, rho_icu_ts, window, out, M, Mt)

    SEIR parameters with mobility information (out, M, Mt) and multiple populations.
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
        time_icu, rho_icu_ts, window, out, M, Mt, subpop, r0pop, icupop, contact, 
        r_atten, icu_atten, max_doses, doses_min_window, doses_max_window)
        ls1 = length(s1)
        @assert length(e1) == ls1
        @assert length(i1) == ls1
        @assert length(r1) == ls1
        @assert length(availICU) == ls1
        @assert time_icu > 0.0
        @assert size(rho_icu_ts) == (ls1, 10)
        @assert size(out) == (ls1,)
        @assert all(out .>= 0.0)
        @assert size(M) == (ls1, ls1)
        @assert all(M .>= 0.0)
        @assert size(Mt) == (ls1, ls1)
        @assert all(Mt .>= 0.0)

        # Check subpopulation information
        @assert isapprox(sum(subpop), 1.0)
        npops = length(subpop)
        @assert length(r0pop) == npops
        @assert length(icupop) == npops
        @assert size(contact) == (npops, npops)
        @assert isapprox(sum(contact, dims=2), ones(npops))

        # For now vaccine action is hard coded
        vstates = length(r_atten)
        effect_window = [14, 14]

        new(tinc, tinf, rep, ndays, ls1, s1, e1, i1, r1, alternate, availICU, time_icu,
            rho_icu_ts, window, out, M, Mt,            
            vstates, r_atten, icu_atten, 
            effect_window, doses_min_window, doses_max_window, max_doses,
            npops, subpop, r0pop, icupop, contact)
    end

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate,
        availICU, rho_icu_ts, window, out, M, Mt)

    SEIR parameters with mobility information (out, M, Mt).
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
        time_icu, rho_icu_ts, window, out, M, Mt, r0pop, r_atten, icu_atten, max_doses,
        doses_min_window, doses_max_window)

        # Default subpopulation distribution is hard coded for now.
        subpop = [0.30, 0.48, 0.14, 0.08]
        icupop = [0.06, 0.58, 2.06, 5.16]
        contact = [
            0.57 0.27 0.10 0.06;
            0.20 0.59 0.15 0.06;
            0.15 0.46 0.27 0.12;
            0.18 0.24 0.18 0.40
        ]

        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
            time_icu, rho_icu_ts, window, out, M, Mt, subpop, r0pop, icupop, contact,
            r_atten, icu_atten, max_doses, doses_min_window, doses_max_window)
    end

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate,
        availICU, rho_icu_ts, window)

    SEIR parameters without mobility information, which is assumed to be 0.
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
        time_icu, rho_icu_ts, window)
        ls1 = length(s1)
        out = zeros(ls1)
        M = spzeros(ls1, ls1)
        Mt = spzeros(ls1, ls1)
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
            time_icu, rho_icu_ts, window, out, M, Mt, [1.0], [1.0], [1.0], Float64[],
            Float64[], Float64[])
    end

    """
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU)

    SEIR parameters with unit time window and without mobility information, which is assumed 
    to be 0.
    """
    function SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU,
        time_icu, rho_icu_ts)
        SEIR_Parameters(tinc, tinf, rep, ndays, s1, e1, i1, r1, alternate, availICU, 
            time_icu, rho_icu_ts, 1)
    end
end


"""
    mapind(d, prm)

    Allows for natural use of rt while computing the right index mapping in time d.
"""
function mapind(d, prm)
    return prm.window*div(d - 1, prm.window) + 1
end


"""
    expand(rt, prm)

Expand rt to a full prm.ndays vector.
"""
function expand(rt, prm)
    full_rt = zeros(prm.ncities, prm.ndays)
    for c in 1:prm.ncities, d in 1:prm.ndays
        full_rt[c, d] = rt[mapind(d, prm)]
    end
    return full_rt
end


"""
    best_linear_solver

    Helper function to check what is the best linear solver available for Ipopt, 
        ma97 (preferred) or mumps.
"""
function best_linear_solver()
    PREFERRED = "ma97"
    #PREFERRED = "pardiso"
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
              "print_level" => 0, "linear_solver" => PREFERRED))
    @variable(m, x)
    @objective(m, Min, x^2)
    optimize!(m)
    if termination_status(m) != MOI.INVALID_OPTION
        return PREFERRED
    else
        return "mumps"
    end
end


"""
    seir_model_with_free_initial_value(prm)

Build an optimization model with the SEIR discretization as constraints. The inicial
parameters are not initialized and remain free. This can be useful, for example, to fit the
initial parameters to observed data.
"""
function seir_model_with_free_initial_values(prm, verbosity=0)
    # Create the optimization model.
    # I am reverting to mumps because I can not limit ma97 to use
    # only the actual cores in my machine and mumps seems to be doing 
    # fine.
    if verbosity >= 1
        println("Initializing optimization model...")
    end

    verbosity_ipopt = 0
    if verbosity >= 1
        verbosity_ipopt = 5 # Print summary and progress
    end

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => verbosity_ipopt, "linear_solver" => best_linear_solver()))
    if verbosity >= 1
        println("Initializing optimization model... Ok!")
    end
    # For simplicity I am assuming that one step per day is OK.
    dt = 1.0

    # Note that I do not fix the initial state. It should be defined elsewhere.
    # State variables
    if verbosity >= 1
        println("Adding variables to the model...")
    end
    @variable(m, 0.0 <= s[1:prm.npops, 1:prm.vstates, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= e[1:prm.npops, 1:prm.vstates, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= i[1:prm.npops, 1:prm.vstates, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= r[1:prm.npops, 1:prm.vstates, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= ei[1:prm.npops, 1:prm.vstates - 1, 1:prm.ndays] <= 1.0)
    @variable(m, 0.0 <= ir[1:prm.npops, 1:prm.vstates - 1, 1:prm.ndays] <= 1.0)

    # Control variables
    @variable(m, 0.0 <= rt[1:prm.window:prm.ndays] <= prm.rep)
    @variable(m, 0.0 <= v[1:prm.npops, 1:prm.vstates - 1, 1:prm.ndays] <= 1.0)

    # Extra variables to better separate linear and nonlinear expressions and
    # to decouple and "sparsify" the matrices.
    # Obs. I tried many variations, only adding the variable below worded the best.
    #      I tried to get rid of all SEIR variables and use only the initial conditions.
    #      Add variables for sp, ep, ip, rp. Add a variable to represent s times i.
    @variable(m, rti[p=1:prm.npops, t=1:prm.ndays])
    @variable(m, vs[1:prm.npops, 1:prm.vstates - 1, t=1:prm.ndays])
    if verbosity >= 1
        println("Adding variables to the model... Ok!")
    end

    # Expressions that define "sub-states"
    if verbosity >= 1
        println("Defining additional expressions...")
    end

    # TODO: I think that this is the key place to add the contact matrix
    @expression(m, pop_i[p=1:prm.npops, t=1:prm.ndays],
        (sum(i[p, d, t] for d=1:prm.vstates) + sum(ir[p, d, t] for d=1:prm.vstates - 1))/prm.subpop[p]
    )
    @expression(m, prob_i[p=1:prm.npops, t=1:prm.ndays],
        sum(prm.contact[p, pl]*pop_i[pl, t] for pl=1:prm.npops)
    )
    @constraint(m, [p=1:prm.npops, t=1:prm.ndays],
        rti[p, t] == prm.r0pop[p]*rt[mapind(t, prm)]*prob_i[p, t]
    )
    @constraint(m, [p=1:prm.npops, d=1:prm.vstates - 1, t=1:prm.ndays],
        vs[p, d, t] == v[p, d, t]*s[p, d, t]
    )
    
    # Compute the gradients at time t of the SEIR model.

    # Estimates the infection rate of the susceptible people from city c
    # that went to the other cities k.
    if verbosity >= 1
        println("Defining SEIR equations...")
    end

    ds = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates, prm.ndays)
    for p=1:prm.npops, d=1:prm.vstates, t=1:prm.ndays
        if d == 1
            ds[p, d, t] = @expression(m, 
                -prm.r_atten[d]*rti[p, t]*(s[p, d, t] - vs[p, d, t])/prm.tinf - vs[p, d, t]
            )
        elseif 1 < d && d < prm.vstates
            ds[p, d, t] = @expression(m, 
                -prm.r_atten[d]*rti[p, t]*(s[p, d, t] - vs[p, d, t])/prm.tinf - vs[p, d, t] 
                + vs[p, d - 1, t]
            )
        else
            ds[p, d, t] = @expression(m, 
                -prm.r_atten[d]*rti[p, t]*s[p, d, t]/prm.tinf + vs[p, d - 1, t]
            )
        end
    end
    de = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates, prm.ndays)
    for p=1:prm.npops, d=1:prm.vstates, t=1:prm.ndays
        if d < prm.vstates
            de[p, d, t] = @expression(m, 
                prm.r_atten[d]*rti[p, t]*(s[p, d, t] - vs[p, d, t])/prm.tinf 
                - (1.0  - v[p, d, t])/prm.tinc*e[p, d, t] - v[p, d, t]*e[p, d, t]
            )
        else
            de[p, d, t] = @expression(m, 
                prm.r_atten[d]*rti[p, t]*s[p, d, t]/prm.tinf - e[p, d, t]/prm.tinc
            )
        end
    end
    di = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates, prm.ndays)
    for p=1:prm.npops, d=1:prm.vstates, t=1:prm.ndays
        if d == 1
            di[p, d, t] = @expression(m, 
                (1.0  - v[p, d, t])/prm.tinc*e[p, d, t] 
                - (1.0 - v[p, d, t])/prm.tinf*i[p, d, t] - v[p, d, t]*i[p, d, t]
            )
        elseif 1 < d && d < prm.vstates
            di[p, d, t] = @expression(m, 
                (1.0  - v[p, d, t])/prm.tinc*e[p, d, t] + 2/prm.tinc*ei[p, d - 1, t]
                - (1.0 - v[p, d, t])/prm.tinf*i[p, d, t] - v[p, d, t]*i[p, d, t]
            )
        else
            di[p, d, t] = @expression(m, 
                e[p, d, t]/prm.tinc + 2/prm.tinc*ei[p, d - 1, t] - i[p, d, t]/prm.tinf
            )
        end
    end
    dr = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates, prm.ndays)
    for p=1:prm.npops, d=1:prm.vstates, t=1:prm.ndays
        if d == 1
            dr[p, d, t] = @expression(m,
                (1 - v[p, d, t])/prm.tinf*i[p, d, t] - v[p, d, t]*r[p, d, t]
            )
        elseif 1 < d && d < prm.vstates
            dr[p, d, t] = @expression(m,
                (1 - v[p, d, t])/prm.tinf*i[p, d, t] + 2/prm.tinf*ir[p, d - 1, t] 
                + v[p, d - 1, t]*r[p, d - 1, t] - v[p, d, t]*r[p, d, t]
            )
        else
            dr[p, d, t] = @expression(m,
                i[p, d, t]/prm.tinf + 2/prm.tinf*ir[p, d - 1, t] + v[p, d - 1, t]*r[p, d - 1, t]
            )
        end
    end
    dei = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates - 1, prm.ndays)
    dir = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates - 1, prm.ndays)
    for p=1:prm.npops, d=1:prm.vstates - 1, t=1:prm.ndays
        dei[p, d, t] = @expression(m, v[p, d, t]*e[p, d, t] - 2/prm.tinc*ei[p, d, t])
        dir[p, d, t] = @expression(m, v[p, d, t]*i[p, d, t] - 2/prm.tinf*ir[p, d, t])
    end
    
    if verbosity >= 1
        println("Defining SEIR equations... Ok!")
    end

    discr_method = "finite_difference"
    k_curr_t = 0.5
    k_prev_t = 0.5

    # Discretize SEIR equations
    if discr_method == "finite_difference"
        if verbosity >= 1
            if k_curr_t == 1.0 && k_prev_t == 0.0
                println("Discretizing SEIR equations (backward)...")
            elseif k_curr_t == 0.0 && k_prev_t == 1.0
                println("Discretizing SEIR equations (forward)...")
            elseif k_curr_t == 0.5 && k_prev_t == 0.5
                println("Discretizing SEIR equations (central)...")
            end
        end

        @constraint(m, [p=1:prm.npops, d=1:prm.vstates, t=2:prm.ndays],
            s[p, d, t] == s[p, d, t - 1] + (k_prev_t*ds[p, d, t - 1] + k_curr_t*ds[p, d, t])*dt
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates, t=2:prm.ndays],
            e[p, d, t] == e[p, d, t - 1] + (k_prev_t*de[p, d, t - 1] + k_curr_t*de[p, d, t])*dt
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates, t=2:prm.ndays],
            i[p, d, t] == i[p, d, t - 1] + (k_prev_t*di[p, d, t - 1] + k_curr_t*di[p, d, t])*dt
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates, t=2:prm.ndays],
            r[p, d, t] == r[p, d, t - 1] + (k_prev_t*dr[p, d, t - 1] + k_curr_t*dr[p, d, t])*dt
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays],
            ei[p, d, t] == ei[p, d, t - 1] + (k_prev_t*dei[p, d, t - 1] + k_curr_t*dei[p, d, t])*dt
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays],
            ir[p, d, t] == ir[p, d, t - 1] + (k_prev_t*dir[p, d, t - 1] + k_curr_t*dir[p, d, t])*dt
        )
        if verbosity >= 1
            println("Discretizing SEIR equations... Ok!")
        end
    else
        throw("Invalid discretization method")
    end

    return m
end


"""
    seir_model(prm)

Creates a SEIR model setting the initial parameters for the SEIR variables from prm.
For now it splits the origina S, E, I, R proportionally to prm.subpop.
"""
function seir_model(prm, verbosity)
    m = seir_model_with_free_initial_values(prm, verbosity)

    # Initial state
    s1, e1, i1, r1 = m[:s][:, :, 1], m[:e][:, :, 1], m[:i][:, :, 1], m[:r][:, :, 1]
    ei1, ir1 = m[:ei][:, :, 1], m[:ir][:, :, 1]
    for p = 1:prm.npops
        fix(s1[p, 1], prm.subpop[p]*prm.s1[1]; force=true)
        fix(e1[p, 1], prm.subpop[p]*prm.e1[1]; force=true)
        fix(i1[p, 1], prm.subpop[p]*prm.i1[1]; force=true)
        fix(r1[p, 1], prm.subpop[p]*prm.r1[1]; force=true)
    end
    for p=1:prm.npops, d = 2:prm.vstates
        fix(s1[p, d], 0.0; force=true)
        fix(e1[p, d], 0.0; force=true)
        fix(i1[p, d], 0.0; force=true)
        fix(r1[p, d], 0.0; force=true)
        fix(ei1[p, d - 1], 0.0; force=true)
        fix(ir1[p, d - 1], 0.0; force=true)
    end
    return m
end


"""
    fixed_rt_model(prm)

Creates a model from prm setting all initial parameters and defining the RT as the natural
R0 set in prm.
"""
function fixed_rt_model(prm)
    m = seir_model(prm)
    rt = m[:rt]

    # Fix all rts
    for t = 1:prm.window:prm.ndays
        fix(rt[t], prm.rep; force=true)
    end
    return m
end


"""
    window_control_multcities

Built a simple control problem that tries to force the infected to remain below target every
day for every city using daily controls but only allow them to change in the start of the
time windows.

# Arguments

- prm: SEIR parameters with initial state and other informations.
- population: population of each city.
- target: limit of infected to impose at each city for each day.
- window: number of days for the time window.
- force_difference: allow to turn off the alternation for a city in certain days. Should be
    used if alternation happens even after the eipdemy is dieing off.
- hammer_durarion: Duration in days of a intial hammer phase.
- hammer: Rt level that should be achieved during hammer phase.
- min_rt: minimum rt achievable outside the hammer phases.
"""
function window_control_multcities(prm, population, target, force_difference, 
    hammer_duration=0, hammer=0.89, min_rt=1.0, verbosity=0)
    @assert sum(mod.(hammer_duration, prm.window)) == 0
    pools = [[1]]

    # TODO: These should be parameters - first try
    times_series = [Simple_ARTS(prm.rho_icu_ts[c, :]...) for c in 1:prm.ncities]

    m = seir_model(prm, verbosity)

    if verbosity >= 1
        println("Setting limits for rt...")
    end
    # Fix rt during hammer phase
    rt = m[:rt]
    for c = 1:prm.ncities, d = 1:prm.window:hammer_duration[c]
        fix(rt[d], hammer[c]; force=true)
    end

    rt_profile_data = "profile_data.csv"
    has_rt_profile = isfile(rt_profile_data)
    if has_rt_profile
        println("********************** Using an rt profile")
        df = data = CSV.read(rt_profile_data, DataFrame)
        target_rt = (df[df.Variable .== "rt", 4:end])[1, :]
        total_target = 0
        for t = hammer_duration[1] + 1:prm.window:prm.ndays
            total_target += target_rt[t]
            if target_rt[t] < 0.99*prm.rep
                set_lower_bound(rt[t], min_rt)
            else
                fix(rt[t], target_rt[t]; force=true)
            end
        end
        @constraint(m,
             sum(rt[t] for t=hammer_duration[1] + 1:prm.window:prm.ndays) >= total_target 
        )
    else
        # Set the minimum rt achievable after the hammer phase.
        for c = 1:prm.ncities, d = hammer_duration[c] + 1:prm.window:prm.ndays
            set_lower_bound(rt[d], min_rt)
        end
    end
    if verbosity >= 1
        println("Setting limits for rt... Ok!")
    end

    # Bound the maximal infection rate taking into account the maximal ICU rooms available
    # using a chance contraint.
    if verbosity >= 1
        println("Setting limits for number of infected...")
    end

    # We implement two variants one based on max I and another on sum on
    # entering in R, the max I is simples and gives good results so we
    # are keeping it in the code.
    i = m[:i]
    firstday = hammer_duration .+ 1

    println(pools)
    n_pools = length(pools)
    first_pool_day = [minimum(firstday[pool]) for pool in pools]    
    pool_population = [sum(population[pool]) for pool in pools]
    s, e, r, v, ir = m[:s], m[:e], m[:r], m[:v], m[:ir]
    @variable(m, V[pool_id=1:n_pools, p=1:prm.npops, d=1:prm.ndays] >= 0)
    for pool_id in 1:n_pools
        pool = pools[pool_id]
        # Uses the time series that gives the ratio of IC needed from the first
        # city in the pool.
        ρ_icu = times_series[pool[1]]
        reset(ρ_icu)

        # As in the paper, V represents the number of people that will leave
        # infected and potentially go to ICU.
        # I simplified this assuming that there is a single region (and it is)
        # Whole pool.
        @constraint(m, [p=1:prm.npops, t=1:prm.ndays],
            V[pool_id, p, t] == prm.icupop[p]*prm.time_icu/prm.tinf*(
                sum(prm.icu_atten[d]*(1.0 - v[p, d, t])*i[p, d, t] for d=1:prm.vstates - 1) 
                + prm.icu_atten[prm.vstates]*i[p, prm.vstates, t] 
                + sum(2.0*prm.icu_atten[d]*ir[p, d, t] for d=1:prm.vstates - 1)
            )
        )

        for t in 1:prm.ndays - prm.time_icu
            Eicu, safety_level = iterate(ρ_icu)
            if t >= first_pool_day[pool_id]
                @constraint(m,
                    (Eicu + safety_level)*sum(V[pool_id, p, t] for p=1:prm.npops) <= 
                    sum(target[c, t]*population[c]*prm.availICU[c] for c in pool) / pool_population[pool_id]
                )
            end
        end
    end

    if verbosity >= 1
        println("Setting limits for number of infected... Ok!")
    end

    if verbosity >= 1
        println("Setting constraints to define vaccines")
    end
    # There are no transitions before effect window
    for p=1:prm.npops, d=1:prm.vstates - 1
        for t = 1:prm.effect_window[d]
            fix(v[p, d, t], 0.0; force=true)
        end
    end

    # Define the number of doses applied each day
    if sum(prm.max_doses) == 0
        for p=1:prm.npops, d=1:prm.vstates - 1, t=1:prm.ndays
            fix(v[p, d, t], 0; force=true)
        end
    else
        applied = Array{GenericQuadExpr{Float64,VariableRef}, 3}(undef, prm.npops, prm.vstates - 1, prm.ndays)
        for p=1:prm.npops, d=1:prm.vstates - 1
            for t=1:(prm.ndays - prm.effect_window[d])
                effect = t + prm.effect_window[d]
                applied[p, d, t] = @expression(m,
                    v[p, d, effect]*(s[p, d, effect] + e[p, d, effect] + i[p, d, effect] + r[p, d, effect])
                )
            end
            for t=prm.ndays - prm.effect_window[d] + 1:prm.ndays
                applied[p, d, t] = @expression(m, 0.0*v[p, d, prm.ndays])
            end
        end

        # Apply the doses in order
        @variable(m, 0 <= cumv[p=1:prm.npops, 1:prm.vstates - 1, 1:prm.ndays])
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates - 1], 
            cumv[p, d, 1] == applied[p, d, 1]
        )
        @constraint(m, [p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays],
            cumv[p, d, t] == cumv[p, d, t - 1] + applied[p, d, t]
        )
        for p=1:prm.npops, d=2:prm.vstates - 1, t = 1:prm.ndays
            if t > prm.doses_min_window[d - 1]
                @constraint(m,
                    cumv[p, d, t] <= cumv[p, d - 1, t - prm.doses_min_window[d - 1]]
                )
            else
                fix(cumv[p, d, t], 0.0; force=true)
            end
            if t > prm.doses_max_window[d - 1]
                @constraint(m,
                    cumv[p, d, t] >= cumv[p, d - 1, t - prm.doses_max_window[d - 1]]
                )
            end
        end

        # Respect the maximum daily ammount of vaccine doses
        for t = 1:prm.ndays
            @constraint(m,
                sum(applied[p, d, t] for p=1:prm.npops for d=1:prm.vstates - 1) <= prm.max_doses[t]
            )
        end

        # # Give all the doses to all that received the first dose
        # @constraint(m, [p=1:prm.npops, d=2:prm.vstates - 1],
        #     cumv[p, d, prm.ndays] >= cumv[p, d - 1, prm.ndays]
        # )

        # Give all the doses to 95% of the population by the first time it is 
        # possible
        cum_md = cumsum(prm.max_doses)
        min_vacc = 0.95
        if minimum(prm.doses_max_window - prm.doses_min_window) < 7
            delta = 28
        else
            delta = 14
        end
        target_day = argmax(cum_md .>= min_vacc*(prm.vstates - 1)) + delta
        if target_day <= prm.ndays
            @constraint(m, 
                sum(applied[p, prm.vstates - 1, t] 
                    for p=1:prm.npops for t=1:target_day) 
                >= min_vacc
            )
        end
        println("Target date to $(100 * min_vacc)% cover = ", target_day)
    end

    if verbosity >= 1
        println("Setting constraints to define vaccines... Ok!")
    end

    # Used to compute the total rt variation
    # @variable(m, ttv1[p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays])
    # @variable(m, ttv2[p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays] >= 0)
    # @constraint(m, con_ttv[p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays], 
    #     ttv1[p, d, t] >= v[p, d, t - 1] - v[p, d, t]
    # )
    # @constraint(m, con_ttv2[p=1:prm.npops, d=1:prm.vstates - 1, t=2:prm.ndays], 
    #     ttv2[p, d, t] >= v[p, d, t] - v[p, d, t - 1]
    # )

    # Compute the weights for the objectives terms
    if verbosity >= 1
        println("Computing objective function...")
    end
    effect_pop = population # You may try to use other metrics like sqrt.(population)
    mean_population = mean(effect_pop)
    dif_matrix = Matrix{Float64}(undef, prm.ncities, prm.ndays)
    for c = 1:prm.ncities, d = 1:prm.ndays
        dif_matrix[c, d] = force_difference[c, d] / mean_population / (2*prm.ncities)
    end
    # Define objective
    if has_rt_profile
        # Minimize the number of UTI needed - I am assuming that Eicu is constant,
        # that is, there is no trend in the time series. If that is not the case
        # Eicu should multiply V below
        @objective(m, Min,
            sum(V[1, p, t] for p=1:prm.npops for t=1:prm.ndays)
            + 0.0e-1*sum((v[p, d, t] - v[p, d, t - 1])^2 
                for p=1:prm.npops for d=1:prm.vstates - 1 for t=2:prm.ndays
            )
        )
    else
        @objective(m, Min,
            # Try to keep as many people working as possible
            prm.window*sum(effect_pop[c]/mean_population*(prm.rep - rt[d])
                for c = 1:prm.ncities for d = hammer_duration[c]+1:prm.window:prm.ndays)
            # Maximize S
            #-sum(s[d, prm.ndays] for d =1:prm.vstates)
            # Paga para aplicar vacinas
            # + sum(applied)
            + sum((v[p, d, t] - v[p, d, t - 1])^2 
                for p=1:prm.npops for d=1:prm.vstates - 1 for t=2:prm.ndays
            )
            # + 1.0e-4*(sum(ttv1) + sum(ttv2))
            # + prm.window*sum(effect_pop[c]/mean_population*(prm.rep - rt[d])^2
            #     for c = 1:prm.ncities for d = hammer_duration[c]+1:prm.window:prm.ndays)
        )
    end
    if verbosity >= 1
        println("Computing objective function... Ok!")
    end

    return m
end


"""
    add_ramp

Adds ramp constraints to the robot-dance model.

# Input arguments
m: the optimization model
prm: struct with parameters
hammer_duration: initial hammer for each city
delta_rt_max: max increase in the control rt between (t-1) and (t)

# Output: the optimization model m
"""
function add_ramp(m, prm, hammer_duration, delta_rt_max, verbosity=0)
    if verbosity >= 1
        println("Adding ramp constraints (delta_rt_max = $delta_rt_max)...")
    end
    rt = m[:rt]
    @constraint(m, [c=1:prm.ncities, d = hammer_duration[c] + 1:prm.window:prm.ndays],
    rt[d] - rt[d - prm.window] <= delta_rt_max
    )
    if verbosity >= 1
        println("Adding ramp constraints (delta_rt_max = $delta_rt_max)... Ok!")
    end

    return m
end


"""
    simulate_control

Simulate the SEIR mdel with a given control checking whether the target is achieved.

TODO: fix this, hammer_duration does not even exist.

"""
function simulate_control(prm, population, control, target)
    @assert sum(mod.(hammer_duration[c], prm.window)) == 0

    m = seir_model(prm)

    rt = m[:rt]
    for c=1:prm.ncities, d=1:prm.ndays
        fix(rt[c, mapind(d, prm)], control[c, d]; force=true)
    end

    # Constraint on maximum level of infection
    i = m[:i]
    @constraint(m, [c=1:prm.ncities, t=2:prm.ndays], i[c, t] <= target)

    # Compute the weights for the objectives terms
    effect_pop = sqrt.(population)
    mean_population = mean(effect_pop)
    dif_matrix = Matrix{Float64}(undef, prm.ncities, prm.ndays)
    for c = 1:prm.ncities, d = 1:prm.ndays
        dif_matrix[c, d] = force_difference[c, d] / mean_population / (2*prm.ncities)
    end
    # Define objective
    @objective(m, Min,
        # Try to keep as many people working as possible
        sum(prm.window*effect_pop[c]/mean_population*(prm.rep - rt[c, d])
            for c = 1:prm.ncities for d = +1:prm.window:prm.ndays) -
        # Try to enforce different cities to alternate the controls
        100.0/(prm.rep^2)*sum(
            minimum((effect_pop[c], effect_pop[cl]))*
            minimum((dif_matrix[c, d], dif_matrix[cl, d]))*
            (rt[c, d] - rt[cl, d]^2)
            for c = 1:prm.ncities 
            for cl = c + 1:prm.ncities 
            for d = hammer_duration[c] + 1:prm.window:prm.ndays
            )
    )

    return m
end
