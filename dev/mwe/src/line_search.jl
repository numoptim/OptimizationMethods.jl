"""Armijo search: accept f(x-a*g) <= f(x)-rho*a*||g||²; failed trials never replace x."""
function backtracking!(m, p, s)
    stop = get(s, :stop_spec, StoppingCriteria())
    count(s, :obj) >= stop.max_obj_evals && (s[:stop_reason] = :max_obj_evals; return false)
    obj!(p; store=s, x=s[:x])
    s[:obj_x] = s[:obj]
    s[:x_prev] .= s[:x]
    a = m.step_size
    s[:ls_iterations] = 0
    for j in 0:m.max_ls_iterations
        if count(s, :obj) >= stop.max_obj_evals
            s[:stop_reason] = :max_obj_evals
            return false
        end
        s[:trial] .= s[:x_prev] .- a .* s[:grad]
        obj!(p; store=s, x=s[:trial])
        s[:ls_iterations] = j + 1
        if isfinite(s[:obj]) && s[:obj] <= s[:obj_x] - m.rho * a * s[:grad_norm]^2
            s[:x] .= s[:trial]
            s[:obj_x] = s[:obj]; s[:step_size] = a
            return true
        end
        a *= m.delta
    end
    s[:stop_reason] = :line_search_failed
    return false
end
