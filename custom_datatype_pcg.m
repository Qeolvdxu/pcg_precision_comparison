function [xout, flag, relres, iter, percision, flush_flag] = custom_datatype_pcg(A, b, t, maxint, M, toCache, perci, flush, Aperci)
% A: matrix
% b: right hand side vector
% t: tolerance
% max inter: maximum number of iteractions
% M: preconditioner, approximate of inv(A)
% toCache: set to 1 to cache best results, 0 otherwise
% perci: floating point percision
% flush: flush denormal flag

    options.format = Aperci; % bfloat16 percision 
    options.subnormal = flush; %denormal flushing enabled

    A = chop(A,options);

    options.format = perci; % bfloat16 percision 
    options.subnormal = flush; %denormal flushing enabled
    
    % initial setup
    x = zeros(size(A,1), 1);    % zero initial guess
    r = chop(b - A*x,options);     %residual
    z = chop(M * r,options);       % preconditioning
    p = z;
    i = 0;

    % best results
    best_x = x;
    best_rel_res = chop(norm(r)/norm(b),options);
    best_it = i;

    % loop
    while (i < maxint) && (norm(r)/norm(b) > t)
        q = chop(A*p,options);
        v = chop(dot(r,z),options);
        alpha = chop(v / chop(dot(p,q),options),options);

        x = chop(x + alpha*p,options);    % improve approximation
        r = chop(r - alpha*q,options);    % update residual
        z = chop(M*r,options);            % preconditioning

        beta = chop(chop(dot(r,z),options) /  v,options);
        p = chop(z + chop(beta*p,options),options);     % new search direction
        i = i+1;

        % cache if this approximation is the best so far
        if (toCache)
            if (norm(r)/norm(b) < best_rel_res)
                best_x = x;
                best_rel_res = norm(r)/norm(b);
                best_it = i;
            end
        end
    end

    % assign output
    if (toCache)
        xout = best_x;
        relres = best_rel_res;
        iter = best_it;
    else
        xout = x;
        relres = norm(r)/norm(b);
        iter = i;
    end
    

    % set convergence flag
    if (i == maxint)
        flag = 1;
    else
        flag = 0;
    end

    return;
end
