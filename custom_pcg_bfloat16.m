function [xout, flag, relres, iter] = custom_pcg_bfloat16(A, b, t, maxint, M, toCache)
    % A: matrix
    % b: right hand side vector
    % t: tolerance
    % max inter: maximum number of iteractions
    % M: preconditioner, approximate of inv(A)
    % toCache: set to 1 to cache best results, 0 otherwise

    chop(A);
    
    % initial setup
    x = zeros(size(A,1), 1);    % zero initial guess
    r = chop(b - A*x,7);     %residual
    z = chop(M * r,7);       % preconditioning
    p = z;
    i = 0;

    % best results
    best_x = x;
    best_rel_res = chop(norm(r)/norm(b),7);
    best_it = i;

    % loop
    while (i < maxint) && (norm(r)/norm(b) > t)
        q = chop(A*p,7);
        v = chop(dot(r,z),7);
        alpha = chop(v / chop(dot(p,q),7),7);
        x = chop(x + alpha*p,7);    % improve approximation
        r = chop(r - alpha*q,7);    % update residual
        z = chop(M*r,7);            % preconditioning
        beta = chop(chop(dot(r,z),7) /  v,7);
        p = chop(z + chop(beta*p,7),7);     % new search direction
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