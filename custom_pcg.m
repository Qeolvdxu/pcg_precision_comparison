function [xout, flag, relres, iter] = mypcg(A, b, t, maxint, M, toCache)
    % A: matrix
    % b: right hand side vector
    % t: tolerance
    % max inter: maximum number of iteractions
    % M: preconditioner, approximate of inv(A)
    % toCache: set to 1 to cache best results, 0 otherwise

    % initial setup
    x = zeros(size(A,1), 1);    % zero initial guess
    r = b - A*x;     %residual
    z = M * r;       % preconditioning
    p = z;
    i = 0;

    % best results
    best_x = x;
    best_rel_res = norm(r)/norm(b);
    best_it = i;

    % loop
    while (i < maxint) && (norm(r)/norm(b) > t)
        q = A*p;
        v = dot(r,z);
        alpha = v / dot(p,q);
        x = x + alpha*p;    % improve approximation
        r = r - alpha*q;    % update residual
        z = M*r;            % preconditioning
        beta = dot(r,z) /  v;
        p = z + beta*p;     % new search direction
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

