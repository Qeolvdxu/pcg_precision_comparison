function [xout, flag, relres, iter, percision, flush_flag] = custom_datatype_pcg(A, b, t, maxint, M, toCache, perci, flush, Aperci)
% A: matrix
% b: right hand side vector
% t: tolerance
% max inter: maximum number of iteractions
% M: preconditioner, approximate of inv(A)
% toCache: set to 1 to cache best results, 0 otherwise
% perci: floating point percision
% flush: flush denormal flag

    % initial setup
    x = zeros(size(A,1), 1);    % zero initial guess
    
    r = b - A*x;     %residual
    
    p = z;
    
    i = 0;

    % loop
    while (i < maxint) && (norm(r)/norm(b) > t)
    
        q = A*p;
        
        v = dot(r,z);
        
        alpha = v / dot(p,q);
        

        x = x + alpha*p;    % improve approximation
        
        r = r - alpha*q;    % update residual
        
        z = MT\(M\r);            % preconditioning
        
        z = z;
        

        beta = dot(r,z) /  v;
        
        p = z + beta*p;     % new search direction
        
        i = i+1;
    end
    return;
end
