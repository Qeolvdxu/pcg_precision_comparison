function [values] = ans_compare(v1, v2)
    values = cell(3,10);
    vs = {v1, v2, v1-v2};
    for i=1:3
        v = vs{i}
        values{i,1} =max(v); % maximum element
        values{i,2} =       min(v); % maximum element
        values{i,3} =    mse(v); % mean((A-B).^2)
        values{i,4} =      sse(v); % sum((A-B).^2)
        values{i,5} =      mae(v); % mean(abs(A-B))
        values{i,6} =     sae(v); % sum(abs(A-B))
        values{i,7} =      norm(v,1); % sum(abs(A-B))
        values{i,8} =     norm(v,INF); % max(abs(A-B))
        values{i,9} =    norm(v); % sqrt(sse(A-B))
        values{i,10} =    norm(v,"fro"); % frobenius norm
    end
end