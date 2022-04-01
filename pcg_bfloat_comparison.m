% TEST custom PCG algorithm with IEEE vs BFloat data types

test_count = 100;

for cur_test=1:test_count
    % cur_test is the current tests ID

    n = 300; % size
    d = rand; % density       
                               
    rc = rand * 1e-4; % random reciprocal condition               
    A = sprandsym(n, d, rc, 1); % create a random sparse matrix with positive values
    b = randn(size(A,1), 1);

    [xout2, f2, relres2, count2] = custom_pcg(A, b, 1e-7, 1000, eye(n), 1);

    fprintf("-----\nTest ID: %3d \t Size: %4d \t Density: %.4f \t \n IEEE\t-\tConvergance: %1d \t Interaction count: %3d\n BFLOAT\t-\tConvergance: \t Interaction count:\n",...
    cur_test, n, d, f2, count2);
end