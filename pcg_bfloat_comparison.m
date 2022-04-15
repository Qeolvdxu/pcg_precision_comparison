% TEST custom PCG algorithm with IEEE vs BFloat data types

test_count = 20;
sz = [0 6];
varTypes = ["int16","string","int16","double","int16","int16"];
varNames = ["Test Number","Matrix","Size", "Density", "Default Iteration Count", "Bfloat16 Iteration Count"];
temps = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

    fprintf("Tests done:")
for cur_test=1:test_count
    % cur_test is the current tests ID

    n = 300; % size
    d = rand; % density       
                               
    rc = rand * 1e-4; % random reciprocal condition               
    A = sprandsym(n, d, rc, 1); % create a random sparse matrix with positive values
    b = randn(size(A,1), 1);

    % calculate data
    % [xout2, f2, relres2, count2] = custom_pcg(A, b, 1e-7, 1000, eye(n), 1); 

    %A = chop(A,7);
    [xout2, f2, relres2, count2] = custom_pcg(A, b, 1e-7, 1000, eye(n), 1); 

    temps(cur_test,:) = {cur_test,"foobar",n,d,count2,count2};
    fprintf("%3d",cur_test)

end 
temps