% TEST custom PCG algorithm with IEEE vs BFloat data types
test_count = 20;
sz = [0 6];
varTypes = ["int16","string","int16","double","int16","int16"];
varNames = ["Test Number","Matrix","Size", "Density", "Default Iteration Count", "Bfloat16 Iteration Count"];
temps = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

% Sparse Matrix files from Suitesparse Collection
matrices = {'arc130.mtx','494_bus.mtx','662_bus.mtx','685_bus.mtx','1138_bus.mtx'...
            ,'bcsstk01.mtx','bcsstk02.mtx','bcsstk03.mtx','bcsstk04.mtx','bcsstk05.mtx'...
            ,'bcsstk06.mtx','bcsstk07.mtx','bcsstk08.mtx','bcsstk09.mtx','bcsstk10.mtx'...
            ,'bcsstk11.mtx','bcsstk12.mtx','bcsstk13.mtx','bcsstk14.mtx','bcsstk15.mtx'};

% counters
it_total1 = 1;
it_total2 = 1;
nonzero_its = 0;

fprintf("Tests done:")
for cur_test=1:test_count

    % Read matrix files
    cur_matrix = matrices{cur_test};%'arc130.mtx';
    [A, ~, A_size, nonzero_count] = mmread(cur_matrix);

    % calculate iteration count using standard floats
    [~, ~, ~, itcount1] = custom_pcg(A, b, 1e-7, 10000, eye(A_size), 1);

     % calculate iteration count using brain floats
    A = chop(A,7);
    [~, ~, ~, itcount2] = custom_pcg(A, b, 1e-7, 10000, eye(A_size), 1);

    % add data to table, show progress, get mean
    temps(cur_test,:) = {cur_test,cur_matrix,A_size,A_size/nonzero_count,itcount1,itcount2};
    fprintf("%3d",cur_test)
    if(itcount1 ~= 0 && itcount2 ~= 0)
        it_total1 = it_total1 * itcount1;
        it_total2 = it_total2 * itcount2;
        nonzero_its = nonzero_its + 1;
    end
 
end 

% add mean entry to table
temps(test_count+1,:) = {0,"Geometric Mean",0,0,it_total1^(1/nonzero_its),it_total2^(1/nonzero_its)};
temps