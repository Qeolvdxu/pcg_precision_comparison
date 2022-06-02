pkg load tablicious

% TEST custom PCG algorithm with IEEE vs BFloat data types
test_count = 20;
max_iters = 20000;
tol = 1e-6

sz = [0 6];

table_cell = cell(test_count,6);
name_cell = cell(9,1);
name_cell{1,1}="Test Number";
name_cell{2,1}="Matrix";
name_cell{3,1}="Size";
name_cell{4,1}="Density";
name_cell{5,1}="Condition Number";
name_cell{6,1}="Default IC";
name_cell{7,1}="Bfloat16 IC";
name_cell{8,1}="RCM Default IC";
name_cell{9,1}="RCM Bfloat16 IC";

matrices = dir('test_subjects/*.mtx')

% Sparse Matrix files from Suitesparse Collection
matrices = {'arc130.mtx','494_bus.mtx','662_bus.mtx','685_bus.mtx','1138_bus.mtx'...
            ,'bcsstk01.mtx','bcsstk02.mtx','bcsstk03.mtx','bcsstk04.mtx','bcsstk05.mtx'...
            ,'bcsstk06.mtx','bcsstk07.mtx','bcsstk08.mtx','bcsstk09.mtx','bcsstk10.mtx'...
            ,'bcsstk11.mtx','bcsstk12.mtx','bcsstk13.mtx','bcsstk14.mtx','bcsstk15.mtx'};

% counters
it_total1 = 1;
it_total2 = 1;
it_total3 = 1;
it_total4 = 1;
nonzero_its = 0;

fprintf("Tests done:\n")
for cur_test=1:test_count

    % Read matrix files
    cur_matrix = matrices{cur_test};
    [A, ~, A_size, nonzero_count] = mmread(cur_matrix);
    
    %x = randn(size(A,2), 1) * max(A);
    %b = A * x;
    %b = randn(size(A,1), 1);
    b = ones(size(A,1),1);

    %Standard Ordering
    % calculate iteration count using standard floats
    [~, ~, ~, itcount1] = custom_pcg(A, b, tol, max_iters, eye(A_size), 0);
     % calculate iteration count using brain floats
    [~, ~, ~, itcount2] = custom_pcg_bfloat16(A, b, tol, max_iters, eye(A_size), 0);

    perm = symrcm(A);
    A = A(perm,perm);
    
    %RMC Ordering
    % calculate iteration count using standard floats
    [~, ~, ~, itcount3] = custom_pcg(A, b, tol, max_iters, eye(A_size), 0);
     % calculate iteration count using brain floats
    [~, ~, ~, itcount4] = custom_pcg_bfloat16(A, b, tol, max_iters, eye(A_size), 0);

    % add data to table, show progress, get mean
    table_cell{cur_test,1} = cur_test;
    table_cell{cur_test,2} = cur_matrix;
    table_cell{cur_test,3} = A_size;
    table_cell{cur_test,4} = A_size/nonzero_count;
    table_cell{cur_test,5} = cond(A);
    table_cell{cur_test,6} = itcount1;
    table_cell{cur_test,7} = itcount2;
    table_cell{cur_test,8} = itcount3;
    table_cell{cur_test,9} = itcount4;

    fprintf("%3d: %s, %3d, %3d, %3d, %3d, %3d, %3d, %3d\n",cur_test,cur_matrix,A_size,A_size/nonzero_count,cond(A),itcount1,itcount2,itcount3,itcount4)
end

% Create and print table
table = cell2table(table_cell,'VariableNames',name_cell);
fprintf("\n")
prettyprint(table)


% Write to CVS File
t = table_cell;
fid = fopen( 'results.csv', 'wt' );
for i = 1:size(name_cell,1) fprintf(fid,"%s,",name_cell{i}) end
fprintf(fid," CRLF\n");

for i = 1:test_count
  for j = 1:size(name_cell,1)
    if j != 2
        fprintf(fid,"%d,",table_cell{i,j});
    end
    if j == 2
	fprintf(fid,"%s,",t{i,j});
    end
  end
  fprintf(fid," CLRF\n");
end
fprintf("# table written to results.csv/n")
