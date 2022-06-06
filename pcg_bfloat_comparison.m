% test custom PCG algorithm with IEEE vs BFloat data types
pkg load tablicious

% customizable settings 
max_iters = 20000;
tol = 1e-6;

% find mtx sparse matrix files
matrices = dir('test_subjects/*.mtx');
test_count = size(matrices,1);

% table data vars
table_cell = cell(test_count,6);
name_cell = cell(9,1);
name_cell{1,1}="Test Number";
name_cell{2,1}="Matrix";
name_cell{3,1}="Size";
name_cell{4,1}="Density";
name_cell{5,1}="Condition Number";
name_cell{6,1}="IEEE 32 IC";
name_cell{7,1}="Bfloat16 IC";
name_cell{8,1}="RCM IEEE 32 IC";
name_cell{9,1}="RCM Bfloat16 IC";

iteration_cell = cell(4,1);
for i=1:4
    iteration_cell{i,1} = 0;
end

for cur_test=1:test_count

    % Read matrix files
    cur_matrix = matrices(cur_test).name;
    [A, ~, A_size, nonzero_count] = mmread(strcat('test_subjects/',cur_matrix));

    x = randn(size(A,2), 1) * max(A);
    b = A * x;
    %b = randn(size(A,1), 1);
    %b = ones(size(A,1),1);

    % start print output
    fprintf("%3d: %s, %3d, %3d, %3d, ", ...
            cur_test,cur_matrix,A_size,A_size/nonzero_count,cond(A))

    %Standard Ordering
    % calculate iteration count using standard floats
    [~, ~, ~, iteration_cell{1,1}] = custom_datatype_pcg(A, b, tol, max_iters, eye(A_size), 0, 'fp64', 0);
    fprintf("%3d, ",iteration_cell{1,1})
    
    % calculate iteration count using brain floats
    [~, ~, ~, iteration_cell{2,1}] = custom_datatype_pcg(A, b, tol, max_iters, eye(A_size), 0, 'bfloat16', 0);
    fprintf("%3d, ",iteration_cell{2,1})

    perm = symrcm(A);
    A = A(perm,perm);

    %RMC Ordering
    % calculate iteration count using standard floats
    [~, ~, ~, iteration_cell{3,1}] = custom_datatype_pcg(A, b, tol, max_iters, eye(A_size), 0, 'fp64', 0);
    fprintf("%3d, ",iteration_cell{3,1})

    % calculate iteration count using brain floats
    [~, ~, ~, iteration_cell{4,1}] = custom_datatype_pcg(A, b, tol, max_iters, eye(A_size), 0, 'bfloat16', 0);
    fprintf("%3d\n",iteration_cell{4,1})

    % add data to table, show progress, get mean
    table_cell{cur_test,1} = cur_test;
    table_cell{cur_test,2} = cur_matrix;
    table_cell{cur_test,3} = A_size;
    table_cell{cur_test,4} = A_size/nonzero_count;
    table_cell{cur_test,5} = cond(A);
    table_cell{cur_test,6} = iteration_cell{1,1};
    table_cell{cur_test,7} = iteration_cell{2,1};
    table_cell{cur_test,8} = iteration_cell{3,1};
    table_cell{cur_test,9} = iteration_cell{4,1};
end

% Create and print table
        table = cell2table(table_cell,'VariableNames',name_cell);
        fprintf("\n")
        prettyprint(table)


        % Write to CVS File
        fid = fopen( 'results.csv', 'wt' );
        for i = 1:size(name_cell,1) fprintf(fid,"%s,",name_cell{i}) end
        fprintf(fid," CRLF\n");

        for i = 1:test_count
            for j = 1:size(name_cell,1)
                if j != 2
                    fprintf(fid,"%d,",table_cell{i,j});
                    end
                if j == 2
	            fprintf(fid,"%s,",table_cell{i,j});
                    end
            end
            fprintf(fid," CRLF\n");
        end
        fprintf("# table written to results.csv/n")
