% test custom PCG algorithm with various data types

% customizable settings 
max_iters = 8000;
tol = 1e-6;
datatype_count = 5;

% find mtx sparse matrix files
matrices = dir('test_subjects/*.mtx');
test_count = 2;%size(matrices,1);

% table data vars
table_cell = cell(test_count,5+(datatype_count*2));
ans_cell = cell(test_count,datatype_count*2);
name_cell = cell(9,1);

name_cell{1,1}="Test Number";
name_cell{2,1}="Matrix";
name_cell{3,1}="Size";
name_cell{4,1}="Density";
name_cell{5,1}="Condition Number";

name_cell{6,1}="IEEE 64";
name_cell{7,1}="IEEE 32";
name_cell{8,1}="BFloat16";
name_cell{9,1}="64A BFloat16";
name_cell{10,1}="64A IEEE 32 IC";

name_cell{11,1}="RCM IEEE 64";
name_cell{12,1}="RCM IEEE 32";
name_cell{13,1}="RCM BFloat16";
name_cell{14,1}="RCM 64A BFloat16";
name_cell{15,1}="RCM 64A IEEE 32 IC";

iteration_cell = cell(datatype_count*2,1);

for i=1:4
    iteration_cell{i,1} = 0;
end

for cur_test=1:test_count

    % Read matrix files
    cur_matrix = matrices(cur_test).name;
    [A, ~, A_size, nonzero_count] = mmread(strcat('test_subjects/',cur_matrix));

    %x = randn(size(A,2), 1) * max(A);
    %b = A * x;
    %b = randn(size(A,1), 1);
    b = ones(size(A,1),1);

    % start print output
    fprintf("%3d: %s, %3d, %3d, %3d, ", ...
            cur_test,cur_matrix,A_size,A_size/nonzero_count,condest(A))

    alpha = max(sum(abs(A),2)./diag(A))-2;
    precond = ichol(A, struct('type','ict','droptol',1e-3,'diagcomp',alpha));
    %precond = eye(A_size);
    
    %Standard Ordering
    % calculate iteration count using 
    %iteration_cell{1,1} = pcg(A,b,tol,max_iters,precond);
    [ans_cell{cur_test,1}, ~, ~, iteration_cell{1,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp64', 0, 'fp64');
    fprintf("%3d,",iteration_cell{1,1})
    %disp(ans_cell{1,1})
    
    [ans_cell{cur_test,2}, ~, ~, iteration_cell{2,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp32', 0, 'fp32');
    fprintf("%3d, ",iteration_cell{2,1})

    [ans_cell{cur_test,3}, ~, ~, iteration_cell{3,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'bfloat16', 0, 'bfloat16');
    fprintf("%3d, ",iteration_cell{3,1})

    [ans_cell{cur_test,4}, ~, ~, iteration_cell{4,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'bfloat16', 0, 'fp64');
    fprintf("%3d, ",iteration_cell{4,1})

    [ans_cell{cur_test,5}, ~, ~, iteration_cell{5,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp32', 0, 'fp64');
    fprintf("%3d, ",iteration_cell{5,1})



    perm = symrcm(A);
    A = A(perm,perm);


    %RCM Ordering
    [ans_cell{cur_test,6}, ~, ~, iteration_cell{6,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp64', 0, 'fp64');
    fprintf("%3d, ",iteration_cell{6,1})
    
    
    [ans_cell{cur_test,7}, ~, ~, iteration_cell{7,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp32', 0, 'fp32');
    fprintf("%3d, ",iteration_cell{7,1})

    [ans_cell{cur_test,8}, ~, ~, iteration_cell{8,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'bfloat16', 0, 'bfloat16');
    fprintf("%3d, ",iteration_cell{8,1})

    [ans_cell{cur_test,9}, ~, ~, iteration_cell{9,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'bfloat16', 0, 'fp64');
    fprintf("%3d, ",iteration_cell{9,1})

    [ans_cell{cur_test,10}, ~, ~, iteration_cell{10,1}] = custom_datatype_pcg(A, b, tol, max_iters, precond, 0, 'fp32', 0, 'fp64');
    fprintf("%3d \n",iteration_cell{10,1})


    % add data to table, show progress, get mean
    table_cell{cur_test,1} = cur_test;
    table_cell{cur_test,2} = cur_matrix;
    table_cell{cur_test,3} = A_size;
    table_cell{cur_test,4} = A_size/nonzero_count;
    table_cell{cur_test,5} = condest(A);

    for i=1+5:(datatype_count*2)+5
        table_cell{cur_test,i} = iteration_cell{i-5,1};
    end
end

% Write to CVS File
fid = fopen( 'results.csv', 'wt' );

for i=1:size(name_cell,1)
    fprintf(fid,"%s,",name_cell{i})
end

fprintf(fid," \n");
for i = 1:test_count
    for j = 1:size(name_cell,1)
        if j ~= 2
            fprintf(fid,"%d,",table_cell{i,j});
        end
        if j == 2
	    fprintf(fid,"%s,",table_cell{i,j});
        end
    end
    fprintf(fid," \n");
end

fprintf(fid,"\nVECTOR COMPARISON\n");
for i=1:20
    for j=1:datatype_count*2
        for k=1:datatype_count*2
            test = ans_compare(ans_cell{i,j},ans_cell{i,k});
            for i=1:4
                for j=1:10
                    if i == 1
                        fprintf(fid,"%s, ",test{i,j})
                    end
                    if i ~= 1
                        fprintf(fid,"%d, ",test{i,j})
                    end
                end
                fprintf(fid,"\n")

            end
            fprintf(fid,"\n")
        end
    end
end
fprintf("# table written to results.csv/n")
