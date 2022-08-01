matrices = dir('test_subjects/*.mtx');
test_count = size(matrices,1);

for cur_test=1:test_count
  cur_matrix = matrices(cur_test).name;
  [matrix, m, n, numnonzero] = mmread(strcat('test_subjects/',cur_matrix));
  [val, row_ptr, col_ind] = sparse2csr(matrix);

  ofile = fopen(strcat('test_subjects/',cur_matrix,'.crs'), 'w');
  fprintf(ofile, '%d %d %d\n', m, n+1, numnonzero);
  fprintf(ofile, '%s\n', sprintf("%d ", row_ptr));
  fprintf('wrote row_ptr...')
  fprintf(ofile, '%s\n', sprintf("%d ", col_ind));
  fprintf('wrote col_ind...')
  fprintf(ofile, '%s\n', sprintf("%f ", val));
  fprintf('wrote val...')
  fclose(ofile);
end
