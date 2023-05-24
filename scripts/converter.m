myDir = '../test_subjects';
myFiles = dir(fullfile(strcat(myDir, '/mm'), '*.mtx'));

for i = 1 : length(myFiles)
  baseFileName = myFiles(i).name;
  fullFileName = fullfile(strcat(myDir, '/mm'), baseFileName);
  icholTOutFileName = fullfile(strcat(myDir, '/precond'), strcat(myFiles(i).name, '.PRECONDITIONER.mtx'));
  [ matrix, m, n, numnonzero ] = mmread(fullFileName);
  fprintf('icholing matrix %s...', baseFileName)
  alpha = max(sum(abs(matrix),2)./diag(matrix))-2;
  precond = ichol(matrix, struct('type','ict','droptol',1e-3,'diagcomp',alpha));
  mmwrite(icholTOutFileName,precond);
  fprintf('Done!\n')
end

myPreconds = dir(fullfile(strcat(myDir, '/precond'), '*.mtx'));

for i = 1:length(myPreconds)
    baseFileName = myPreconds(i).name
    fullFileName = fullfile(strcat(myDir, '/precond'), baseFileName);
    normOutFileName = fullfile(strcat(myDir, '/precond_norm'), strcat(myPreconds(i).name, '.csr'));
    rcmOutFileName =fullfile(strcat(myDir, '/precond_rcm'), strcat(myPreconds(i).name, '.rcm.csr'));

    [ matrix, m, n, numnonzero ] = mmread(fullFileName);
    fprintf('Converting matrix %s...', baseFileName)

    tick = tic;
    perm = symrcm(matrix);
    reordered = matrix(perm, perm);
    tock = toc(tick);
    fprintf('reordered in %f...', tock)

        [val, row_ptr, col_ind] = sparse2csr(matrix);
    fprintf('converted original to csr...')

        commandDeletePrior = [ 'rm -f ', normOutFileName ];
    system(commandDeletePrior);

    ofile = fopen(normOutFileName, 'w');
    fprintf(ofile, '%d %d %d\n', m, n, numnonzero);
    fprintf(ofile, '%s\n', sprintf("%d ", row_ptr));
    fprintf('wrote row_ptr...')
    fprintf(ofile, '%s\n', sprintf("%d ", col_ind));
    fprintf('wrote col_ind...')
    fprintf(ofile, '%s\n', sprintf("%f ", val));
    fprintf('wrote val...')
    fclose(ofile);

    [ val, row_ptr, col_ind ] = sparse2csr(reordered);
    fprintf('converted reordered to csr...')

    commandDeletePrior = [ 'rm -f ', rcmOutFileName ];
    system(commandDeletePrior);

    ofile = fopen(rcmOutFileName, 'w');
    fprintf(ofile, '%d %d %d\n', m, n, numnonzero);
    fprintf(ofile, '%s\n', sprintf("%d ", row_ptr));
    fprintf('wrote row_ptr...')
    fprintf(ofile, '%s\n', sprintf("%d ", col_ind));
    fprintf('wrote col_ind...')
    fprintf(ofile, '%s\n', sprintf("%f ", val));
    fprintf('wrote val...')
    fclose(ofile);

    fprintf('done\n')
end

for i = 1:length(myFiles)
    baseFileName = myFiles(i).name
    fullFileName = fullfile(strcat(myDir, '/mm'), baseFileName);
    normOutFileName = fullfile(strcat(myDir, '/norm'), strcat(myFiles(i).name, '.csr'));
    rcmOutFileName =fullfile(strcat(myDir, '/rcm'), strcat(myFiles(i).name, '.rcm.csr'));

    [ matrix, m, n, numnonzero ] = mmread(fullFileName);
    fprintf('Converting matrix %s...', baseFileName)

    tick = tic;
    perm = symrcm(matrix);
    reordered = matrix(perm, perm);
    tock = toc(tick);
    fprintf('reordered in %f...', tock)

        [val, row_ptr, col_ind] = sparse2csr(matrix);
    fprintf('converted original to csr...')

        commandDeletePrior = [ 'rm -f ', normOutFileName ];
    system(commandDeletePrior);

    ofile = fopen(normOutFileName, 'w');
    fprintf(ofile, '%d %d %d\n', m, n, numnonzero);
    fprintf(ofile, '%s\n', sprintf("%d ", row_ptr));
    fprintf('wrote row_ptr...')
    fprintf(ofile, '%s\n', sprintf("%d ", col_ind));
    fprintf('wrote col_ind...')
    fprintf(ofile, '%s\n', sprintf("%f ", val));
    fprintf('wrote val...')
    fclose(ofile);

    [ val, row_ptr, col_ind ] = sparse2csr(reordered);
    fprintf('converted reordered to csr...')

    commandDeletePrior = [ 'rm -f ', rcmOutFileName ];
    system(commandDeletePrior);

    ofile = fopen(rcmOutFileName, 'w');
    fprintf(ofile, '%d %d %d\n', m, n, numnonzero);
    fprintf(ofile, '%s\n', sprintf("%d ", row_ptr));
    fprintf('wrote row_ptr...')
    fprintf(ofile, '%s\n', sprintf("%d ", col_ind));
    fprintf('wrote col_ind...')
    fprintf(ofile, '%s\n', sprintf("%f ", val));
    fprintf('wrote val...')
    fclose(ofile);

    fprintf('done\n')
end


fprintf('CONVERSION DONE!\n');
