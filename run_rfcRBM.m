function [acc1, acc2] = run_rfcRBM() 

load rmbmsrdata.mat;
addpath './dbn/';
addpath(genpath('/home/vincentz/workspace/localization/working/lapsvmp_v02/'));

% Get location index-to-coordinate mapping
index = unique(Y1tst,'rows'); % find unique location index
nloc = length(index);
location = zeros(nloc,2);
for i=1:nloc
    j = index(i);
    tmp = find(Y1tst == j);
    location(j,:) = P1tst(tmp(1),:);
end

% data
% batchdata = B1trn, batchlabel = C1trn, batchdata_aux = B2trn

%-----------------------------------------------------------------------

[numcases numdim numbatches] = size(C1trn);
D1trn = zeros(numcases, numbatches);
for batch=1:numbatches
    [value, index] = ismember(C1trn(:,:,batch), location, 'rows');
    D1trn(:, batch) = index;
end

[acc1 acc2] = rfcRBM(B1trn, B2trn, D1trn, NX1tst, NX2tst, Y1tst, Y2tst, location);

end 
