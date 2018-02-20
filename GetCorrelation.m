function [batchcor] = GetCorrelation(batchdata, batchdata_aux, K)
% This program first calculates the Pearson correlation between an unlabeled instance from batchdata_aux and an labeled instance from batchdata,
% then it finds the top K labled neighbors for each unlabeled instance.
%
% The program assumes that the following variables are set externally:
% batchdata  -- the labeled data that is divided into batches (numcases numdims numbatches)
% batchdata_aux   -- the auxiliary unlabeled data that is divided into batches (numcases numdims numbatches)
% K -- the top K neighbors in the correlation graph

[numcases numdims numbatches]=size(batchdata);
[numcases_aux numdims_aux numbatches_aux]=size(batchdata_aux);
batchcor = zeros(numcases_aux, K*3, numbatches_aux);

stackdata = [];
for batch=1:numbatches
    stackdata = [stackdata; batchdata(:,:,batch)];
end

for batch=1:numbatches_aux
    data = batchdata_aux(:,:,batch);
    for i=1:numcases
	rho = corr(data(i,:)', stackdata', 'type', 'Pearson');
	[sortvals, sortind] = sort(rho, 'descend');
	corvec = [];
	for j=1:K
	    [batchId, insId] = GetIds(numcases, numbatches, sortind(j));
	    tmp = [batchId, insId, sortvals(j)];
	    corvec = [corvec, tmp];
	end
	batchcor(i,:,batch) = corvec;
    end
end

end

function [batchId, insId] = GetIds(numcases, numbatches, ind) 

batchId = ceil(ind/numcases);
insId = ind - (batchId-1) * numcases;

end
