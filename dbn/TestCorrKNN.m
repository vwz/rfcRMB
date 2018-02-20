function [errdistB] = TestCorrKNN()

% RBM + Max-margin Semi-supervised Regression
% device A full, device B unlabeled data

load rmbmsrdata.mat; %mean
addpath '../dbn/';

K = 1;

% compute correlation 
numcases = size(NX1trn,1);
numcases_aux = size(NX2tst,1);

P2pre = [];

    for i=1:numcases_aux
	rho = corr(NX2tst(i,:)', NX1trn', 'type', 'Pearson');
	[sortvals, sortind] = sort(rho, 'descend');
	total = 0;
	for j=1:K
	    total = total + sortvals(j);
	end
	pred = zeros(1,2);
	for j=1:K
	    pred = pred + P1trn(sortind(j),:) * sortvals(j) / total;
	end
	P2pre = [P2pre; pred];
    end

temp = sum((P2pre-P2tst).^2,2);
errdistB = mean(sqrt(temp));

end
