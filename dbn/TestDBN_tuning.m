function [errdistA errdistB] = TestDBN_tuning(numhid1)

% device A full, device B none, no MTL

%load dbnbatchdata.mat; %normalized by column
load dbnbatchdata_row.mat; %normalized by row
addpath '../libsvm-3.18/matlab/';
addpath '../dbn/';

maxepoch = 5000;
restart = 1;

weight_sparse_array = [0.1, 1, 10, 100]; 
weight_pair_array = [0.1, 1, 10, 100];

for i=1:length(weight_sparse_array)
  for j=1:length(weight_pair_array)
     	weight_sparse = weight_sparse_array(i);
    	weight_pair = weight_pair_array(j);
 	disp(weight_sparse);
	disp(weight_pair);
     	[vishid1, visbias1, hidbias1, batchposhidprobs1, poshidstates] = rbm_layer1_sparse_tuning(B1trn, numhid1, maxepoch, restart, weight_sparse, weight_pair);

	Z1trn = rbm_layer_feature(NX1trn, vishid1, visbias1, hidbias1);
	Z1tst = rbm_layer_feature(NX1tst, vishid1, visbias1, hidbias1);
	Z2tst = rbm_layer_feature(NX2tst, vishid1, visbias1, hidbias1);

	% for x-coordinate
	Y = P1trn(:,1);
	model = svmtrain(Y, Z1trn, '-s 3 -t 0');

	Y = P1tst(:,1);
	[tstYx, accuracy, dec_values] = svmpredict(Y, Z1tst, model);
	P1pre = tstYx;

	Y = P2tst(:,1);
	[tstYx, accuracy, dec_values] = svmpredict(Y, Z2tst, model);
	P2pre = tstYx;

	% for y-coordinate
	Y = P1trn(:,2);
	model = svmtrain(Y, Z1trn, '-s 3 -t 0');

	Y = P1tst(:,2);
	[tstYx, accuracy, dec_values] = svmpredict(Y, Z1tst, model);
	P1pre = [P1pre tstYx];

	Y = P2tst(:,2);
	[tstYx, accuracy, dec_values] = svmpredict(Y, Z2tst, model);
	P2pre = [P2pre tstYx];

	% calculate error distance
	temp = sum((P1pre-P1tst).^2,2);
	errdistA = mean(sqrt(temp));

	temp = sum((P2pre-P2tst).^2,2);
	errdistB = mean(sqrt(temp));

	disp(errdistA);
	disp(errdistB);
  end
end

end 

function [poshidprobs] = rbm_layer_feature(data, vishid, visbias, hidbias)

numcases = size(data,1);
poshidprobs =  1./(1 + exp(-data*vishid - repmat(hidbias,numcases,1)));

end
