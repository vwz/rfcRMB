function [Theta, batchposhidprobs, batchposhidprobs_aux] = rfcRBM_rbmtrain(batchdata, batchdata_aux, batchlabel, batchtau, batchtau_aux, params, Theta, F)

%% This program tains regularized RBM of the following form:
%%	min - L1 + lambda1 * R1 + lambda2 * L2
%% where 
%% 	L1 = log P(v),
%%	R1 = |1^T vishid|_F^2
%%	L2 = sum_m L2_m 
%%	L2_m = sum_k max{1 - z_k f(g(x^k)), 0} + gamma_A |f|_A^2 + gamma_I |f|_I^2%%
%% The code is based on G. Hinton's code: rbmhidlinear.m
%
% The program assumes that the following variables are set externally:% batchdata  -- the labeled data that is divided into batches (numcases numdims numbatches)
% batchdata_aux   -- the auxiliary unlabeled data that is divided into batches (numcases numdims numbatches)
% batchlabel -- the labels that are divided into batches (numcases numbatches), and each label is a location class index
% batchtau   -- the adjacency matrix based on batchdata, it is a numbatches*1 cell array, each cell is a sparse matrix on numcases*numcases
% batchtau_aux  -- the adjacency matrix based on batchdata_aux, it is a numbatches*1 cell array, each cell is a sparse matrix on numcases*numcases
% params   -- the parameter struct, defined in the main function% Theta    -- the RBM parameters, where:
%   Theta.vishid     -- the weight w for visible-hidden, it is a d1*d2 matrix, each entry of which denotes the weight w_ij for visible i to hidden j
%   Theta.visbias    -- the weight a for visible, it is a d1*1 vector, each dimension denotes the weight a_i
%   Theta.hidbias    -- the weight b for hidden, it is a d2*1 vector, each dimension denotes the weight b_j
% F         -- the parameters for LapSVM, where
%    F.X_all            -- the training data, it is a model_num*1 cells, each cell is a numcases*numhids matrix
%    F.classifier_all   -- the classifiers for each binary classfication task, it is a model_num*1 cells, each cell includes classifier.alpha and classifier.b
%    F.index_map         -- the mapping function between model number and the class pair, it is a model_num*2 matrix, each row (i,j) indicates that class i is +1 and class j is -1
%
% The program outputs the following variables:
% Theta      -- the RBM parameters, where:
%   Theta.vishid     -- the weight w for visible-hidden, it is a d1*d2 matrix, each entry of which denotes the weight w_ij for visible i to hidden j
%   Theta.visbias    -- the weight a for visible, it is a d1*1 vector, each dimension denotes the weight a_i
%   Theta.hidbias    -- the weight b for hidden, it is a d2*1 vector, each dimension denotes the weight b_j
% batchposhidprobs   -- the probability vector [P(h_1=1|x), ..., P(h_d2=1|x)] based on batchdata, it is a numcases*numhid*numbatches
% batchposhidprobs_aux  -- the probability vector [P(h_1=1|x), ..., P(h_d2=1|x)] based on batchdata_aux, it is a numcases_aux*numhid*numbatches_aux

restart	       = 1;
steplen_w      = 0.0001; % Learning rate for weights
steplen_vb     = 0.0001; % Learning rate for biases of visible units
steplen_hb     = 0.0001; % Learning rate for biases of hidden units
initialmomentum  = 0.9;
finalmomentum    = 0.9;

lambda1 = params.lambda1;
lambda2 = params.lambda2;
numhid = params.numhid;
maxepoch = params.maxepoch_rbm_init;
gammaI = params.gammaI;
gammaA = params.gammaA;
samplerate = params.samplerate;

vishid = Theta.vishid;
visbias = Theta.visbias;
hidbias = Theta.hidbias;

index_map = F.index_map;

[numcases numdims numbatches]=size(batchdata);
[numcases_aux numdims_aux numbatches_aux]=size(batchdata_aux);
one_mat = ones(numdims,numdims);
ntotal = numcases*numbatches + numcases_aux*numbatches_aux;

if restart ==1,
    restart=0;
    epoch=1;
    
    % Initializing symmetric weights and biases.
    %load('vishid3');
    %hidbias  = zeros(1,numhid); %% the bias of hidden nodes
    %visbias  = zeros(1,numdims);%% the bias of visable nodes
    
    poshidprobs = zeros(numcases,numhid);
    neghidprobs = zeros(numcases,numhid);
    poshidprobs_aux = zeros(numcases_aux,numhid);
    neghidprobs_aux = zeros(numcases_aux,numhid);
    posprods    = zeros(numdims,numhid);
    negprods    = zeros(numdims,numhid);
    vishidinc  = zeros(numdims,numhid);
    visbiasinc = zeros(1,numdims);
    hidbiasinc = zeros(1,numhid);
    sigmainc = zeros(1,numhid);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    batchposhidstates = zeros(numcases,numhid,numbatches);
    batchposhidprobs_aux = zeros(numcases_aux,numhid,numbatches_aux);
    batchposhidstates_aux = zeros(numcases_aux,numhid,numbatches_aux);
end

for epoch = epoch:maxepoch,
    %fprintf(1,'rbm epoch %d\r',epoch);
    errsum=0;

    %% This is for the labeled data
    for batch = 1:numbatches,
        fprintf(1,'labeled part: epoch %d batch %d\r\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs =  1./(1 + exp(-data*vishid - repmat(hidbias,numcases,1)));
        batchposhidprobs(:,:,batch) = poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs>rand(numcases,numhid);
	batchposhidstates(:,:,batch) = poshidstates;
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = poshidstates*vishid'+repmat(visbias,numcases,1);
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbias,numcases,1)));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%%%%%%%%% GRADIENT OF REGULARIZERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	batchgradOmega1w = zeros(numdims,numhid);
	batchgradOmega1b = zeros(1,numhid);
	batchgradOmega2w = zeros(numdims,numhid);
	batchgradOmega2b = zeros(1,numhid);
	gradgw = poshidprobs .* (1 - poshidprobs); % numcases*numhid matrix
	label = batchlabel(:,batch);
	z = zeros(length(label),1);
	tau = batchtau{batch}; % it is a sparse matrix for adjacency
	
	R1 = sum((ones(1,size(vishid,1))*vishid).^2);
	err2 = lambda1*R1;
	K = size(index_map,1); % the number of binary classifiers
	%for j=1:K
	% for efficiency, only sample a subuset of index_map to compute the regularizer
	sample = randperm(K);
	N = floor(K * samplerate);
	for l=1:N
	    j = sample(l);

	    %% construct positive and negative sets
	    posclass = index_map(j,1);
	    negclass = index_map(j,2);
	    pindex = find(label==posclass);
	    z(pindex) = 1;
	    nindex = find(label==negclass);
	    z(nindex) = -1;

	    v = F(j).v_all; % 1*numhid vector
	    e = F(j).classifier_all.b;

	    %% labeled data part, for Omega1
	    index = find(z~=0); % labeled data for binary classifier on (posclass, negclass)
	    gx = poshidprobs(index,:);
	    y = z(index);
	    delta = CheckHingeLoss(gx, y, v, e); % length(y)*1 vector		
	    temp = -1 * repmat(y.*delta, [1,numhid]) .* gradgw(index,:) .* repmat(v, [length(index),1]); % numcases*numhid matrix
	    gradOmega1w = data(index,:)' * temp; % numdims*numhid matrix
	    gradOmega1b = ones(1,length(index)) * temp; % 1*numhid vector
	    batchgradOmega1w = batchgradOmega1w + gradOmega1w;
	    batchgradOmega1b = batchgradOmega1b + gradOmega1b;

	    L2 = sum(HingeLoss(gx, y, v, e).^2);

	    %% Laplacian part, for Omega2
	    if ~isempty(tau)
	        [ind1, ind2, tau12] = find(tau);
	        gx1 = poshidprobs(ind1,:);
	        gx2 = poshidprobs(ind2,:);
	        y1 = gx1 * v' + e;
	        y2 = gx2 * v' + e;
		%for i=1:length(ind1)
		%    batchgradOmega2w = batchgradOmega2w + 2 * gammaI * (y2(i) - y1(i)) * tau12(i) * (data(ind1(i),:)' * gradgw(ind1(i),:) - data(ind2(i),:)' * gradgw(ind2(i),:));
		%    batchgradOmega2b = batchgradOmega2b + 2 * gammaI * (y2(i) - y1(i)) * tau12(i) * (gradgw(ind1(i),:) - gradgw(ind2(i),:));
		%end
		%% an equivalent, but more efficient form
		a = 2 * gammaI * (y2 - y1) .* tau12;
		b1 = repmat(a, [1,numhid]) .* gradgw(ind1,:);
		b2 = repmat(a, [1,numhid]) .* gradgw(ind2,:);
		batchgradOmega2w = batchgradOmega2w + (data(ind1,:) .'* b1 - data(ind2,:) .'* b2);
		batchgradOmega2b = batchgradOmega2b + sum(b1 - b2, 1);

		L2 = L2 + gammaI * sum(((y2-y1).^2) .* tau12);
	    end

	    err2 = err2 + lambda2 * L2;
	end
	

	%%%%%%%%% SET MOMENTUM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + err2 + errsum;
        if epoch>50,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            steplen_w*( (posprods-negprods)/numcases - lambda1*2*one_mat*vishid - lambda2*(batchgradOmega1w + batchgradOmega2w)/N );
        visbiasinc = momentum*visbiasinc + steplen_vb*(posvisact-negvisact)/numcases;
        hidbiasinc = momentum*hidbiasinc + steplen_hb*( (poshidact-neghidact)/numcases - lambda2*(batchgradOmega1b + batchgradOmega2b)/N );
        vishid = vishid + vishidinc;
        visbias = visbias + visbiasinc;
        hidbias = hidbias + hidbiasinc;

	%a = (posprods-negprods)/numcases;
	%b = lambda1*2*one_mat*vishid;
	%c = lambda2*(batchgradOmega1w + batchgradOmega2w)/N;
	%[mean(mean(a)), mean(mean(b)), mean(mean(c))]
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end

    %% This is for the unlabeled data
    for batch = 1:numbatches_aux,
        fprintf(1,'unlabeled part: epoch %d batch %d\r\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data_aux = batchdata_aux(:,:,batch);
        poshidprobs_aux =  1./(1 + exp(-data_aux*vishid - repmat(hidbias,numcases_aux,1)));
        batchposhidprobs_aux(:,:,batch) = poshidprobs_aux;
        posprods    = data_aux' * poshidprobs_aux;
        poshidact   = sum(poshidprobs_aux);
        posvisact = sum(data_aux);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates_aux = poshidprobs_aux>rand(numcases_aux,numhid);
	batchposhidstates_aux(:,:,batch) = poshidstates_aux;
        
       
	%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata_aux = poshidstates_aux*vishid'+repmat(visbias,numcases_aux,1);
        neghidprobs_aux = 1./(1 + exp(-negdata_aux*vishid - repmat(hidbias,numcases_aux,1)));
        negprods  = negdata_aux' * neghidprobs_aux;
        neghidact = sum(neghidprobs_aux);
        negvisact = sum(negdata_aux);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%%%%%%%%% GRADIENT OF REGULARIZERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	batchgradOmega2w_aux = zeros(numdims,numhid);
	batchgradOmega2b_aux = zeros(1,numhid);
	L2 = 0;
	R1 = sum((ones(1,size(vishid,1))*vishid).^2);
	err2 = lambda1*R1;
	tau = batchtau_aux{batch}; % it is a sparse matrix for adjacency
	if ~isempty(tau)
	    gradgw = poshidprobs_aux .* (1 - poshidprobs_aux); % numcases_aux*numhid matrix
	    K = size(index_map,1); % the number of binary classifiers

	    %for j=1:K
	    % for efficiency, only sample a subuset of index_map to compute the regularizer
	    sample = randperm(K);
	    N = floor(K * samplerate);
	    for l=1:N
	    	j = sample(l);

		v = F(j).v_all; % 1*numhid vector
		e = F(j).classifier_all.b;

	    	%% Laplacian part, for Omega2
	    	[ind1, ind2, tau12] = find(tau);
	    	gx1 = poshidprobs_aux(ind1,:);
	    	gx2 = poshidprobs_aux(ind2,:);
	    	y1 = gx1 * v' + e;
		y2 = gx2 * v' + e;
		%for i=1:length(ind1)
		%    batchgradOmega2w_aux = batchgradOmega2w_aux ...
		%	+ 2 * gammaI * (y2(i) - y1(i)) * tau12(i) * (data_aux(ind1(i),:)' * gradgw(ind1(i),:) - data_aux(ind2(i),:)' * gradgw(ind2(i),:));
		%    batchgradOmega2b_aux = batchgradOmega2b_aux + 2 * gammaI * (y2(i) - y1(i)) * tau12(i) * (gradgw(ind1(i),:) - gradgw(ind2(i),:));
		%end
		%% an equivalent, but more efficient form
		a = 2 * gammaI * (y2 - y1) .* tau12;
		b1 = repmat(a, [1,numhid]) .* gradgw(ind1,:);
		b2 = repmat(a, [1,numhid]) .* gradgw(ind2,:);
		batchgradOmega2w_aux = batchgradOmega2w_aux + (data_aux(ind1,:) .'* b1 - data_aux(ind2,:) .'* b2);
		batchgradOmega2b_aux = batchgradOmega2b_aux + sum(b1 - b2, 1);

		L2 = L2 + gammaI * sum(((y2-y1).^2) .* tau12);
	    end

	    err2 = err2 + lambda2 * L2;
	end

	%%%%%%%%% SET MOMENTUM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data_aux-negdata_aux).^2 ));
        errsum = err + err2 + errsum;
        if epoch>50,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;        

	%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            steplen_w*( (posprods-negprods)/numcases_aux - lambda1*2*one_mat*vishid - lambda2*batchgradOmega2w_aux/N );
        visbiasinc = momentum*visbiasinc + steplen_vb*(posvisact-negvisact)/numcases_aux;
        hidbiasinc = momentum*hidbiasinc + steplen_hb*( (poshidact-neghidact)/numcases_aux - lambda2*batchgradOmega2b_aux/N );
	vishid = vishid + vishidinc;
        visbias = visbias + visbiasinc;
        hidbias = hidbias + hidbiasinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end

    total_error(epoch) = errsum / ntotal; % normalize the error by the number of instances 
    fprintf(1, 'rbm epoch %4i error %f \n', epoch, total_error(epoch));
    
end
%%plot(total_error)
rbm_obj = total_error(maxepoch);

Theta = struct('vishid', vishid, ...
               'visbias', visbias, ...
               'hidbias', hidbias);

end


function delta = CheckHingeLoss(gx, y, v, e) 

%% This function takes the following variables as input:
%% gx  -- RBM features, it is a numcases*numhid matrix
%% y   -- labels, it is a numcases*1 vector
%% v   -- LapSVM feature weights, it is a 1*numhid vector
%% e   -- LapSVM bias term, it is a scalar
%%
%% This function returns the following varaible:
%% delta -- indicator function, it is a numcases*1 vector: value(i) = 1 if 1 - y_i*f(g(x_i)) >= 0, value(i)= 0 otherwise.

numcases = size(gx,1);
f = gx * v' + e;
delta = (y.*f <= 1);

end

function f = HingeLoss(gx, y, v, e) 

%% This function takes the following variables as input:
%% gx  -- RBM features, it is a numcases*numhid matrix
%% y   -- labels, it is a numcases*1 vector
%% v   -- LapSVM feature weights, it is a 1*numhid vector
%% e   -- LapSVM bias term, it is a scalar
%%
%% This function returns the following varaible:
%% delta -- indicator function, it is a numcases*1 vector: value(i) = 1 if 1 - y_i*f(g(x_i)) >= 0, value(i)= 0 otherwise.

numcases = size(gx,1);
f = gx * v' + e;
index = find(1-f < 0);
f(index) = 0;

end
