function [Theta, batchposhidprobs, batchposhidprobs_aux] = rfcRBM_rbminit(batchdata, batchdata_aux, params)

%% This program tains regularized RBM of the following form:
%%	min - L1 + lambda1 * R1
%% where 
%% 	L1 = log P(v),
%%	R1 = |1^T vishid|_F^2.%%
%% The code is based on G. Hinton's code: rbmhidlinear.m
%
% The program assumes that the following variables are set externally:
% batchdata  -- the labeled data that is divided into batches (numcases numdims numbatches)
% batchdata_aux   -- the auxiliary unlabeled data that is divided into batches (numcases numdims numbatches)
% params     -- the parameter struct, defined in main function
%
% The program outputs the following variables:
% Theta      -- the RBM parameters, where:
%   Theta.vishid     -- the weight w for visible-hidden, it is a d1*d2 matrix, each entry of which denotes the weight w_ij for visible i to hidden j
%   Theta.visbias    -- the weight a for visible, it is a d1*1 vector, each dimension denotes the weight a_i
%   Theta.hidbias    -- the weight b for hidden, it is a d2*1 vector, each dimension denotes the weight b_j
% batchposhidprobs   -- the probability vector [P(h_1=1|x), ..., P(h_d2=1|x)] based on batchdata, it is a numcases*numhid*numbatches
% batchposhidprobs_aux  -- the probability vector [P(h_1=1|x), ..., P(h_d2=1|x)] based on batchdata_aux, it is a numcases_aux*numhid*numbatches_aux


restart	       = 1; % set to 1 if learning starts from beginning
steplen_w      = 0.0001; % Learning rate for weights
steplen_vb     = 0.0001; % Learning rate for biases of visible units
steplen_hb     = 0.0001; % Learning rate for biases of hidden units
initialmomentum  = 0.25;
finalmomentum    = 0.9;

lambda1 = params.lambda1;
numhid = params.numhid;
maxepoch = params.maxepoch_rbm_noinit;

[numcases numdims numbatches]=size(batchdata);
[numcases_aux numdims_aux numbatches_aux]=size(batchdata_aux);
one_mat = ones(numdims,numdims);
ntotal = numcases*numbatches + numcases_aux*numbatches_aux;

if restart ==1,
    restart=0;
    epoch=1;
    
    % Initializing symmetric weights and biases.
    %vishid   = 0.1*randn(numdims, numhid); %% the weight of vishid
    %save('vishid2', 'vishid');
    load('vishid3');
    hidbias  = zeros(1,numhid); %% the bias of hidden nodes
    visbias  = zeros(1,numdims);%% the bias of visable nodes
    
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

	R1 = sum((ones(1,size(vishid,1))*vishid).^2);

	%%%%%%%%% SET MOMENTUM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 )) + lambda1*R1;
        errsum = err + errsum;
        if epoch>50,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            steplen_w*( (posprods-negprods)/numcases - lambda1*2*one_mat*vishid);
        visbiasinc = momentum*visbiasinc + (steplen_vb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (steplen_hb/numcases)*(poshidact-neghidact);
        vishid = vishid + vishidinc;
        visbias = visbias + visbiasinc;
        hidbias = hidbias + hidbiasinc;
        
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

	R1 = sum((ones(1,size(vishid,1))*vishid).^2);

	%%%%%%%%% SET MOMENTUM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data_aux-negdata_aux).^2 )) + lambda1*R1;
        errsum = err + errsum;
        if epoch>50,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;        

	%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            steplen_w*( (posprods-negprods)/numcases_aux - lambda1*2*one_mat*vishid);
        visbiasinc = momentum*visbiasinc + (steplen_vb/numcases_aux)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (steplen_hb/numcases_aux)*(poshidact-neghidact);
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
