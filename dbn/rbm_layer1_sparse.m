function [vishid, visbiases, hidbiases, batchposhidprobs, poshidstates] = rbm_layer1_sparse(batchdata, numhid, maxepoch, restart)
%% input is real-valued number
%% hidden layer is binary
%% min -log P(v) + weight_sparse * |W|_1 + weight_pair * |1^T W|_F^2 (here W is d_v*d_h)
%% coded by Shenghua Gao (ADSC)
%% The code is based on G. Hinton's code: rbmhidlinear.m

% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, tochastic real-valued feature detectors drawn from a unit
% variance Gaussian whose mean is determined by the input from
% the logistic visible units. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning

epsilonw      = 0.00001; % Learning rate for weights
epsilonvb     = 0.00001; % Learning rate for biases of visible units
epsilonhb     = 0.00001; % Learning rate for biases of hidden units
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;
weight_sparse = 0.5;%%control the weight of sparsity
weight_pair = 100;% control |1^T W|_F^2 = 0
[numcases numdims numbatches]=size(batchdata);
one_mat = ones(numdims,numdims);

if restart ==1,
    restart=0;
    epoch=1;
    
    % Initializing symmetric weights and biases.
    vishid     = 0.1*randn(numdims, numhid); %% the weight of W
    hidbiases  = zeros(1,numhid); %% the bias of hidden nodes
    visbiases  = zeros(1,numdims);%% the bias of visual nodes
    
    
    poshidprobs = zeros(numcases,numhid);
    neghidprobs = zeros(numcases,numhid);
    posprods    = zeros(numdims,numhid);
    negprods    = zeros(numdims,numhid);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    sigmainc = zeros(1,numhid);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    batchposhidstates = zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch,
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    
    for batch = 1:numbatches,
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs =  1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch) = poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %         poshidprobs =  (data*vishid) + repmat(hidbiases,numcases,1);
        %         batchposhidprobs(:,:,batch)=poshidprobs;
        %         posprods    = data' * poshidprobs;
        %         poshidact   = sum(poshidprobs);
        %         posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs>rand(numcases,numhid);
	batchposhidstates(:,:,batch) = poshidstates;
        
        
        %         poshidstates = poshidprobs+randn(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = poshidstates*vishid'+repmat(visbiases,numcases,1);
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        
        %         negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        %         neghidprobs = (negdata*vishid) + repmat(hidbiases,numcases,1);
        %         negprods  = negdata'*neghidprobs;
        %         neghidact = sum(neghidprobs);
        %         negvisact = sum(negdata);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         vishidinc = momentum*vishidinc + ...
%             epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);

        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weight_sparse*sign(vishid) - weight_pair*2*one_mat*vishid);

        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    total_error(epoch) =errsum; 
    fprintf(1, 'rbm epoch %4i error %f \n', epoch, errsum);
    
end
plot(total_error)
