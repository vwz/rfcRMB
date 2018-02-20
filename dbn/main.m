function main
%% topy data

close all

numhid1=10; numhid2=10; numhid3=10; numhid4=50; numhid5 = 5;
maxepoch = 5000;
restart = 1;

[batchdata batchlabel,] = toydata_generator();
% load('gaussian_data.mat','batchdata', 'batchlabel');

[numcases numdims numbatches]=size(batchdata);

% save('gaussian_data.mat','batchdata', 'batchlabel');
%% train the first layer rbm with real-valued visual nodes and binary
%% hidden nodes

ind=find(batchlabel(10,:)==-1);
ind2=find(batchlabel(10,:)==1);


% [vishid1, visbias1, hidbias1,batchposhidprobs1] = rbm_layer1(batchdata,10,maxepoch,restart);

[vishid1, visbias1, hidbias1,batchposhidprobs1,poshidstates] = rbm_layer1_sparse(batchdata,numhid1,maxepoch,restart);
% % % % % % % figure; imshow(mat2gray(batchposhidprobs1([ind ind2],:,10)))
% % % % % % % 
% % % % % % % 
% % % % % % % [vishid, visbias, hidbias,batchposhidprobs2] = rbm_binary_binary(batchposhidprobs1,numhid2,maxepoch,restart);
% % % % % % % figure; imshow(mat2gray(batchposhidprobs2([ind ind2],:,10)))
% % % % % % % 
% % % % % % % [vishid, visbias, hidbias,batchposhidprobs3] = rbm_binary_binary(batchposhidprobs2,numhid3,maxepoch,restart);
% % % % % % % figure; imshow(mat2gray(batchposhidprobs3([ind ind2],:,10)))
% % % % % % % 
% % % % % % % [vishid, visbias, hidbias,batchposhidprobs4] = rbm_binary_binary(batchposhidprobs3,numhid4,maxepoch,restart);
% % % % % % % figure; imshow(mat2gray(batchposhidprobs4([ind ind2],:,10)))
% % % % % % % 
% % % % % % % 
% % % % % % % [vishid, visbias, hidbias,batchposhidprobs5] = rbm_binary_binary(batchposhidprobs4,numhid5,maxepoch,restart);
% % % % % % % figure; imshow(mat2gray(batchposhidprobs5([ind ind2],:,10)))


% % % % % % % save('results.mat','batchposhidprobs1','batchposhidprobs2','batchposhidprobs3','batchposhidprobs4');




