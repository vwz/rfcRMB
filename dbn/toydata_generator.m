function [data_train label] = toydata_generator()
%% data dimension: 10
%% data in group 1 500
%% data in group 2 500
%% data is divided into 10 groups
% % % % % % 
% C1 = -1+2*rand(10,1);
% %% C1^T D1= 1
% 
% C2 = -1+2*rand(10,1);
% %% C2^T D2= 1

% D1 = rand(10,500);
% t = C1'*D1;
% D1 = D1.*repmat(1./t,10,1);
% 
% D2 = rand(10,500);
% t = C2'*D2;
% D2 = D2.*repmat(1./t,10,1);
% 
% D = [D1';D2'];
% 
% save('hyperplane.mat','D','C1','C2','D1','D2');

% mu1 = 1*ones(1,10);
% sigma = eye(10);
% mu2 = -1*ones(1,10);
% 
% D1 = mvnrnd(mu1,sigma,500);
% D2 = mvnrnd(mu2,sigma,500);
% 
% a = -1;
% b = 1;
% D = [D1; D2]+3*randn(1000,10);

% D = [zeros(500,9), 5*ones(500,1);zeros(500,9),-5*ones(500,1)];%+0.2*rand(1000,10);


base1 = rand(1,10);
theta= 20*rand(500,1);

D1 = repmat(base1,500,1).*repmat(theta,1,10);

base2 = rand(1,10);
theta= 20*rand(500,1);

D2 = repmat(base2,500,1).*repmat(theta,1,10);%+10;
D = [D1; D2];%+3*randn(1000,10);    
degree = acosd(base1*base2'/(sqrt(base1*base1')*sqrt(base2*base2')))
save('line.mat','base1','base2','degree');


label_init = [ones(500,1), -1*ones(500,1)];

indperm = randperm(1000);

for batchNo = 0:9
    data_train(:,:,batchNo+1) = D(indperm(batchNo*100+1:(batchNo+1)*100),:);
    label(batchNo+1,:) = label_init(indperm(batchNo*100+1:(batchNo+1)*100));
end

% figure; imshow(mat2gray(D1));
% figure;imshow(mat2gray(D2));
% figure;imshow(mat2gray(D));
