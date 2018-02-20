function GenDataDBN

load Data_all_device.mat;

xmax = max(max(X1trn));
X1trn = X1trn / xmax;
xmean = mean(X1trn,1);

%NX1trn = getnorm(X1trn, xmean);
%NX1tst = getnorm(X1tst, xmean);
%NX2trn = getnorm(X2trn, xmean);
%NX2tst = getnorm(X2tst, xmean);

NX1trn = getmaxnorm(X1trn, xmean, xmax);
NX1tst = getmaxnorm(X1tst, xmean, xmax);
NX2trn = getmaxnorm(X2trn, xmean, xmax);
NX2tst = getmaxnorm(X2tst, xmean, xmax);

B1trn = GenDBNfeatures(NX1trn);
B2trn = GenDBNfeatures(NX2trn);
%save('dbnbatchdata'); %normalized by column
%save('dbnbatchdata_row'); %normalized by row
%save('dbnbatchdata_mean'); %only mean
save('dbnbatchdata_max_mean'); %normalized by max and then mean

end

function B = GenDBNfeatures(X)

numbatches = 10;
numtotal = size(X,1);
rindex = randperm(numtotal);
numeach = floor(numtotal / numbatches);
batchdata = [];
for i=1:10
  startId = 1 + (i-1) * numeach;
  endId = i * numeach;
  %if i==10
  %  endId = numtotal;
  %end
  batch = X(startId:endId, :);
  batchdata(:,:,i) = batch; 
end

[numcases numdims numbatches] = size(batchdata);
B = batchdata;

end

function [NX] = getnorm(X, xmean)

NX = X - repmat(xmean, size(X,1), 1);
%NX1 = repmat(sqrt(sum(NX.^2,1)), size(NX,1), 1); %normalized by column
%NX1 = repmat(sqrt(sum(NX.^2,2)), 1, size(NX,2)); %normalized by row
%NX = NX ./ NX1;

end

function [NX] = getmaxnorm(X, xmean, xmax)

X = X / xmax;
NX = X - repmat(xmean, size(X,1), 1);
%NX1 = repmat(sqrt(sum(NX.^2,1)), size(NX,1), 1); %normalized by column
%NX1 = repmat(sqrt(sum(NX.^2,2)), 1, size(NX,2)); %normalized by row
%NX = NX ./ NX1;

end
