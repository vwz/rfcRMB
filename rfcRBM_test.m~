function [Y] = rfcRBM_test(X, Theta, F)

% This program tests the rfcRBM. 
%
% The program assumes that the following variables are set externally:
% X  -- the test data, it is a matrix (numcases numhid)
% Theta  -- the RBM parameters, where:
%   Theta.vishid     -- the weight w for visible-hidden, it is a d1*d2 matrix, each entry of which denotes the weight w_ij for visible i to hidden j
%   Theta.visbias    -- the weight a for visible, it is a d1*1 vector, each dimension denotes the weight a_i
%   Theta.hidbias    -- the weight b for hidden, it is a d2*1 vector, each dimension denotes the weight b_j
% F      -- the parameters for LapSVM, where
%    F.X_all            -- the training data, it is a model_num*1 cells, each cell is a numcases*numhids matrix
%    F.classifier_all   -- the classifiers for each binary classfication task, it is a model_num*1 cells, each cell includes classifier.alpha and classifier.b
%    F.index_map         -- the mapping function between model number and the class pair, it is a model_num*2 matrix, each row (i,j) indicates that class i is +1 and class j is -1
%
% The program outputs the following variables:
% Y  -- the test labels, it is a vector (numcases 1)

vishid = Theta.vishid; 
hidbias = Theta.hidbias;

% do feature transformation
numcases = size(X,1);
poshidprobs =  1./(1 + exp(-X*vishid - repmat(hidbias,numcases,1)));

% do prediction
[Y] = rfcRBM_LapSVMtest(poshidprobs, F);

end
