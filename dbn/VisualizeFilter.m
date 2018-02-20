function VisualizeFilter();
%% The script is coded by Shenghua GAO

faceW = 28; 
faceH = 28; 
numPerLine = 1; 
ShowLine = 2; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	 Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(vishid(:,i*numPerLine+j+1),[faceH,faceW]); 
  	end 
end 
figure;
imshow(mat2gray(Y))