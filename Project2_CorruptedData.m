F=dlmread('corrupted_iris_dataset.dat');
F =  F(randperm(end),:);                     % Shuffling of data
cumulativeAccuracy=0;
for j=1:10
   % Following section is used for deviding given data set into test
   % dataset and trainging data set 
   testUpperBound=j*15;
   testLowerBound=testUpperBound-14;
   testData=F(testLowerBound:testUpperBound,:);     % Creating different test data for each iteration 
   trainData= F;                                    % Copying original data in training data 
   trainData(testLowerBound:testUpperBound,:)=[];   % Removing test data from original data to create valid training data 
   %Follwoing section is used to identify 3 classes of training dataset
   trainClassA = trainData(trainData(:,5) == 1,:);
   trainClassB = trainData(trainData(:,5) == 2,:);
   trainClassC = trainData(trainData(:,5) == 3,:);
   %Following section is used to identify 3 classes of test dataset
   testClassA = testData(testData(:,5) == 1,:);
   testClassB = testData(testData(:,5) == 2,:);
   testClassC = testData(testData(:,5) == 3,:);
   % Follwoing section is used to identify number of rows in each class within training dataset 
   [rowTrainClassA,colClassA]=size(trainClassA);
   [rowTrainClassB,colClassB]=size(trainClassB);
   [rowTrainClassC,colClassC]=size(trainClassC);
   % Follwoing section is used to identify number of rows in each class
   % within test dataset 
   rowTestClassA=50-rowTrainClassA;
   rowTestClassB=50-rowTrainClassB;
   rowTestClassC=50-rowTrainClassC;
   %Following section is used for considering first 4 columns of all
   %classes in training dataset
    a=trainClassA(:,1:4);
    b=trainClassB(:,1:4);
    c=trainClassC(:,1:4);
    d=4;                        % Here d= Number of features 
    %Finding mean of each class of training dataset
    mua=mean(a);
    mub=mean(b);
    muc=mean(c);
    % Subtracting mua from class elements 
    xminusmua=a- mua;                         % Mean matrix of a
    xminusmuaT=xminusmua';
    xminusmub=b- mub;                         % Mean matrix of b
    xminusmubT=xminusmub';
    xminusmuc=c- muc;                         % Mean matrix of c
    xminusmucT=xminusmuc';
    % Finding covariance of each matrix 
    cova=(1/(rowTrainClassA-1))*(xminusmuaT*xminusmua);         % Covariance matrix of a
    covb=(1/(rowTrainClassB-1))*(xminusmubT*xminusmub);         % Covariance matrix of b
    covc=(1/(rowTrainClassC-1))*(xminusmucT*xminusmuc);         % Covariance matrix of c
    newga=[];
    newgb=[];
    newgc=[];
    testDataLabel=testData(:,5)';
    resultDataLabel=[];
    % Finding discriminant function 
    for i=1:15
         testga= ((-0.5)*sum(sum(((testData(i,1:4)-mua)*inv(cov(c))*(testData(i,1:4)-mua)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cova))))-log(1/3);
         newga(end+1)=testga;
         testgb= ((-0.5)*sum(sum(((testData(i,1:4)-mub)*inv(cov(b))*(testData(i,1:4)-mub)'))))-((d/2)*log(2*pi))-(0.5*(log(det(covb))))-log(1/3);
         newgb(end+1)=testgb;
         testgc= ((-0.5)*sum(sum(((testData(i,1:4)-muc)*inv(cov(c))*(testData(i,1:4)-muc)'))))-((d/2)*log(2*pi))-(0.5*(log(det(covc))))-log(1/3);
         newgc(end+1)=testgc;      
    end
    resultclassa=0;
    resultclassb=0;
    resultclassc=0;
    %Comparing discriminant function
    for k=1:15
        if(newga(k)>newgb(k) && newga(k)>newgc(k))
            resultclassa=resultclassa+1;
        end
        if(newgb(k)>newga(k) && newgb(k)>newgc(k))
            resultclassb=resultclassb+1;
        end
        if(newgc(k)>newga(k) && newgc(k)>newgb(k))
            resultclassc=resultclassc+1; 
        end
    end
  accuracy= ((resultclassa/rowTestClassA)+(resultclassb/rowTestClassB)+(resultclassc/rowTestClassC))/3;
  fprintf("Iteration number:");
  disp(j);
  fprintf("Accuarcy during the iteration:");
  disp(accuracy);
  cumulativeAccuracy=cumulativeAccuracy+accuracy;
end
averageAccuracy=cumulativeAccuracy/10;           
fprintf("Average Accuarcy :");
disp(averageAccuracy);
