function [] = find_test_set_size()
%manually saved this file
load('aa_Male_Male.mat');

%load F0s
F0_1 = P1.f0;
F0_2 = P2.f0;

%load h1h2
H_1 = P1.h1h2;
H_2 = P2.h1h2;

%find good test set size
ks = 1:min(length(H_1),length(H_2));
accTrain = zeros(length(ks),1);
accTest = zeros(length(ks),1);
for n=1:length(ks)
    %make training set
    k = ks(n);
    [H_1Tr,i1] = datasample(H_1,k);
    F0_1Tr = F0_1(i1);
    [H_2Tr,i2] = datasample(H_2,k);
    F0_2Tr = F0_2(i2);
    
    %indices for test set
    x1 = ones(length(H_1),1);
    x1(i1) = 0;
    x1 = logical(x1);
    x2 = ones(length(H_2),1);
    x2(i2) = 0;
    x2 = logical(x2);
    
    %make test set
    H_1Te = H_1(x1);
    F0_1Te = F0_1(x1);
    H_2Te = H_2(x2);
    F0_2Te = F0_2(x2);
    
    Training = [H_1Tr , F0_1Tr; H_2Tr , F0_2Tr];
    GroupTrain = [ones(length(H_1Tr),1);zeros(length(H_2Tr),1)];
    
    %GET SVM
    svm = svmtrain(Training,GroupTrain,'kernel_function','rbf','method','LS');
    
    %check accuracy
    G = svmclassify(svm,Training);
    correct = sum(G==GroupTrain);
    accuracy = correct/length(GroupTrain);
    accTrain(n) = accuracy;
    
    Testing = [H_1Te , F0_1Te; H_2Te , F0_2Te];
    %actual results
    GroupTest = [ones(length(H_1Te),1);zeros(length(H_2Te),1)];
    
    %use classifier
    GroupSVM = svmclassify(svm,Testing);
    
    %Check accuracy
    correctTest = sum(GroupTest==GroupSVM);
    accuracyTest = correctTest/length(GroupSVM);
    accTest(n) = accuracyTest;
end

plot(accTrain,'DisplayName','accTrain');hold all;plot(accTest,'DisplayName','accTest');hold off;

end