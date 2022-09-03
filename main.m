clc; clear all ; close all ; 
load('Ex3.mat'); load('AllElectrodes.mat');
fs = 256;
%% Generating C0 , C1 matrix
train_num = size(TrainData,3);                % num of data
TrainData_m = zeros(size(TrainData,1),size(TrainData,2),size(TrainData,3));
% assumption : mean of each channel is zero
for i = 1:train_num
    TrainData_m(:,:,i) = TrainData(:,:,i)- mean(TrainData(:,:,i),2);
end
class0_idx = find(TrainLabel==0);             % idx of class0
class1_idx = find(TrainLabel==1);             % idx of class1
X0 = TrainData_m(:,:,class0_idx);               % class 0 Data
X1 = TrainData_m(:,:,class1_idx);               % class 1 Data
C0 = zeros(30,30);
C1 = zeros(30,30);
for i = 1:length(class0_idx)
    C0 = C0 + X0(:,:,i)*transpose(X0(:,:,i));
end
C0 = C0/(length(class0_idx));
for j = 1:length(class1_idx)
    C1 = C1 + X1(:,:,j)*transpose(X1(:,:,j));
end
C1 = C1/(length(class1_idx));
%% claculating the filters
[W,D] = eig(C0,C1);
[d,ind] = sort(diag(D),'ascend');
D = D(ind,ind);
W = W(:,ind);
for i = 1:30
    W(:,i) = W(:,i)/norm( W(:,i));
end
%% W for c0/c1 
out_1 = W(:,1)'*TrainData_m(:,:,2);
out_2 = W(:,30)'*TrainData_m(:,:,2);
t = 0:1/fs:(length(out_1)-1)/fs;
figure()
subplot(2,1,1)
plot(t,out_1);
title('effect of 1st filter for label 0');
xlabel('time');
subplot(2,1,2);
plot(t,out_2);
title('effect of lat filter for label 0');
xlabel('time');

figure()
plot(t,out_1);
hold on
plot(t,out_2);
legend('filter one','filter 30');
title('effect of filters for a data with label 0');


out_1 = W(:,1)'*TrainData_m(:,:,3);
out_2 = W(:,30)'*TrainData_m(:,:,3);
t = 0:1/fs:(length(out_1)-1)/fs;
figure()
subplot(2,1,1)
plot(t,out_1);
title('effect of 1st filter for label1');
xlabel('time');
subplot(2,1,2);
plot(t,out_2);
title('effect of last filter for label =1');
xlabel('time');

figure()
plot(t,out_1);
hold on
plot(t,out_2);
legend('filter one','filter 30');
title('effect of filters for a data with label 1');

%% part B
load('AllElectrodes.mat')
cell_electrod = struct2cell(AllElectrodes);
thirty_labels = [37 7 5 40 38 42 10 47 45 15 13 48 50 52 18 32 55 23 22 21 20 31 57 58 59 60 26 63 27 64];
X_pos = zeros(30,1);
Y_pos = zeros(30,1);
labels = {};
for i = 1:30
    labels{i} = cell_electrod{1,1,thirty_labels(i)}; 
    X_pos(i) = cell_electrod{4,1,thirty_labels(i)};
    Y_pos(i) = cell_electrod{5,1,thirty_labels(i)};
end

figure()
plottopomap(X_pos,Y_pos,labels,abs(W(:,1)));
title('CSP filter for 1st filter');
figure()
plottopomap(X_pos,Y_pos,labels,abs(W(:,30)));
title('CSP filter for last filter');



%%
acc_tot = zeros(1,15);
for f = 1:15
    X = zeros(2*f,165);
    w_temp = W(:,[1:f,30-f+1:30]);
    for i = 1:165
        temp = w_temp'*TrainData_m(:,:,i);
        X(:,i) = var(transpose(temp));
    end
    for i = 1:3
        selected = 1+(i-1)*55:i*55;
        all = 1:165;
        other = setdiff(all,selected);
        train_data = X(:,other);
        test_data = X(:,selected);
        train_label = TrainLabel(other);
        test_label = TrainLabel(selected);
        SVMModel = fitcsvm(train_data',train_label');
        predicted_label = predict(SVMModel,test_data');
        acc_tot(f) =  acc_tot(f) + sum(predicted_label == test_label')/(55*3);
    end
end
f_best = find(max(acc_tot) == acc_tot);
disp([" the best f is : "+ num2str(f_best)+ " and the corrsponded acc is : "+ num2str(acc_tot(f_best))])
%%
w_best = W(:,[1:f_best,30-f_best+1:30]);
X_test = zeros(2*f_best,45);
for i = 1:45
    temp = w_best'*TestData(:,:,i);
    X_test(:,i) = var(transpose(temp));
end
X = zeros(2*f_best,165);
for i = 1:165
    temp = w_best'*TrainData(:,:,i);
    X(:,i) = var(transpose(temp));
end

SVMModel = fitcsvm(X',TrainLabel);
y_test = predict(SVMModel,X_test');
save("TestLabel_eval.mat","y_test")



