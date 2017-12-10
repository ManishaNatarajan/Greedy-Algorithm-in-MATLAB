%In the train, valid and test data, remember the last column is label. Plot
%your figures in the defination of functions.
clc;
clear all;
load('train-greedy.mat')
load('valid-greedy.mat')
load('test-greedy.mat')
load('true-beta.mat')



[beta_fwd, pre_e_fwd, est_e_fwd] = forward_greedy(train,validation,test,beta);
pause;
[beta_ridge, pre_e_ridge,est_e_ridge] = ridge_reg(train,validation,test,beta);
pause;
[beta_lasso,pre_e,estim_e_lasso,optim_lambda] = lasso_wrapper(train,validation,test,beta);
pause;
[estim_e_refit,beta_refit] =refined_est(train,validation,test,beta);

% Part a, implement the forward greedy algorithm
% Input: train data, validation data and test data
% Output: number of optimized features, optimal beta, estimation error and 
%        prediction error.
%        Plot your errors as iteration changes
function [beta1, p_error, e_error] = forward_greedy(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = train(:,1:1000);
Y = train(:,1001);
X_v = validation(:,1:1000);
Y_v = validation(:,1001);
X_t = test(:,1:1000);
Y_t = test(:,1001);
A = [];
K = 30;
beta1 = zeros(1000,1);
val_error = [];
estim_error = [];
predict_error = [];
for k =1:K
    [val, index] = max(abs(X'*(X*beta1 - Y)));
    A =  [index, A];
    temp = quadprog(2*X(:,A)'*X(:,A),-2*X(:,A)'*Y);
    beta1(A,:) = temp;
%     train_error(k) = norm((Y-X*beta1),2)^2;
    val_error(k) = norm((Y_v-X_v*beta1),2)^2;
    estim_error(k) = norm((beta1 - beta),2);
    predict_error(k) = norm((Y_t-X_t*beta1),2)^2/(3000);
end
% 
p_error = norm((Y_t-X_t*beta1),2)^2/(3000);
e_error = norm((beta1 - beta),2);
%  plot(val_error);
%  title('validation error v/s no. of iterations');
%  xlabel('iterations');
%  ylabel('validation error');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%Part b, implement the ridge regreesion estimator
%Input: train data, validation data and test data
%Output: optimal beta, optimal lambda, estimation error and prediction error.
%        Plot your errors as iteration changes
function [beta_r,e_error,pre_error] = ridge_reg(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%        Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = train(:,1:1000);
Y = train(:,1001);
X_v = validation(:,1:1000);
Y_v = validation(:,1001);
X_t = test(:,1:1000);
Y_t = test(:,1001);
lambda = [0.0125, 0.025, 0.05, 0.1, 0.2];
beta_ridge = [];
for i = 1:length(lambda)
    beta_ridge(:,i) = inv(X'*X + 2*100*lambda(i)*eye(size(X'*X)))*X'*Y;
    error(i) = norm((beta_ridge(:,i) - beta),2);
    v_error(i) = norm((Y_v-X_v*beta_ridge(:,i)),2)^2;
    p_error(i) = norm(Y_t-X_t*beta_ridge(:,i),2)^2/3000;
end
beta_r = beta_ridge(:,1);
e_error = norm((beta_ridge(:,1) - beta),2);
pre_error = norm(Y_t-X_t*beta_ridge(:,1),2)^2/3000;

figure();
plot(lambda,error);
title('Estimation error v/s Lambda');
xlabel('Lambda');
ylabel('Estimation error');
figure();
plot(lambda,p_error);
title('prediction error v/s Lambda');
xlabel('Lambda');
ylabel('Estimation error');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%Part c, use lasso package to get optimized parameter.
%Input: train data, validation data and test data.
%Output: optimal beta, optimal lambda, estimation error and prediction
%        error. 
function [beta_lasso, pre_e, estim_e, optim_lambda] = lasso_wrapper(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = train(:,1:1000);
Y = train(:,1001);
X_v = validation(:,1:1000);
Y_v = validation(:,1001);
X_t = test(:,1:1000);
Y_t = test(:,1001);
[B,STATS] = lasso(X,Y);
B0 = STATS.Intercept;
l = STATS.Lambda;
% B = [B0; B];
index = 1;
error_min = inf;
for i = 1:size(B,2)
    error = norm((B(:,i) - beta),2);
    val_e(i) = norm((Y_v-X_v*B(:,i)-ones(100,1)*B0(i)),2)^2; %including intercept
    if(val_e(i)<error_min)
        index = i;
        error_min = val_e;
    end
end
plot(l,val_e);
beta_lasso = B(:,index);
pre_e = norm((Y_t-X_t*beta_lasso-ones(3000,1)*B0(index)),2)^2/3000;
estim_e = norm((beta_lasso - beta),2);
optim_lambda = STATS.Lambda(index);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end




%Part d, get your refined optimizer.
%Output: refined beta and estimation error.
function [estim_e,beta_refit] = refined_est(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = [];
X = train(:,1:1000);
Y = train(:,1001);
X_v = validation(:,1:1000);
Y_v = validation(:,1001);
X_t = test(:,1:1000);
Y_t = test(:,1001);
beta_lasso = lasso_wrapper(train,validation,test,beta);
for i =1:length(beta_lasso)
    if(beta_lasso(i)~=0)
        A = [i,A];
    end
    
end
beta_refit = zeros(size(X,1),1);
beta_refit(A)  = inv(X(:,A)'*X(:,A))*X(:,A)'*Y; %Consider only features for which beta is not 0

pre_e = norm((Y_t-X_t*beta_refit),2)^2/3000;
estim_e = norm((beta_refit - beta),2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Ends Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



    

