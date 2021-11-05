%%% Advanced Econometrics: Problem Set 5
%%% Jieyang Wang, UNI: 4067
clear;
clc;
warning('off');

%% Question 1
load datasetPS5

% Splitting data into training, cv and test sets
rng(2021);
dataset = [x,y];
r = randperm(size(dataset,1));
trainset = dataset(r(1:19), :);
cvset = dataset(r(20:25), :);
testset = dataset(r(26:31), :);

%%
% Getting training x and y
x1t = trainset(:,1);
yt = trainset(:,2);

%%
% Generating polynomial values
x2t = x1t.^2;
x3t = x1t.^3;
x4t = x1t.^4;
x5t = x1t.^5;
x6t = x1t.^6;
x7t = x1t.^7;
x8t = x1t.^8;
x9t = x1t.^9;
x10t = x1t.^10;

X1 = [ones(19,1),x1t];
X2 = [ones(19,1),x1t,x2t];
X3 = [ones(19,1),x1t,x2t,x3t];
X4 = [ones(19,1),x1t,x2t,x3t,x4t];
X5 = [ones(19,1),x1t,x2t,x3t,x4t,x5t];
X6 = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t];
X7 = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t,x7t];
X8 = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t,x7t,x8t];
X9 = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t,x7t,x8t,x9t];
X10 = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t,x7t,x8t,x9t,x10t];

% Generating Theta values
T1 = regress(yt,X1);
T2 = regress(yt,X2);
T3 = regress(yt,X3);
T4 = regress(yt,X4);
T5 = regress(yt,X5);
T6 = regress(yt,X6);
T7 = regress(yt,X7);
T8 = regress(yt,X8);
T9 = regress(yt,X9);
T10 = regress(yt,X10);

% Graphing cost of training
J1t = computeCost(X1,yt,T1);
J2t = computeCost(X2,yt,T2);
J3t = computeCost(X3,yt,T3);
J4t = computeCost(X4,yt,T4);
J5t = computeCost(X5,yt,T5);
J6t = computeCost(X6,yt,T6);
J7t = computeCost(X7,yt,T7);
J8t = computeCost(X8,yt,T8);
J9t = computeCost(X9,yt,T9);
J10t = computeCost(X10,yt,T10);

Jt = [J1t,J2t,J3t,J4t,J5t,J6t,J7t,J8t,J9t,J10t];
Jtl = log(Jt);

%%
% Cross Validation
x1c = cvset(:,1);
yc = cvset(:,2);

x2c = x1c.^2;
x3c = x1c.^3;
x4c = x1c.^4;
x5c = x1c.^5;
x6c = x1c.^6;
x7c = x1c.^7;
x8c = x1c.^8;
x9c = x1c.^9;
x10c = x1c.^10;

X1 = [ones(6,1),x1c];
X2 = [ones(6,1),x1c,x2c];
X3 = [ones(6,1),x1c,x2c,x3c];
X4 = [ones(6,1),x1c,x2c,x3c,x4c];
X5 = [ones(6,1),x1c,x2c,x3c,x4c,x5c];
X6 = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c];
X7 = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c,x7c];
X8 = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c,x7c,x8c];
X9 = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c,x7c,x8c,x9c];
X10 = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c,x7c,x8c,x9c,x10c];

J1 = computeCost(X1,yc,T1);
J2 = computeCost(X2,yc,T2);
J3 = computeCost(X3,yc,T3);
J4 = computeCost(X4,yc,T4);
J5 = computeCost(X5,yc,T5);
J6 = computeCost(X6,yc,T6);
J7 = computeCost(X7,yc,T7);
J8 = computeCost(X8,yc,T8);
J9 = computeCost(X9,yc,T9);
J10 = computeCost(X10,yc,T10);

J = [J1,J2,J3,J4,J5,J6,J7,J8,J9,J10];
Jl = log(J);

[M,I] = min(J); 
disp('Polynomial degree that minimizes Cross Validation cost is:');
disp(I);
% Lowest J value occurs at polynomial degree 7

figure(1);
plot(Jl);
hold on
plot(Jtl);
title 'Cross Validation vs Training Costs';
legend('Log CV Cost','Log Training Cost');
xlabel('Polynomial Degree');
ylabel('log(Cost)');
hold off

%%
% Testing
x1ts = testset(:,1);
yts = testset(:,2);

x2ts = x1ts.^2;
x3ts = x1ts.^3;
x4ts = x1ts.^4;
x5ts = x1ts.^5;
x6ts = x1ts.^6;
x7ts = x1ts.^7;
x8ts = x1ts.^8;
x9ts = x1ts.^9;
x10ts = x1ts.^10;

X7 = [ones(6,1),x1ts,x2ts,x3ts,x4ts,x5ts,x6ts,x7ts];
Jtest = computeCost(X7,yts,T7);

disp('Cost of testing sample for degree 7 is');
disp(Jtest);

%% Question 2

% Demeaning training variables and normalizing
x1tm = (x1t - mean(x1t))/std(x1t);
x2tm = (x2t - mean(x2t))/std(x2t);
x3tm = (x3t - mean(x3t))/std(x3t);
x4tm = (x4t - mean(x4t))/std(x4t);
x5tm = (x5t - mean(x5t))/std(x5t);
x6tm = (x6t - mean(x6t))/std(x6t);
x7tm = (x7t - mean(x7t))/std(x7t);
x8tm = (x8t - mean(x8t))/std(x8t);
x9tm = (x9t - mean(x9t))/std(x9t);
x10tm = (x10t - mean(x10t))/std(x10t);
ytm = (yt - mean(yt))/std(yt);

stdx1 = std(x1t);
stdx2 = std(x2t);
stdx3 = std(x3t);
stdx4 = std(x4t);
stdx5 = std(x5t);
stdx6 = std(x6t);
stdx7 = std(x7t);
stdx8 = std(x8t);
stdx9 = std(x9t);
stdx10 = std(x10t);
stdy = std(yt);

Xmean = [mean(x1t),mean(x2t),mean(x3t),mean(x4t),mean(x5t),mean(x6t),mean(x7t),mean(x8t),mean(x9t),mean(x10t)];
X10m = [x1tm,x2tm,x3tm,x4tm,x5tm,x6tm,x7tm,x8tm,x9tm,x10tm];
stant = [stdx1,stdx2,stdx3,stdx4,stdx5,stdx6,stdx7,stdx8,stdx9,stdx10];
XX = X10m'*X10m;
XY = X10m'*ytm;
X10c = [ones(6,1),x1c,x2c,x3c,x4c,x5c,x6c,x7c,x8c,x9c,x10c];
X10t = [ones(19,1),x1t,x2t,x3t,x4t,x5t,x6t,x7t,x8t,x9t,x10t];
jvalst = [];
jvalscv = [];

for i = 0:0.001:1
    beta = ones(10,1);
    betar = inv(XX+i*eye(10))*XY;
    for j = 1:10
        beta(j) = betar(j)*stdy/(stant(j));
    end
    b0 = mean(yt)-Xmean*beta;
    bols = [b0,beta'];
    jvalst = [jvalst,computeCost(X10t,yt,bols')];
    jvalscv = [jvalscv,computeCost(X10c,yc,bols')];
end

[M,I] = min(jvalscv); 
I = (I-1)*0.001;
disp('Lambda that minimizes Cross Validation cost is:');
disp(I);
disp('CV Cost at minimum is:');
disp(M);

jvalscvl = log(jvalscv);
jvalstl = log(jvalst);

figure(2);
plot(jvalscvl);
hold on
plot(jvalstl);
title 'Log Cross Validation and Log Training Costs';
legend('Log CV Cost','Log Training Cost');
xlabel('Lambda Index');
ylabel('log(Cost)');
hold off

%%

% Taking Lambda = 0.1000 for testing

x1ts = testset(:,1);
yts = testset(:,2);

x2ts = x1ts.^2;
x3ts = x1ts.^3;
x4ts = x1ts.^4;
x5ts = x1ts.^5;
x6ts = x1ts.^6;
x7ts = x1ts.^7;
x8ts = x1ts.^8;
x9ts = x1ts.^9;
x10ts = x1ts.^10;

XX = X10m'*X10m;
XY = X10m'*ytm;

beta = ones(10,1);
betar = inv(XX+0.1*eye(10))*XY;
for j = 1:10
    beta(j) = betar(j)*stdy/(stant(j));
end
b0 = mean(yt)-Xmean*beta;
bols = [b0,beta'];

X10ts = [ones(6,1),x1ts,x2ts,x3ts,x4ts,x5ts,x6ts,x7ts,x8ts,x9ts,x10ts];
jvalsts = computeCost(X10ts,yts,bols');

disp('Cost of testing sample for lambda = 0.1000 is');
disp(jvalsts);