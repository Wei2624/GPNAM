function [w, Z, c, kern_width] = GPAMreg(y,X)

%%% TRAIN %%%
[num,dim] = size(X);
num_feat = 50;
kern_width = std(X,[],1)/24; + (quantile(X,.60)-quantile(X,.4));
Z = norminv((1:num_feat)/(num_feat+1),0,1)';
[~,c] = sort(rand(num_feat,dim),1);
c = 2*pi*c/num_feat;
c = ones(num_feat,dim);

% CONJUGATE GRADIENTS
partition_size = 1;
A = zeros(num_feat*dim+1);
b = zeros(num_feat*dim+1,1);
vec = zeros(num_feat*dim+1,partition_size);
out = zeros(partition_size,1);
count = 0;
for i = 1:num
    count = count + 1;
    mat = sqrt(2/num_feat)*cos(Z*(X(i,:)./kern_width)+c);
    vec(:,count) = [mat(:) ; 1];
    out(count) = y(i);
    if mod(i,partition_size) == 0
        disp(num2str(i));
        A = A + vec*vec';
        b = b + vec*out;
        vec = zeros(num_feat*dim+1,partition_size);
        out = zeros(partition_size,1);
        count = 0;
    end
end
A = A + vec*vec';
b = b + vec*out;

A2 = A + eye(dim*num_feat+1)/20;
A2(end,end) = A(end,end);

w = zeros(dim*num_feat+1,1);
r = A2*w - b;
p = -r;
b2 = b'*b;
for k = 1:dim*num_feat+1
    r1 = r'*r;
    Ap = A2*p;
    alpha = r1/(p'*Ap);
    w = w + alpha*p;
    r = r + alpha*Ap;
    r2 = r'*r;
    beta = r2/r1;
    p = -r + beta*p;
    if sqrt(r2/b2) < 10^-4
        break
    end
    if mod(k,100) == 0
        disp([num2str(k) ' : ' num2str(sqrt(r2/b2))])
    end
end

%%% TEST %%%
num_feat = 50;
feat = Xtest;
resp = ytest;
[num_test,dim] = size(feat);
MSE = 0;
err = [];
for i = 1:num_test
    mat = sqrt(2/num_feat)*cos(Z*(feat(i,:)./kern_width)+c);
    vec = [mat(:) ; 1];
    MSE = MSE + (resp(i)-vec'*w)^2/num_test;
    err(i) = resp(i)-vec'*w;
end
sqrt(MSE)

