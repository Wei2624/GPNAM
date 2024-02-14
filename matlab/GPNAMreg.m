function [w, Z, c] = GPAMreg(y,X, kern_width, rff_num_feat)

%%% TRAIN %%%
[num,dim] = size(X);
Z = norminv((1:rff_num_feat)/(rff_num_feat+1),0,1)';
[~,c] = sort(rand(rff_num_feat,dim),1);
c = 2*pi*c/rff_num_feat;
c = ones(rff_num_feat,dim);

% CONJUGATE GRADIENTS
partition_size = 1;
A = zeros(rff_num_feat*dim+1);
b = zeros(rff_num_feat*dim+1,1);
vec = zeros(rff_num_feat*dim+1,partition_size);
out = zeros(partition_size,1);
count = 0;
for i = 1:num
    count = count + 1;
    mat = sqrt(2/rff_num_feat)*cos(Z*(X(i,:)./kern_width)+c);
    vec(:,count) = [mat(:) ; 1];
    out(count) = y(i);
    if mod(i,partition_size) == 0
        disp(num2str(i));
        A = A + vec*vec';
        b = b + vec*out;
        vec = zeros(rff_num_feat*dim+1,partition_size);
        out = zeros(partition_size,1);
        count = 0;
    end
end
A = A + vec*vec';
b = b + vec*out;

A2 = A + eye(dim*rff_num_feat+1)/20;
A2(end,end) = A(end,end);

w = zeros(dim*rff_num_feat+1,1);
r = A2*w - b;
p = -r;
b2 = b'*b;
for k = 1:dim*rff_num_feat+1
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
rff_num_feat = 50;
feat = Xtest;
resp = ytest;
[num_test,dim] = size(feat);
MSE = 0;
err = [];
for i = 1:num_test
    mat = sqrt(2/rff_num_feat)*cos(Z*(feat(i,:)./kern_width)+c);
    vec = [mat(:) ; 1];
    MSE = MSE + (resp(i)-vec'*w)^2/num_test;
    err(i) = resp(i)-vec'*w;
end
sqrt(MSE)

