function [w, Z, c] = GPAMclass(y,X)

%%% TRAIN
[num,dim] = size(X);
num_feat = 100;
kern_width = .2 +0*std(X,[],1)/3;
[~,c] = sort(rand(num_feat,dim),1);
c = 2*pi*c/num_feat;
Z = norminv((1:num_feat)/(num_feat+1),0,1)';
c = ones(num_feat,dim);

%%% LEARN LINEAR TERMS
w = zeros(dim*num_feat+1,1);
grad_mom = zeros(size(w));
err_save = 1;
for pass = 1:200
    tic
    temp = 0;
    % [~,id] = sort(rand(1,num));
    alpha = (1/100)/sqrt(pass);
    for i = 1:num
        mat = sqrt(2/num_feat)*cos(Z*(X(i,:)./kern_width)+c);
        vec = [mat(:) ; 1];
        sigmoid = exp(vec'*w);
        sigmoid = sigmoid/(1+sigmoid);
        % temp = temp + abs(sigmoid-y(i));
        temp = temp + (1 - abs(round(sigmoid) - y(i)));
        grad_mom = (1-.1)*grad_mom + .1*(y(i)-sigmoid)*vec;
        if i == 1 && pass == 1
            grad_mom = (y(i)-sigmoid)*vec;
        end

        % w = w + alpha*(grad_mom  - 1*w/num);
        w = w + alpha*grad_mom;
    end

            % %%% TEST
            % feat = Xtest;
            % lab = ytest;
            % [num_test,dim] = size(feat);
            % RMSE = 0;
            % prob = zeros(size(lab));
            % for i = 1:num_test
            %     mat = sqrt(2/num_feat)*cos(Z*(feat(i,:)./kern_width)+c);
            %     vec = [mat(:) ; 1];
            %     sigmoid = exp(vec'*w);
            %     prob(i) = sigmoid/(1+sigmoid);
            % end
            % err = mean(abs(lab-prob));
            % [prob,t2] = sort(prob,'descend');
            % lab_sort = lab(t2);
            % v1 = cumsum(lab_sort)/sum(lab_sort==1);
            % v2 = cumsum(1-lab_sort)/sum(lab_sort==0);
            % auc = v1'*(v2 - [0;v2(1:end-1)]);
    toc
        % [pass temp/num auc]
    [pass temp/num]
end

%%% TEST
feat = Xtest;
lab = ytest;
[num_test,num_feat] = size(feat);
RMSE = 0;
prob = zeros(size(lab));
for i = 1:num_test
    mat = sqrt(2/num_feat)*cos(Z*(feat(i,:)./kern_width)+c);
    vec = [mat(:) ; 1];
    sigmoid = exp(vec'*w);
    prob(i) = sigmoid/(1+sigmoid);
end
err = mean(abs(lab-prob));
[prob,t2] = sort(prob,'descend');
lab_sort = lab(t2);
v1 = cumsum(lab_sort)/sum(lab_sort==1);
v2 = cumsum(1-lab_sort)/sum(lab_sort==0);
auc = v1'*(v2 - [0;v2(1:end-1)])



