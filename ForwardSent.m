function [Z] = ForwardSent(sentence,params,oWe,Wsem)
% input one sentence at a time

N = length(sentence.word_index); %sus: number of words in sentence
Z = zeros(params.h, N); %sus: hxN. h is the size of the semantic space
n=1;

for i=1:N  %sus: iterate over each word in the sentence

    ix1 = sentence.word_index(i);  %sus: id word
    if ix1==-1, continue; end

    w1 = oWe(:,ix1);  %sus: get vector of word
    
    vcat = [w1; 1]; % cat sus: concatenate weights of w1 and 1 for bias
    % vcat <word2vec_dim + 1>
    % Wsem <h, word2vec_dim + 1>
    vlin = Wsem * vcat; % mul sus: multiply
    % 
    v = params.f(vlin); % nonlin sus: apply nonlinearity. negative elements become zero if relu is used for non linearity function
    Z(:,n) = v;

    n=n+1;
end

Z = Z(:, 1:n-1); % crop. sus: This is because some times the above loop is not fully
% executed (e.g., pc(1)<=0). We don't need the extra columns full of zeros.
% sus: n was increased 1 too many in the for loop, so here we have n-1.
    
    
end
