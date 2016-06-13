function [Z] = ForwardSent(sentence_batch,params,oWe,Wsem)

N = length(sentence_batch.word_index); %sus: number of words
Z = zeros(params.h, N); %sus: hxN. h is the size of the semantic space
triples = zeros(3, N);
n=1;

for i=1:N  %sus: iterate over each word in the sentence

    ix1 = sentence_batch.word_index(i);  %sus: id word
    if ix1==-1, continue; end

    w1 = oWe(:,ix1);  %sus: get vector of word

    vcat = [w1; 1]; % cat sus: concatenate weights of w1 and 1 for bias
    vlin = Wsem * vcat; % mul sus: multiply
    v = params.f(vlin); % nonlin sus: apply nonlinearity
    Z(:,n) = v;

    n=n+1;
end

Z = Z(:, 1:n-1); % crop. sus: This is because some times the above loop is not fully
% executed (e.g., pc(1)<=0). We don't need the extra columns full of zeros.
% sus: n was increased 1 too many in the for loop, so here we have n-1.
    
    
end
