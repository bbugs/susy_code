function [Wi2s, Wsem] = init_weights(params)

fanIn = params.word2vec_dim + 1; % +1 for bias
fanOut = params.h;
range = 1/sqrt(6*fanIn + fanOut);
Wsem = -range + (2*range).*rand(params.h,fanIn);

% image side mapping is just a linear transform
fanIn = params.cnn_dim;
fanOut = params.h ;
range = 1/sqrt(6*fanIn + fanOut);
Wi2s = -range + (2*range).*rand(fanOut,fanIn);


end