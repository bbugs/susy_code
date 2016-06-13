function [oWe, Img, Sent, itos, stoi] = get_toy_data(params)

%% Create some random toy data data


%% Create toy random image struct
Img = cell(params.n_imgs, 1);

for i=1:params.n_imgs
    Img{i} = struct();
    Img{i}.codes = rand(params.n_region_per_img, params.cnn_dim);
end

%% Create toy random word2vec
oWe = randn(params.word2vec_dim, params.V);

%% Create toy random sentence struct

Sent = cell(1, params.n_sentences);
for i=1:params.n_sentences
    n_words = randi(params.max_sentence_length);
    Sent{i} = struct();
    Sent{i}.word_index = randi(params.V, 1, n_words);
end


%% Create itos (image to sentence indeces)
itos = cell(params.n_imgs, 1);
k = 1;
for i=1:params.n_imgs
    % make a vector of size 
    tmp = [1, params.n_sentence_per_img];
    for j=1:params.n_sentence_per_img
        tmp(j) = k;
        k = k + 1;
    end
    itos{i} = tmp;
end

%% Create stoi (sentence to image)

k = 1;
stoi = zeros(params.n_sentences,1);
for i=1:params.n_sentences
    stoi(i) = k;
    if mod(i,5) == 0
        k = k + 1;
    end
end


end
