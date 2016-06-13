%% 
clear all
%% Make toy data
params = struct();

params.n_imgs = 100;
params.n_sentence_per_img = 5;
params.n_sentences = params.n_sentence_per_img * params.n_imgs;
params.n_region_per_img = 4;

params.V = 100;  % size of vocabulary
params.word2vec_dim = 10;  % dimension of pretrianed word2vec
params.h = 5;  % dimension of multimodal space
params.cnn_dim = 16;  % CNN dimension
params.max_sentence_length = 12;  % to generate random sentences up to this number of words
params.batch_size = 20;

[oWe, Img, Sent, itos, stoi] = get_toy_data(params);

%% Init Weights
[Wi2s, Wsem] = init_weights(params);

%% Set up activation function and its derivative

params.f = @(x) (max(0,x));
params.df = @(z) (z>0);


%% stach weights into theta
[theta, decodeInfo] = param2stack(Wi2s, Wsem);
%%  Assemble image batches
n_imgs = length(Img);
n_sentences = length(Sent);

% sample an image batch
rand_perm_img_indices = randperm(n_imgs);
batch_img_indices = rand_perm_img_indices(1:params.batch_size);
img_batch = Img(batch_img_indices);

%% Assemble Sentence batches
batch_sentence_indices = zeros(params.batch_size, 1);  % 35 x 1
for q=1:params.batch_size
    sent_indices_in_img = itos{batch_img_indices(q)}; % sentence indeces for this image (the qth image)
    batch_sentence_indices(q) = sent_indices_in_img(randi(length(sent_indices_in_img), 1)); % take a random sentence
end
sent_batch = Sent(batch_sentence_indices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the cost for a random set of weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% forward prop all image fragments and arrange them into a single large matrix
N = length(sent_batch);  % number of sentences in the batch. Also corresponds to number of images in the batch and size of batch 
img_cnn_codes = cell(1, N);
imgVecICell = cell(1, N);
for i=1:N        
    img_cnn_codes{i} = img_batch{i}.codes;  % extract only the CNN codes of img_batch
end
region_img_cnn_codes = cat(1, img_cnn_codes{:}); % <total n_regions in batch, cnn_dim + 1>

% write region to img_id
for i=1:N    
    imgVecICell{i} = ones(size(img_cnn_codes{i},1), 1)*i;  % <1, batch_size>
    % imgVecICell {[1;1;1;1],[2;2;2;2],[3;3;3;3],[4;4;4;4]}
end
region2img_id = cat(1, imgVecICell{:});

% Project all regions in batch to multimodal space
projected_regions = Wi2s * region_img_cnn_codes';  % <h, number regions in batch>
n_regions_in_batch = size(projected_regions, 2);

%% forward prop all sentences and arrange them
sentVecsCell = cell(1, N);  % <1, number of sentences in batch>
sentVecICell = cell(1, N);
for i = 1:N
    [z] = ForwardSent(sent_batch{i},params,oWe,Wsem);
    sentVecsCell{i} = z;
    sentVecICell{i} = ones(size(z,2), 1)*i;
end
sentVecI = cat(1, sentVecICell{:});
allSentVecs = cat(2, sentVecsCell{:});
Ns = size(allSentVecs, 2);

% compute fragment scores
dots = projected_regions' * allSentVecs;  %TODO tomorrow figure this out: <n_regions, ??? number words or sentences???>


