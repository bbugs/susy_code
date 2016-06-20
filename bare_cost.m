%% Compute the cost

%%
useglobal = true;
maxaccum = true;
thrglobalscore = 0;
smoothnum = 0;  % in CVPR paper there is no smoothing
gmargin = 40;
gscale = 0.5;


%% Global Cost
% compute global objective
cost = 0;
if useglobal
    
    % forward scores in all regions
    SG = zeros(N,N);  % N: number of image-sentence pairs in batch
    SGN = zeros(N,N); % the number of values (for mean)
    accumsis = cell(N,N);
    for i=1:N
        for j=1:N
            % d: <n_regions_image_i, n_words_sentence_j>
            sim_img_i_sent_j = sim_region_word(region2pair_id == i, word2pair_id == j);  % slice the similiraty matrix to extract two things:
            % all regions corresponding to pair i and all words
            % corresponding to pair j. When i and j are equal, it's the
            % same img-sentence pair, therefore Karpathy wants these high.
            if thrglobalscore, sim_img_i_sent_j(sim_img_i_sent_j<0) = 0; end
            if maxaccum
                % sv: find the max in each column (highest scoring region for each word)
                % si: find the region (row) where the max happens. si <1, n_words_in_sentence_j>
                [sv, si] = max(sim_img_i_sent_j, [], 1); % score will be max (i.e. we're finding support of each fragment in image)
                % sv <1, n_words_in_sentence_j>
                
                %TODO: what i don't get is why he doesn't also compute 
                % [sv, si] = max(sim_img_i_sent_j, [], 1) to get support
                % for each region in the text, but ok for now.
                accumsis{i,j} = si; % remember switches for backprop %sus: remember the region for each word
                s = sum(sv);  % score of image-sentence pair
            else
                s = sum(sim_img_i_sent_j(:)); % score is sum
            end
            nnorm = size(sim_img_i_sent_j,2); % number of sent fragments
            nnorm = nnorm + smoothnum;
            s = s/nnorm;
            SG(i,j) = s;
            SGN(i,j) = nnorm;
        end
    end
end
%%
if useglobal
    % compute the cost
    gcost = 0;
    cdiffs = zeros(N,N);  %sus: column differences?
    rdiffs = zeros(N,N);  %sus: row differences?
    for i=1:N
        % i is the pivot. It should have higher score than col and row
        
        % col term
        cdiff = max(0, SG(:,i) - SG(i,i) + gmargin);
        cdiff(i) = 0; % nvm score with self  %sus: no margin for actual image-sentence pair.  This is the cost of SVM (only those not in correct class contribute to the cost)
        cdiffs(:, i) = cdiff; % useful in backprop
        
        % row term
        rdiff = max(0, SG(i,:) - SG(i,i) + gmargin);
        rdiff(i) = 0;
        rdiffs(i, :) = rdiff; % useful in backprop
        
        gcost = gcost + sum(cdiff) + sum(rdiff);
    end
    
    gcost = gscale * gcost;
    cost = cost + gcost;  %sus: scalar
end

%%
ltop = zeros(n_regions_in_batch, n_words_in_batch);

if useglobal
    % backprop global objective
    
    % backprop margin
    dsg = zeros(N,N);
    for i=1:N
        cd = cdiffs(:,i);
        rd = rdiffs(i,:);
        
        % col term backprop
        dsg(i,i) = dsg(i,i) - sum(cd > 0);
        dsg(:,i) = dsg(:,i) + (cd > 0);
        
        % row term backprop
        dsg(i,i) = dsg(i,i) - sum(rd > 0);
        dsg(i,:) = dsg(i,:) + (rd > 0);
    end
    
    % backprop into scores
    ltopg = zeros(size(ltop));
    for i=1:N
        for j=1:N
            
            d = sim_region_word(region2pair_id == i, word2pair_id == j);
            dd = ones(size(d)) * dsg(i,j) / SGN(i,j);
            if thrglobalscore
                dd(d<0) = 0;
            end
            ltopg(region2pair_id == i, word2pair_id == j) = dd;

        end
    end
    ltop = ltop + gscale * ltopg;
end

%%
% backprop into fragment vectors
allDeltasImg = allSentVecs * ltop';
allDeltasSent = allImgVecs * ltop;

% backprop image mapping
df_Wi2s = allDeltasImg * imgVecs;

if finetuneCNN
    % derivative wrt CNN data so that we can pass on gradient to RCNN
    df_CNN = allDeltasImg' * Wi2s;
end

% backprop sentence mapping
df_Wsem = BackwardSents(depTrees,params,oWe,Wsem,sentVecsCell,allDeltasSent);

cost_struct = struct();
cost_struct.raw_cost = cost;
cost_struct.reg_cost = params.regC/2 * sum(theta.^2);
cost_struct.cost = cost_struct.raw_cost + cost_struct.reg_cost;

%[grad,~] = param2stack(df_Wi2s, df_Wsem);
grad = [df_Wi2s(:); df_Wsem(:);]; % for speed hardcode param2stack
grad = grad + params.regC * theta; % regularization term gradient



