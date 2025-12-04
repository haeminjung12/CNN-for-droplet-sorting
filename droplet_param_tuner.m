function droplet_param_tuner(datasetDir, outputDir, search)
%DROPLET_PARAM_TUNER Quick hyperparameter search for droplet_trainer.
%   DROPLET_PARAM_TUNER(DATASETDIR, OUTPUTDIR, SEARCH) runs a lightweight
%   random search over imbalance-focused options (batch size, focal gamma,
%   label smoothing, mixup/cutmix, freeze epochs, and initial LR) using a
%   short validation-driven loop. Results are saved as tuning_results.csv and
%   best_opts.mat (variable: bestOpts) that can be passed to droplet_trainer
%   via opts.tunedOptsFile to prefill hyperparameters.
%
%   Defaults:
%     datasetDir: ./cropped_dataset
%     outputDir:  DATASETDIR/param_search/run_yyyymmdd_HHMMSS
%     search.numTrials:   5
%     search.maxEpochs:   5
%     search.batchSizes:  [32 64 96]
%     search.focalGamma:  [1.5 2.0 2.5]
%     search.labelSmooth: [0 0.05]
%     search.mixupAlpha:  [0 0.2]
%     search.cutmixAlpha: [0 0.2]
%     search.freezeEpochs:[0 2]
%     search.initLR:      [5e-4 1e-3]
%     search.evalCenterCrop: true
%     search.lrDropPatience: 2 (on macro-F1)
%
%   A brief comment block in droplet_trainer documents the new opts.tunedOptsFile
%   hook used to load these saved settings.

if nargin < 1 || isempty(datasetDir)
    datasetDir = fullfile(pwd, 'cropped_dataset');
end
if nargin < 2 || isempty(outputDir)
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    outputDir = fullfile(datasetDir, 'param_search', ['run_' stamp]);
end
if nargin < 3 || isempty(search)
    search = struct();
end

searchDefaults = struct( ...
    'numTrials', 5, ...
    'maxEpochs', 5, ...
    'batchSizes', [32 64 96], ...
    'focalGamma', [1.5 2.0 2.5], ...
    'labelSmooth', [0 0.05], ...
    'mixupAlpha', [0 0.2], ...
    'cutmixAlpha', [0 0.2], ...
    'freezeEpochs', [0 2], ...
    'initLR', [5e-4 1e-3], ...
    'evalCenterCrop', true, ...
    'lrDropPatience', 2, ...
    'stratifyGroups', '');

search = merge_opts(searchDefaults, search);

if ~isfolder(outputDir)
    mkdir(outputDir);
end

imds = imageDatastore(datasetDir, 'IncludeSubfolders', true, 'LabelSource','foldernames');
if isempty(imds.Files)
    error('No images found in %s', datasetDir);
end
classes = categories(imds.Labels);

fileGroups = containers.Map();
if ~isempty(search.stratifyGroups) && isfile(search.stratifyGroups)
    T = readtable(search.stratifyGroups, 'TextType','string');
    if all(ismember({'File','Group'}, T.Properties.VariableNames))
        for i = 1:height(T)
            fileGroups(char(T.File(i))) = char(T.Group(i));
        end
    else
        warning('Group file missing File/Group columns. Ignoring groups.');
    end
end

[imdsTrain, imdsVal] = quick_split(imds, fileGroups);
weightsTrain = class_weights_per_sample(imdsTrain.Labels, classes);
[classWeights, focalAlpha] = compute_class_weights(imdsTrain.Labels, classes);
minority = find_minority_classes(imdsTrain.Labels);

results = [];
logPath = fullfile(outputDir, 'tuning_log.txt');
fid = fopen(logPath, 'w');
fprintf(fid, 'Droplet param tuner %s\n', datestr(now));
fprintf(fid, 'Dataset: %s\n', datasetDir);

for trial = 1:search.numTrials
    cfg.batchSize    = pick(search.batchSizes);
    cfg.focalGamma   = pick(search.focalGamma);
    cfg.labelSmooth  = pick(search.labelSmooth);
    cfg.mixupAlpha   = pick(search.mixupAlpha);
    cfg.cutmixAlpha  = pick(search.cutmixAlpha);
    cfg.freezeEpochs = pick(search.freezeEpochs);
    cfg.initialLearnRate = pick(search.initLR);
    cfg.evalCenterCrop = search.evalCenterCrop;
    cfg.lrDropPatience = search.lrDropPatience;
    cfg.focalLoss = true;
    cfg.labelSmoothing = cfg.labelSmooth; %#ok<STRNU>

    fprintf('Trial %d/%d: batch %d, gamma %.2f, smooth %.3f, mixup %.2f, cutmix %.2f, freeze %d, lr %.4g\n', ...
        trial, search.numTrials, cfg.batchSize, cfg.focalGamma, cfg.labelSmooth, cfg.mixupAlpha, cfg.cutmixAlpha, cfg.freezeEpochs, cfg.initialLearnRate);
    fprintf(fid, 'Trial %d: batch %d, gamma %.2f, smooth %.3f, mixup %.2f, cutmix %.2f, freeze %d, lr %.4g\n', ...
        trial, cfg.batchSize, cfg.focalGamma, cfg.labelSmooth, cfg.mixupAlpha, cfg.cutmixAlpha, cfg.freezeEpochs, cfg.initialLearnRate);

    augTrain = @(im,label) augment_image(im, label, [64 64 3], minority);
    prepEval = @(im) preprocess_eval(im, [64 64 3], cfg.evalCenterCrop);

    lgraph = droplet_squeezenet_64(numel(classes));
    lgraph = strip_output_layer(lgraph);
    net = dlnetwork(lgraph);
    learnables = net.Learnables;
    backboneMask = contains(learnables.Layer, "fire") | contains(learnables.Layer, "conv1") | contains(learnables.Layer, "pool");
    trainMask = ~backboneMask;

    bestF1 = -inf; lr = cfg.initialLearnRate; velocity = []; stalled = 0;
    for epoch = 1:search.maxEpochs
        if epoch > cfg.freezeEpochs
            trainMask = true(size(trainMask));
        end
        [net, velocity] = train_one_epoch(net, imdsTrain, weightsTrain, classes, classWeights, focalAlpha, augTrain, cfg, lr, 0.9, trainMask, velocity);
        [valLoss, valAcc, valMetrics] = evaluate_epoch(net, imdsVal, classes, prepEval, cfg, classWeights, focalAlpha);
        if valMetrics.macroF1 > bestF1
            bestF1 = valMetrics.macroF1; stalled = 0;
        else
            stalled = stalled + 1;
        end
        if exist('stalled','var') && stalled >= cfg.lrDropPatience
            lr = lr * 0.5;
            stalled = 0;
        end
    end

    results = [results; struct('Trial', trial, 'Batch', cfg.batchSize, 'Gamma', cfg.focalGamma, ...
        'Smooth', cfg.labelSmooth, 'Mixup', cfg.mixupAlpha, 'Cutmix', cfg.cutmixAlpha, ...
        'Freeze', cfg.freezeEpochs, 'LR', cfg.initialLearnRate, 'ValMacroF1', bestF1, ...
        'ValAcc', valAcc, 'ValLoss', valLoss)]; %#ok<AGROW>
end

fclose(fid);

% pick best
[~, bestIdx] = max([results.ValMacroF1]);
best = results(bestIdx);
bestOpts = struct( ...
    'batchSize', best.Batch, ...
    'focalGamma', best.Gamma, ...
    'labelSmoothing', best.Smooth, ...
    'mixupAlpha', best.Mixup, ...
    'cutmixAlpha', best.Cutmix, ...
    'freezeEpochs', best.Freeze, ...
    'initialLearnRate', best.LR, ...
    'focalLoss', true, ...
    'evalCenterCrop', search.evalCenterCrop, ...
    'lrDropPatience', search.lrDropPatience);

resultsTbl = struct2table(results);
resPath = fullfile(outputDir, 'tuning_results.csv');
writetable(resultsTbl, resPath);
save(fullfile(outputDir, 'best_opts.mat'), 'bestOpts');

fprintf('Best trial %d: macro-F1 %.3f (saved to %s)\n', best.Trial, best.ValMacroF1, resPath);

end

%% --- helpers ---
function val = pick(arr)
    val = arr(randi(numel(arr)));
end

function [trainDS, valDS] = quick_split(imdsAll, fileGroups)
    labels = imdsAll.Labels;
    idx = (1:numel(labels))';
    if ~isempty(fileGroups)
        groups = repmat("", numel(imdsAll.Files),1);
        for k = 1:numel(imdsAll.Files)
            f = char(imdsAll.Files{k});
            [~, name, ext] = fileparts(f);
            key = [name ext];
            if isKey(fileGroups, key)
                groups(k) = string(fileGroups(key));
            end
        end
        if any(groups ~= "")
            [trainIdx, valIdx, ~] = stratify_by_group(idx, labels, groups);
        else
            [trainIdx, valIdx, ~] = stratify_by_label(idx, labels);
        end
    else
        [trainIdx, valIdx, ~] = stratify_by_label(idx, labels);
    end
    trainDS = subset(imdsAll, trainIdx);
    valDS   = subset(imdsAll, valIdx);
end

function [trainIdx, valIdx, testIdx] = stratify_by_label(idx, labels)
    rng('shuffle');
    trainIdx = [];
    valIdx = [];
    testIdx = [];
    cats = categories(labels);
    for i = 1:numel(cats)
        mask = labels == cats{i};
        clsIdx = idx(mask);
        clsIdx = clsIdx(randperm(numel(clsIdx)));
        n = numel(clsIdx);
        nTrain = round(0.7 * n);
        nVal   = round(0.15 * n);
        trainIdx = [trainIdx; clsIdx(1:nTrain)]; %#ok<AGROW>
        valIdx   = [valIdx;   clsIdx(nTrain+1:nTrain+nVal)]; %#ok<AGROW>
        testIdx  = [testIdx;  clsIdx(nTrain+nVal+1:end)]; %#ok<AGROW>
    end
end

function [trainIdx, valIdx, testIdx] = stratify_by_group(idx, labels, groups)
    trainIdx = [];
    valIdx = [];
    testIdx = [];
    cats = categories(labels);
    for i = 1:numel(cats)
        mask = labels == cats{i};
        clsIdx = idx(mask);
        clsGroups = groups(mask);
        [uGroups,~,gIdx] = unique(clsGroups);
        perm = randperm(numel(uGroups));
        uGroups = uGroups(perm);
        nG = numel(uGroups);
        nTrain = round(0.7 * nG);
        nVal   = round(0.15 * nG);
        trainGroups = uGroups(1:nTrain);
        valGroups   = uGroups(nTrain+1:nTrain+nVal);
        testGroups  = uGroups(nTrain+nVal+1:end);
        trainMask = ismember(clsGroups, trainGroups);
        valMask   = ismember(clsGroups, valGroups);
        testMask  = ismember(clsGroups, testGroups);
        trainIdx = [trainIdx; clsIdx(trainMask)]; %#ok<AGROW>
        valIdx   = [valIdx;   clsIdx(valMask)]; %#ok<AGROW>
        testIdx  = [testIdx;  clsIdx(testMask)]; %#ok<AGROW>
    end
end

function w = class_weights_per_sample(labels, classesLocal)
    counts = countcats(labels);
    freq = counts / sum(counts);
    invFreq = 1 ./ max(freq, 1e-6);
    invFreq = min(invFreq, 100);
    w = zeros(numel(labels),1);
    for i = 1:numel(classesLocal)
        mask = labels == classesLocal{i};
        w(mask) = invFreq(i);
    end
end

function [classWeights, focalAlpha] = compute_class_weights(trainLabels, classesLocal)
    counts = countcats(trainLabels);
    freq = counts / sum(counts);
    invFreq = 1 ./ max(freq, 1e-6);
    invFreq = min(invFreq, 100); % clamp extreme
    classWeights = invFreq / mean(invFreq);
    focalAlpha = classWeights(:) / sum(classWeights);
end

function minority = find_minority_classes(trainLabels)
    counts = countcats(trainLabels);
    cats = categories(trainLabels);
    medCnt = median(counts);
    minority = cats(counts < medCnt);
end

function data = augment_image(im, label, outSize, minorityClasses)
    im = im2single(im);
    if size(im,3)==1, im = repmat(im,1,1,3); end
    scale = 0.9 + 0.2*rand();
    cropSize = round(outSize(1:2) * scale);
    [h,w,~] = size(im);
    y = max(1, randi([1, max(1, h - cropSize(1)+1)]));
    x = max(1, randi([1, max(1, w - cropSize(2)+1)]));
    yEnd = min(h, y+cropSize(1)-1);
    xEnd = min(w, x+cropSize(2)-1);
    im = imresize(im(y:yEnd, x:xEnd, :), outSize(1:2));
    if rand()<0.5, im = fliplr(im); end
    if rand()<0.2, im = flipud(im); end
    angle = -10 + 20*rand();
    im = imrotate(im, angle, 'bilinear','crop');
    im = imadjust(im, [], [], 0.9 + 0.2*rand());
    im = im + 0.02*randn(size(im));
    im = imgaussfilt(im, 0.5*rand());
    if any(strcmp(label, minorityClasses))
        if rand()<0.5, im = imbilatfilt(im, 0.5, 2); end
        if rand()<0.5, im = apply_clahe_per_channel(im); end
        if rand()<0.3, im = im + randn(size(im))*0.03; end
    end
    data = im;
end

function out = apply_clahe_per_channel(im)
    if ndims(im) == 2 || size(im,3) == 1
        out = adapthisteq(im, 'ClipLimit',0.01);
    else
        out = im;
        for c = 1:size(im,3)
            out(:,:,c) = adapthisteq(im(:,:,c), 'ClipLimit',0.01);
        end
    end
end

function data = preprocess_eval(im, outSize, doCenterCrop)
    im = im2single(im);
    if size(im,3)==1, im = repmat(im,1,1,3); end
    if doCenterCrop
        [h,w,~] = size(im);
        minSide = min(h,w);
        y = floor((h - minSide)/2)+1;
        x = floor((w - minSide)/2)+1;
        im = im(y:y+minSide-1, x:x+minSide-1, :);
    end
    data = imresize(im, outSize(1:2));
end

function [dlX, dlY] = batch_reader(im, labels, augFcn, doAug)
    if nargin < 4, doAug = true; end
    labelsCat = categorical(labels);
    labelsCat = labelsCat(:); % ensure column vector
    labelsCell = cellstr(labelsCat); % ensure cell array for cellfun
    if doAug
        imgs = cellfun(@(a,b) augFcn(a,b), im, labelsCell, 'UniformOutput', false);
    else
        imgs = cellfun(@(a) augFcn(a), im, 'UniformOutput', false);
    end
    X = cat(4, imgs{:});
    X = dlarray(X, 'SSCB');
    % One-hot encode to classes x batch for stable mixup/cutmix indexing
    Y = onehotencode(labelsCat, 2)';
    dlY = dlarray(single(Y), 'CB');
    dlX = X;
end

function [net, velocity] = train_one_epoch(net, imdsTrain, weights, classesLocal, classWeights, focalAlpha, augFcn, opts, lr, momentum, trainMask, velocity)
    numIters = floor(numel(imdsTrain.Files)/opts.batchSize);
    if isempty(weights), weights = ones(numel(imdsTrain.Files),1); end
    if nargin < 12 || isempty(velocity)
        velocity = [];
    end
    for step = 1:numIters
        idx = randsample(numel(imdsTrain.Files), opts.batchSize, true, weights);
        ims = cell(opts.batchSize,1);
        labs = imdsTrain.Labels(idx);
        for j = 1:opts.batchSize
            ims{j} = imread(imdsTrain.Files{idx(j)});
        end
        [dlX, dlY] = batch_reader(ims, labs, augFcn, true);
        if opts.mixupAlpha>0 || opts.cutmixAlpha>0
            [dlX, dlY] = apply_mix(dlX, dlY, opts.mixupAlpha, opts.cutmixAlpha);
        end
        [gradients, loss] = dlfeval(@modelLoss, net, dlX, dlY, classWeights, focalAlpha, opts); %#ok<NASGU>
        % Initialize or repair the velocity table so it matches gradient structure
        if isempty(velocity) || ~istable(velocity)
            velocity = dlupdate(@(x) zeros(size(x),'like',x), gradients);
        end
        gradCells = gradients.Value;
        for gi = 1:numel(gradCells)
            if ~trainMask(gi)
                gradCells{gi} = zeros(size(gradCells{gi}), 'like', gradCells{gi});
            end
        end
        gradients.Value = gradCells;
        [net, velocity] = sgdmupdate(net, gradients, velocity, lr, momentum);
    end
end

function [lossVal, accVal, metrics] = evaluate_epoch(net, imdsVal, classesLocal, prepFcn, opts, classWeights, focalAlpha)
    numIters = ceil(numel(imdsVal.Files)/opts.batchSize);
    losses = zeros(numIters,1); accs = zeros(numIters,1);
    allPred = []; allTrue = [];
    for step = 1:numIters
        idx = (step-1)*opts.batchSize+1 : min(step*opts.batchSize, numel(imdsVal.Files));
        ims = cell(numel(idx),1); labs = imdsVal.Labels(idx);
        for j = 1:numel(idx)
            ims{j} = imread(imdsVal.Files{idx(j)});
        end
        [dlX, dlY] = batch_reader(ims, labs, @(im) prepFcn(im), false);
        [loss, acc] = evaluate_loss(net, dlX, dlY, classWeights, focalAlpha, opts); %#ok<NASGU>
        losses(step) = loss; accs(step) = acc;
        probs = softmax(forward(net, dlX));
        [~, predIdx] = max(probs, [], 1);
        allPred = [allPred; gather(extractdata(predIdx'))]; %#ok<AGROW>
        [~, trueIdx] = max(dlY, [], 1);
        allTrue = [allTrue; gather(extractdata(trueIdx'))]; %#ok<AGROW>
    end
    lossVal = mean(losses);
    accVal  = mean(accs);
    predLabels = categorical(classesLocal(allPred));
    trueLabels = categorical(classesLocal(allTrue));
    [perClass, macroF1, balAcc] = compute_metrics(trueLabels, predLabels, classesLocal);
    metrics.perClass = perClass;
    metrics.macroF1 = macroF1;
    metrics.balancedAccuracy = balAcc;
end

function [loss, acc] = evaluate_loss(net, dlX, dlY, classWeights, focalAlpha, opts)
    if isempty(classWeights)
        classWeights = ones(size(dlY,1),1);
    end
    if isempty(focalAlpha)
        focalAlpha = ones(size(dlY,1),1)/size(dlY,1);
    end
    [~, loss, acc] = modelLoss(net, dlX, dlY, classWeights, focalAlpha, opts, false);
end

function [gradients, loss, acc] = modelLoss(net, dlX, dlY, classWeights, focalAlpha, opts, isTraining)
    if nargin <7, isTraining = true; end
    scores = forward(net, dlX);
    scores = scores + eps;
    if opts.labelSmoothing > 0
        nCls = size(scores,1);
        smooth = opts.labelSmoothing;
        dlY = dlY*(1-smooth) + smooth/nCls;
    end
    probs = softmax(scores);
    nCls = size(probs,1);
    if opts.focalLoss
        alphaVec = dlarray(cast(focalAlpha(:), 'like', probs));
        if isempty(alphaVec)
            alphaVec = ones(nCls,1,'like',probs)/nCls;
        elseif numel(alphaVec)==1
            alphaVec = repmat(alphaVec, nCls, 1);
        elseif numel(alphaVec)~=nCls
            alphaVec = alphaVec(1:min(end,nCls));
            if numel(alphaVec) < nCls
                alphaVec(end+1:nCls) = alphaVec(end);
            end
        end
        alpha = reshape(alphaVec, [nCls, ones(1, ndims(probs)-1)]);
        gamma = opts.focalGamma;
        ce = -dlY .* log(probs);
        weight = alpha .* (1 - probs).^gamma;
        loss = sum(ce .* weight, 1);
    else
        cwVec = dlarray(cast(classWeights(:), 'like', probs));
        if isempty(cwVec)
            cwVec = ones(nCls,1,'like',probs);
        elseif numel(cwVec)==1
            cwVec = repmat(cwVec, nCls, 1);
        elseif numel(cwVec)~=nCls
            cwVec = cwVec(1:min(end,nCls));
            if numel(cwVec) < nCls
                cwVec(end+1:nCls) = cwVec(end);
            end
        end
        cw = reshape(cwVec, [nCls, ones(1, ndims(probs)-1)]);
        loss = -sum(dlY .* log(probs) .* cw, 1);
    end
    loss = mean(loss, 'all');
    [~, predIdx] = max(probs, [], 1);
    [~, trueIdx] = max(dlY, [], 1);
    acc = mean(gather(extractdata(predIdx == trueIdx)));
    gradients = dlgradient(loss, net.Learnables, 'RetainData', isTraining);
end

function [dlX, dlY] = apply_mix(dlX, dlY, mixupAlpha, cutmixAlpha)
    if mixupAlpha > 0
        alpha = max(eps(single(1)), single(mixupAlpha));
        lambda = single(betarnd(alpha, alpha, [1 1]));
        perm = randperm(size(dlX,4));
        dlX = lambda * dlX + (1-lambda) * dlX(:,:,:,perm);
        dlY = lambda * dlY + (1-lambda) * dlY(:,perm);
    elseif cutmixAlpha > 0
        alpha = max(eps(single(1)), single(cutmixAlpha));
        lambda = single(betarnd(alpha, alpha, [1 1]));
        perm = randperm(size(dlX,4));
        [h,w,~,~] = size(dlX);
        rx = randi([1 w]); ry = randi([1 h]);
        rw = floor(w * sqrt(1-lambda)); rh = floor(h * sqrt(1-lambda));
        x1 = max(1, rx - floor(rw/2)); x2 = min(w, rx + floor(rw/2));
        y1 = max(1, ry - floor(rh/2)); y2 = min(h, ry + floor(rh/2));
        dlX(y1:y2, x1:x2, :, :) = dlX(y1:y2, x1:x2, :, perm);
        lambda = 1 - ((x2-x1+1)*(y2-y1+1)/(w*h));
        dlY = lambda * dlY + (1-lambda) * dlY(:,perm);
    end
end

function [perClass, macroF1, balAcc] = compute_metrics(trueLabels, predLabels, classesLocal)
    conf = confusionmat(trueLabels, predLabels, 'Order', categorical(classesLocal));
    tp = diag(conf);
    fp = sum(conf,1)' - tp;
    fn = sum(conf,2) - tp;
    precision = tp ./ max(tp+fp, 1);
    recall    = tp ./ max(tp+fn, 1);
    f1        = 2*precision.*recall ./ max(precision+recall, eps);
    perClass = table(classesLocal, precision, recall, f1, 'VariableNames', {'Class','Precision','Recall','F1'});
    macroF1 = mean(f1);
    sens = tp ./ max(tp+fn,1);
    spec = (sum(conf,'all') - fp - fn - tp) ./ max(sum(conf,'all') - tp + eps,1);
    balAcc = mean([sens; spec]);
end

function merged = merge_opts(varargin)
    merged = struct();
    for k = 1:nargin
        src = varargin{k};
        if isempty(src), continue; end
        f = fieldnames(src);
        for i = 1:numel(f)
            v = src.(f{i});
            if ~isfield(merged, f{i}) || ~isempty(v) || isempty(merged.(f{i}))
                merged.(f{i}) = v;
            end
        end
    end
end

function lgraph = strip_output_layer(lgraph)
    % Remove classification output layer so dlnetwork can be built for custom training.
    if any(strcmp({lgraph.Layers.Name}, 'classoutput_droplet'))
        lgraph = removeLayers(lgraph, 'classoutput_droplet');
    end
end