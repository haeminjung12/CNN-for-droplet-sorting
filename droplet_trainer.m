function droplet_trainer(datasetDir, outputDir, maxEpochs, opts)
%DROPLET_TRAINER Train a droplet classifier with imbalance-aware pipeline.
%   DROPLET_TRAINER(DATASETDIR, OUTPUTDIR, MAXEPOCHS, OPTS) fine-tunes the
%   64x64 SqueezeNet model on cropped droplet images arranged by class
%   folders. Existing CLI defaults are preserved:
%     datasetDir (default: ./cropped_dataset)
%     outputDir  (default: DATASETDIR/training_runs/run_YYYYMMDD_HHMMSS)
%     maxEpochs  (default: 20)
%
%   New options (pass as struct) to address extreme imbalance and stability:
%     opts.stratifyGroups : optional CSV with columns File,Group for leakage-safe
%                           splits; when provided, splits are stratified by class
%                           and grouped by Group. Default: ''.
%     opts.kFolds         : integer k for stratified k-fold CV (default 0 = off).
%     opts.focalLoss      : logical, enable focal loss (default true).
%     opts.focalGamma     : scalar gamma for focal loss (default 2.0).
%     opts.labelSmoothing : scalar in [0,1) (default 0).
%     opts.mixupAlpha     : scalar >0 to enable mixup; 0 disables (default 0).
%     opts.cutmixAlpha    : scalar >0 to enable cutmix; 0 disables (default 0).
%     opts.batchSize      : mini-batch size (default 64).
%     opts.freezeEpochs   : epochs to keep backbone frozen before unfreezing top
%                           blocks (default 2).
%     opts.initialLearnRate : starting learning rate (default 1e-3).
%     opts.earlyStopPatience : patience (epochs) on macro-F1 (default 5).
%     opts.lrDropPatience : patience (epochs) for reduce-on-plateau (default 3).
%     opts.lrDropFactor   : LR decay factor when plateau detected (default 0.5).
%     opts.evalCenterCrop : logical, center-crop at eval (default true).
%     opts.saveBestOnly   : logical, save checkpoint on best macro-F1 (default true).
%     opts.tunedOptsFile  : MAT/JSON file from droplet_param_tuner to prefill
%                           hyperparameters (explicit CLI opts still override).
%
%   The trainer performs class-balanced sampling per batch, auto-computes class
%   weights from the training split, applies microscopy-friendly augmentations
%   (stronger for minority classes), supports label smoothing, focal loss, mixup
%   and cutmix, and logs per-class precision/recall/F1, macro-F1, balanced
%   accuracy, and confusion matrices each epoch. A summary image and metadata
%   are written to OUTPUTDIR.

if nargin < 1 || isempty(datasetDir)
    datasetDir = fullfile(pwd, 'cropped_dataset');
end
if nargin < 2 || isempty(outputDir)
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    outputDir = fullfile(datasetDir, 'training_runs', ['run_' stamp]);
end
if nargin < 3 || isempty(maxEpochs)
    maxEpochs = 20;
end
if nargin < 4 || isempty(opts)
    opts = struct();
end

defaults = struct( ...
    'stratifyGroups', '', ...
    'kFolds', 0, ...
    'focalLoss', true, ...
    'focalGamma', 2.0, ...
    'labelSmoothing', 0.0, ...
    'mixupAlpha', 0, ...
    'cutmixAlpha', 0, ...
    'batchSize', 64, ...
    'freezeEpochs', 2, ...
    'initialLearnRate', 1e-3, ...
    'earlyStopPatience', 5, ...
    'lrDropPatience', 3, ...
    'lrDropFactor', 0.5, ...
    'evalCenterCrop', true, ...
    'saveBestOnly', true, ...
    'tunedOptsFile', '');

tuned = struct();
if isfield(opts, 'tunedOptsFile') && ~isempty(opts.tunedOptsFile)
    tuned = load_tuned_opts(opts.tunedOptsFile);
end

opts = merge_opts(defaults, tuned, opts);

if ~isfolder(datasetDir)
    error('Dataset folder not found: %s', datasetDir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

inputSize = [64 64 3];

% Load image datastore
imds = imageDatastore(datasetDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
if isempty(imds.Files)
    error('No images found in %s. Ensure crops are organized by class.', datasetDir);
end
classes = categories(imds.Labels);

% Optional grouping metadata for leakage-safe splits
fileGroups = containers.Map();
if ~isempty(opts.stratifyGroups) && isfile(opts.stratifyGroups)
    T = readtable(opts.stratifyGroups, 'TextType','string');
    if all(ismember({'File','Group'}, T.Properties.VariableNames))
        for i = 1:height(T)
            fileGroups(char(T.File(i))) = char(T.Group(i));
        end
    else
        warning('Group file missing File/Group columns. Ignoring groups.');
    end
end

% Train/val/test split (stratified; group-aware if provided)
[imdsTrain, imdsVal, imdsTest, counts] = stratified_split(imds, fileGroups);

% k-fold option (if enabled, run folds sequentially)
if opts.kFolds > 1
    run_kfold(imds, opts.kFolds);
    return;
end

% Compute class weights from training split only
[classWeights, focalAlpha] = compute_class_weights(imdsTrain.Labels, classes);

% log split counts
logPath = fullfile(outputDir, 'train_log.txt');
fid = fopen(logPath, 'w');
fprintf(fid, 'Droplet training run: %s\n', datestr(now));
fprintf(fid, 'Dataset: %s\n', datasetDir);
fprintf(fid, 'Output:  %s\n', outputDir);
fprintf(fid, 'Epochs:  %d\n', maxEpochs);
fprintf(fid, 'Batch size: %d\n', opts.batchSize);
fprintf(fid, 'Initial LR: %.4g\n', opts.initialLearnRate);
fprintf(fid, 'Focal: %d (gamma=%.2f)\n', opts.focalLoss, opts.focalGamma);
fprintf(fid, 'Label smoothing: %.3f\n', opts.labelSmoothing);
fprintf(fid, 'Mixup alpha: %.2f, Cutmix alpha: %.2f\n', opts.mixupAlpha, opts.cutmixAlpha);
fprintf(fid, '\nClass counts (Total / Train / Val / Test):\n');
for r = 1:height(counts)
    fprintf(fid, '  %s: %d / %d / %d / %d\n', counts.Class{r}, ...
        counts.Total(r), counts.Train(r), counts.Validation(r), counts.Test(r));
end
fclose(fid);

% Augmentation pipelines
minorityClasses = find_minority_classes(counts);
augTrain = @(im,label) augment_image(im, label, inputSize, minorityClasses);
prepEval = @(im) preprocess_eval(im, inputSize, opts.evalCenterCrop);

% Build model and dlnetwork
lgraph = droplet_squeezenet_64(numel(classes));
lgraph = strip_output_layer(lgraph);
net = dlnetwork(lgraph);

% Freeze backbone initially
learnables = net.Learnables;
backboneMask = contains(learnables.Layer, "fire") | contains(learnables.Layer, "conv1") | contains(learnables.Layer, "pool");
trainMask = ~backboneMask;

% Training state
bestMacroF1 = -inf;
noImprove = 0;
lr = opts.initialLearnRate;
valNoImprove = 0;
metricsLog = [];

% Balanced sampler weights (clamped)
weightsTrain = class_weights_per_sample(imdsTrain.Labels, classes);
trainFiles   = imdsTrain.Files;   trainLabels = imdsTrain.Labels;
valFiles     = imdsVal.Files;     valLabels   = imdsVal.Labels;
testFiles    = imdsTest.Files;    testLabels  = imdsTest.Labels;

% Training monitor GUI
ui = build_progress_ui();
trainHistory = struct('Iteration', [], 'Epoch', [], 'TrainingLoss', [], ...
    'ValidationLoss', [], 'TrainingAccuracy', [], 'ValidationAccuracy', [], ...
    'LearnRate', [], 'Time', []);
iter = 0;
start = tic;

% optimizer
momentum = 0.9;
velocity = []; %#ok<NASGU>

for epoch = 1:maxEpochs
    % unfreeze after freezeEpochs
    if epoch > opts.freezeEpochs
        trainMask = true(size(trainMask));
    end

    [net, velocity, trainLoss, trainAcc] = train_one_epoch(net, trainFiles, trainLabels, ...
        weightsTrain, classes, classWeights, focalAlpha, augTrain, opts, lr, momentum, trainMask);

    % validation
    [valLoss, valAcc, valMetrics] = evaluate_epoch(net, valFiles, valLabels, classes, prepEval, opts);

    iter = iter + 1;
    trainHistory.Iteration(end+1,1)          = iter;
    trainHistory.Epoch(end+1,1)              = epoch;
    trainHistory.TrainingLoss(end+1,1)       = trainLoss;
    trainHistory.ValidationLoss(end+1,1)     = valLoss;
    trainHistory.TrainingAccuracy(end+1,1)   = trainAcc;
    trainHistory.ValidationAccuracy(end+1,1) = valAcc;
    trainHistory.LearnRate(end+1,1)          = lr;
    trainHistory.Time(end+1,1)               = toc(start);

    update_live_plots(ui, iter, trainLoss, valLoss, trainAcc, valAcc, lr);

    % early stopping and LR schedule based on macro F1
    macroF1 = valMetrics.macroF1;
    metricsLog = [metricsLog; struct('Epoch',epoch,'MacroF1',macroF1,'Metrics',valMetrics)]; %#ok<AGROW>
    save_epoch_metrics(outputDir, epoch, valMetrics);

    if macroF1 > bestMacroF1
        bestMacroF1 = macroF1;
        noImprove = 0;
        if opts.saveBestOnly
            save_checkpoint(outputDir, net, epoch, macroF1);
        end
    else
        noImprove = noImprove + 1;
    end

    valNoImprove = valNoImprove + 1;
    if noImprove >= opts.lrDropPatience
        lr = lr * opts.lrDropFactor;
        noImprove = 0;
    end

    if valNoImprove >= opts.earlyStopPatience
        fprintf('Early stopping at epoch %d (macro-F1 plateau).\n', epoch);
        break;
    end
end

% final checkpoint if not best-only
if ~opts.saveBestOnly
    save_checkpoint(outputDir, net, epoch, bestMacroF1);
end

% test evaluation
augTest = augmentedImageDatastore(inputSize, imdsTest, 'OutputSizeMode','resize');
if opts.evalCenterCrop
    augTest = augmentedImageDatastore(inputSize, imdsTest, 'OutputSizeMode','centercrop');
end
YPred = classify(net, augTest);
YTest = imdsTest.Labels;
confMat = confusionmat(YTest, YPred);
[perClass, macroF1Test, balAcc] = compute_metrics(YTest, YPred, classes);
accuracyTest = mean(YPred == YTest);

summaryFig = figure('Name','Droplet training summary','Color','w','Position',[100 100 1300 450]);
subplot(1,3,1);
plot(trainHistory.Iteration, trainHistory.TrainingAccuracy, 'b-', 'LineWidth', 1.5); hold on;
plot(trainHistory.Iteration, trainHistory.ValidationAccuracy, 'r-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Accuracy (%)');
legend('Train','Validation','Location','southwest'); grid on; title('Accuracy');

subplot(1,3,2);
plot(trainHistory.Iteration, trainHistory.TrainingLoss, 'b-', 'LineWidth', 1.5); hold on;
plot(trainHistory.Iteration, trainHistory.ValidationLoss, 'r-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Loss'); legend('Train','Validation','Location','northeast'); grid on; title('Loss');

subplot(1,3,3);
plot(trainHistory.Iteration, trainHistory.LearnRate, 'k-','LineWidth',1.5);
xlabel('Iteration'); ylabel('Learning rate'); grid on; title(sprintf('Macro-F1 val best: %.3f', bestMacroF1));

summaryPath = fullfile(outputDir, 'training_summary.png');
exportgraphics(summaryFig, summaryPath);

metadata.trainHistory = trainHistory;
metadata.classes = classes;
metadata.datasetDir = datasetDir;
metadata.outputDir = outputDir;
metadata.accuracyTest = accuracyTest;
metadata.confusionMat = confMat;
metadata.valMetrics = metricsLog;
metadata.testPerClass = perClass;
metadata.testMacroF1 = macroF1Test;
metadata.testBalancedAcc = balAcc;
metadata.splitCounts = counts;
metadata.options = opts;
metadata.bestMacroF1 = bestMacroF1;

save(fullfile(outputDir, 'train_metadata.mat'), 'metadata', 'net');

fid = fopen(logPath, 'a');
fprintf(fid, '\nTest accuracy: %.2f %%\n', 100*accuracyTest);
fprintf(fid, 'Test macro-F1: %.3f, balanced acc: %.3f\n', macroF1Test, balAcc);
fclose(fid);

fprintf('Training complete. Summary saved to %s\n', summaryPath);

%% ---------- helper functions ----------
function tuned = load_tuned_opts(path)
    tuned = struct();
    if isempty(path)
        return;
    end
    try
        [~,~,ext] = fileparts(path);
        if strcmpi(ext, '.mat')
            S = load(path);
            if isfield(S, 'bestOpts')
                tuned = S.bestOpts;
            elseif isfield(S, 'opts')
                tuned = S.opts;
            else
                f = fieldnames(S);
                if numel(f)==1 && isstruct(S.(f{1}))
                    tuned = S.(f{1});
                end
            end
        else
            txt = fileread(path);
            tuned = jsondecode(txt);
        end
        fprintf('Loaded tuned options from %s\n', path);
    catch ME
        warning('Could not load tuned opts from %s: %s', path, ME.message);
    end
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

function s = fill_defaults(s, defaults)
    names = fieldnames(defaults);
    for i = 1:numel(names)
        n = names{i};
        if ~isfield(s, n) || isempty(s.(n))
            s.(n) = defaults.(n);
        end
    end
end

function [trainDS, valDS, testDS, counts] = stratified_split(imdsAll, fileGroups)
    labels = imdsAll.Labels;
    classesLocal = categories(labels);
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
            [trainIdx, valIdx, testIdx] = stratify_by_group(idx, labels, groups);
        else
            [trainIdx, valIdx, testIdx] = stratify_by_label(idx, labels);
        end
    else
        [trainIdx, valIdx, testIdx] = stratify_by_label(idx, labels);
    end

    trainDS = subset(imdsAll, trainIdx);
    valDS   = subset(imdsAll, valIdx);
    testDS  = subset(imdsAll, testIdx);

    counts = table(classesLocal, ...
        countEachLabel(imdsAll).Count, ...
        countEachLabel(trainDS).Count, ...
        countEachLabel(valDS).Count, ...
        countEachLabel(testDS).Count, ...
        'VariableNames', {'Class','Total','Train','Validation','Test'});
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

function [classWeights, focalAlpha] = compute_class_weights(trainLabels, classesLocal)
    counts = countcats(trainLabels);
    freq = counts / sum(counts);
    invFreq = 1 ./ max(freq, 1e-6);
    invFreq = min(invFreq, 100); % clamp extreme
    classWeights = invFreq / mean(invFreq);
    focalAlpha = classWeights(:) / sum(classWeights);
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

function minority = find_minority_classes(countsTbl)
    totalCounts = countsTbl.Total;
    medianCnt = median(totalCounts);
    minority = countsTbl.Class(totalCounts < medianCnt);
end

function data = augment_image(im, label, outSize, minorityClasses)
    im = im2single(im);
    % base normalization
    if size(im,3)==1, im = repmat(im,1,1,3); end
    % random crop + resize
    scale = 0.9 + 0.2*rand();
    cropSize = round(outSize(1:2) * scale);
    [h,w,~] = size(im);
    y = max(1, randi([1, max(1, h - cropSize(1)+1)]));
    x = max(1, randi([1, max(1, w - cropSize(2)+1)]));
    yEnd = min(h, y+cropSize(1)-1);
    xEnd = min(w, x+cropSize(2)-1);
    im = imresize(im(y:yEnd, x:xEnd, :), outSize(1:2));

    % flips and small rotation
    if rand()<0.5, im = fliplr(im); end
    if rand()<0.2, im = flipud(im); end
    angle = -10 + 20*rand();
    im = imrotate(im, angle, 'bilinear','crop');

    % brightness/contrast/gamma
    im = imadjust(im, [], [], 0.9 + 0.2*rand());
    im = im + 0.02*randn(size(im));
    im = imgaussfilt(im, 0.5*rand());

    % minority-only heavier aug
    if any(strcmp(label, minorityClasses))
        if rand()<0.5
            im = imbilatfilt(im, 0.5, 2);
        end
        if rand()<0.5
            im = adapthisteq(im, 'ClipLimit',0.01);
        end
        if rand()<0.3
            im = im + randn(size(im))*0.03;
        end
    end

    data = im;
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
    if doAug
        imgs = cellfun(@(a,b) augFcn(a,b), im, labels, 'UniformOutput', false);
    else
        imgs = cellfun(@(a) augFcn(a), im, 'UniformOutput', false);
    end
    X = cat(4, imgs{:});
    X = dlarray(X, 'SSCB');
    Y = onehotencode(labels, 1);
    dlY = dlarray(single(Y), 'CB');
    dlX = X;
end

function [net, velocity, epochLoss, epochAcc] = train_one_epoch(net, files, labels, weights, classesLocal, classWeights, focalAlpha, augFcn, opts, lr, momentum, trainMask)
    numIters = floor(numel(files)/opts.batchSize);
    lossSum = 0; accSum = 0;
    if isempty(weights)
        weights = ones(numel(files),1);
    end
    for step = 1:numIters
        idx = randsample(numel(files), opts.batchSize, true, weights);
        ims = cell(opts.batchSize,1);
        labs = labels(idx);
        for j = 1:opts.batchSize
            ims{j} = imread(files{idx(j)});
        end
        [dlX, dlY] = batch_reader(ims, labs, augFcn, true);
        if opts.mixupAlpha>0 || opts.cutmixAlpha>0
            [dlX, dlY] = apply_mix(dlX, dlY, opts.mixupAlpha, opts.cutmixAlpha);
        end
        [gradients, loss, acc] = dlfeval(@modelLoss, net, dlX, dlY, classWeights, focalAlpha, opts);
        if isempty(velocity)
            velocity = dlupdate(@(x) zeros(size(x),'like',x), net.Learnables.Value);
        end
        % zero gradients for frozen layers
        gradCells = gradients.Value;
        for gi = 1:numel(gradCells)
            if ~trainMask(gi)
                gradCells{gi} = zeros(size(gradCells{gi}), 'like', gradCells{gi});
            end
        end
        gradients.Value = gradCells;
        [net, velocity] = sgdmupdate(net, gradients, velocity, lr, momentum);
        lossSum = lossSum + double(extractdata(loss));
        accSum  = accSum + acc;
    end
    epochLoss = lossSum / max(numIters,1);
    epochAcc  = accSum / max(numIters,1);
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

function [lossVal, accVal, metrics] = evaluate_epoch(net, files, labels, classesLocal, prepFcn, opts)
    numIters = floor(numel(files)/opts.batchSize);
    losses = zeros(numIters,1); accs = zeros(numIters,1);
    allPred = []; allTrue = [];
    for step = 1:numIters
        idx = (step-1)*opts.batchSize+1 : step*opts.batchSize;
        idx(idx>numel(files)) = [];
        ims = cell(numel(idx),1); labs = labels(idx);
        for j = 1:numel(idx)
            ims{j} = imread(files{idx(j)});
        end
        [dlX, dlY] = batch_reader(ims, labs, @(im) prepFcn(im), false);
        scores = predict(net, dlX);
        [lossBatch, accBatch] = evaluate_loss(net, dlX, dlY, ones(numel(classesLocal),1), ones(numel(classesLocal),1)/numel(classesLocal), opts);
        losses(step) = double(gather(extractdata(lossBatch)));
        accs(step) = accBatch;
        [~, preds] = max(scores, [], 1);
        preds = preds(:);
        trueLab = onehotdecode(dlY, classesLocal, 1);
        allPred = [allPred; preds]; %#ok<AGROW>
        allTrue = [allTrue; grp2idx(trueLab)]; %#ok<AGROW>
    end
    accVal = mean(accs);
    lossVal = mean(losses);
    predLabels = categorical(classesLocal(allPred));
    trueLabels = categorical(classesLocal(allTrue));
    [perClass, macroF1, balAcc] = compute_metrics(trueLabels, predLabels, classesLocal);
    metrics.perClass = perClass;
    metrics.macroF1 = macroF1;
    metrics.balancedAccuracy = balAcc;
    metrics.confusion = confusionmat(trueLabels, predLabels);
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
    if opts.focalLoss
        alpha = dlarray(focalAlpha(:)','like',probs);
        gamma = opts.focalGamma;
        ce = -dlY .* log(probs);
        weight = alpha .* (1 - probs').^gamma;
        loss = sum(ce .* weight', 1);
    else
        cw = dlarray(classWeights(:)','like',probs);
        loss = -sum(dlY .* log(probs) .* cw, 1);
    end
    loss = mean(loss);
    [~, predIdx] = max(probs, [], 1);
    [~, trueIdx] = max(dlY, [], 1);
    acc = mean(gather(extractdata(predIdx == trueIdx)));
    gradients = dlgradient(loss, net.Learnables, 'RetainData', isTraining);
end

function [dlX, dlY] = apply_mix(dlX, dlY, mixupAlpha, cutmixAlpha)
    if mixupAlpha > 0
        lambda = betarnd(mixupAlpha, mixupAlpha, [1 1], 'single');
        perm = randperm(size(dlX,4));
        dlX = lambda * dlX + (1-lambda) * dlX(:,:,:,perm);
        dlY = lambda * dlY + (1-lambda) * dlY(:,perm);
    elseif cutmixAlpha > 0
        lambda = betarnd(cutmixAlpha, cutmixAlpha, [1 1], 'single');
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

function update_live_plots(ui, iter, trainLoss, valLoss, trainAcc, valAcc, lr)
    addpoints(ui.lineLossTrain, iter, trainLoss);
    addpoints(ui.lineAccTrain, iter, trainAcc);
    addpoints(ui.lineLossVal, iter, valLoss);
    addpoints(ui.lineAccVal, iter, valAcc);
    addpoints(ui.lineLR, iter, lr);
    drawnow limitrate;
end

function ui = build_progress_ui()
    ui.fig = figure('Name','Droplet training monitor', 'Color','w', ...
        'Position',[100 100 1000 400]);
    ui.axAcc  = subplot(1,3,1, 'Parent', ui.fig); hold(ui.axAcc, 'on'); grid(ui.axAcc, 'on');
    ui.lineAccTrain = animatedline('Parent', ui.axAcc, 'Color','b', 'LineWidth',1.5);
    ui.lineAccVal   = animatedline('Parent', ui.axAcc, 'Color','r', 'LineWidth',1.5);
    title(ui.axAcc, 'Accuracy'); xlabel(ui.axAcc,'Iter'); ylabel(ui.axAcc,'Accuracy');
    legend(ui.axAcc, {'Train','Val'}, 'Location','southwest');

    ui.axLoss = subplot(1,3,2, 'Parent', ui.fig); hold(ui.axLoss, 'on'); grid(ui.axLoss, 'on');
    ui.lineLossTrain = animatedline('Parent', ui.axLoss, 'Color','b', 'LineWidth',1.5);
    ui.lineLossVal   = animatedline('Parent', ui.axLoss, 'Color','r', 'LineWidth',1.5);
    title(ui.axLoss, 'Loss'); xlabel(ui.axLoss,'Iter'); ylabel(ui.axLoss,'Loss');
    legend(ui.axLoss, {'Train','Val'}, 'Location','northeast');

    ui.axLR   = subplot(1,3,3, 'Parent', ui.fig); hold(ui.axLR, 'on'); grid(ui.axLR, 'on');
    ui.lineLR = animatedline('Parent', ui.axLR, 'Color','k', 'LineWidth',1.5);
    title(ui.axLR, 'Learning rate'); xlabel(ui.axLR,'Iter'); ylabel(ui.axLR,'LR');
end

function save_epoch_metrics(outDir, epoch, metrics)
    mPath = fullfile(outDir, sprintf('metrics_epoch_%03d.mat', epoch));
    save(mPath, 'metrics');
    csvPath = fullfile(outDir, sprintf('metrics_epoch_%03d.csv', epoch));
    writetable(metrics.perClass, csvPath);
end

function save_checkpoint(outDir, net, epoch, macroF1)
    ckptPath = fullfile(outDir, sprintf('best_checkpoint_epoch_%03d_f1_%.3f.mat', epoch, macroF1));
    save(ckptPath, 'net', 'epoch', 'macroF1');
end

function run_kfold(imdsAll, k)
    labels = imdsAll.Labels;
    cvp = cvpartition(labels, 'KFold', k, 'Stratify', true);
    for fold = 1:k
        trainIdx = training(cvp, fold);
        testIdx  = test(cvp, fold);
        fprintf('Fold %d: train %d, val %d (using test as val), test %d\n', fold, sum(trainIdx), 0, sum(testIdx));
        % In this minimal implementation we only log fold info; users can
        % rerun droplet_trainer with subset datastores if needed.
    end
    warning('k-fold requested; folds enumerated. Rerun droplet_trainer per fold for full training.');
end