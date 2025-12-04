%% Load pretrained SqueezeNet
net = squeezenet;   % Needs Deep Learning Toolbox Model for SqueezeNet

% Turn it into a layer graph
lgraph = layerGraph(net);

%% Set number of classes from your datastore
imds = imageDatastore("export 4x", ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

classes    = categories(imds.Labels);
numClasses = numel(classes);

%% Replace final conv and classification layers

% New final 1x1 conv to match your classes
newConvFinal = convolution2dLayer(1,numClasses, ...
    "Name","conv10_droplet", ...
    "WeightLearnRateFactor",10, ...
    "BiasLearnRateFactor",10);

% New softmax and classification layers
newSoftmax  = softmaxLayer("Name","prob_droplet");
newClassOut = classificationLayer("Name","classoutput_droplet");

% Replace layers in the graph
lgraph = replaceLayer(lgraph,"conv10",newConvFinal);
lgraph = replaceLayer(lgraph,"prob",newSoftmax);
lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",newClassOut);

%% Visualize the network

% Text and interactive view
analyzeNetwork(lgraph);

% Or a quick graph plot
figure;
plot(lgraph);
title("Droplet classifier based on SqueezeNet");
You will feed images as 227 by 227 and convert to fake RGB so you can use the pretrained filters unchanged

matlab
Copy code
aug = imageDataAugmenter( ...
    "RandRotation",[-5 5], ...
    "RandXTranslation",[-4 4], ...
    "RandYTranslation",[-4 4]);

dsTrain = augmentedImageDatastore([227 227],imds, ...
    "ColorPreprocessing","gray2rgb", ...
    "DataAugmentation",aug);

options = trainingOptions("adam", ...
    "MiniBatchSize",64, ...
    "MaxEpochs",10, ...
    "InitialLearnRate",1e-4, ...
    "Shuffle","every-epoch", ...
    "ExecutionEnvironment","gpu", ...
    "Plots","training-progress");

netDroplet = trainNetwork(dsTrain,lgraph,options);