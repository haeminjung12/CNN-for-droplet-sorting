function lgraph = droplet_squeezenet_64(numClasses)
%DROPLET_SQUEEZENET_64 Create a SqueezeNet graph adapted for 64x64 inputs.
%   LGRAPH = DROPLET_SQUEEZENET_64(NUMCLASSES) returns a layerGraph based on
%   SqueezeNet with a 64x64 input layer, a global average pooling layer, and
%   a final 1x1 convolution + softmax sized for NUMCLASSES. Use this
%   initializer for transfer learning with 64x64 droplet crops.
%
%   Example:
%       imds = imageDatastore('cropped_dataset', ...
%           'IncludeSubfolders',true, 'LabelSource','foldernames');
%       classes = categories(imds.Labels);
%       lgraph = droplet_squeezenet_64(numel(classes));
%       analyzeNetwork(lgraph);
%
%   Requires Deep Learning Toolbox Model for SqueezeNet™ support package.

arguments
    numClasses (1,1) {mustBePositive, mustBeInteger}
end

% Load pretrained SqueezeNet
net = squeezenet();
lgraph = layerGraph(net);

% Replace input with 64x64 layer
inputLayer = imageInputLayer([64 64 3], ...
    'Name','data', ...
    'Normalization','zerocenter');

lgraph = replaceLayer(lgraph, 'data', inputLayer);

% Replace final conv to match classes
newConvFinal = convolution2dLayer(1, numClasses, ...
    'Name','conv10_droplet', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newSoftmax  = softmaxLayer('Name','prob_droplet');
newClassOut = classificationLayer('Name','classoutput_droplet');

lgraph = replaceLayer(lgraph, 'conv10', newConvFinal);
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassOut);

% Swap fixed-size average pooling for global pooling to support 64x64 input
if any(strcmp({lgraph.Layers.Name}, 'pool10'))
    gap = globalAveragePooling2dLayer('Name','pool10_global');
    lgraph = replaceLayer(lgraph, 'pool10', gap);
end

end