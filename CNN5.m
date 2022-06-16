%% 准备工作空间
clc
clear all
close all
digitDatasetPath = fullfile('./datas/');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');% 采用文件夹名称作为数据标记
countEachLabel(imds)
numTrainFiles = 350;%取个做训练
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% 查看图片的大小
img=readimage(imds,1);
size(img)
%% 定义卷积神经网络的结构
layers = [
    imageInputLayer([20 20 1],"Name","imageinput")
    convolution2dLayer([5 5],6,"Name","conv_1","Padding",[1 1 1 1],"Stride",[4 4])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[3 3])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],120,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%绘制层
plot(layerGraph(layers));
%% 训练神经网络
% 设置训练参数
options = trainingOptions('sgdm',...
    'maxEpochs', 50, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'Plots','training-progress');% 显示训练进度
% 训练神经网络，保存网络
net = trainNetwork(imdsTrain, layers ,options);
