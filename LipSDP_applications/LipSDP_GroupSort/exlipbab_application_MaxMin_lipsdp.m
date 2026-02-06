% we read the weights from the exlipbab_saved_networks folder and approximate the Lipschitz constants for the networks trained in the ExLipBab paper using LipSDP
% this code requires the QCS_MaxMin repository from https://github.com/ppauli/QCs_MaxMin

close all
clear all
clc

clear W

% computation for the abalone network:
%str_net = 'abalone_net_(9x16x16x1)_GroupSort_numGroups8_weights.mat';
%load("exlipbab_saved_networks\abalone\abalone_net_(9x16x16x1)_GroupSort_numGroups8_weights.mat")

%computation for the small wine quality network:
%str_net = 'wine_net_(11x12x12x1)_GroupSort_numGroups6_weights.mat';
%load("exlipbab_saved_networks\wine\wine_net_(11x12x12x1)_GroupSort_numGroups6_weights.mat")

%computation for the large wine quality network:
%str_net = 'wine_net_(11x24x24x1)_GroupSort_numGroups12_weights.mat';
%load("exlipbab_saved_networks\wine\wine_net_(11x24x24x1)_GroupSort_numGroups12_weights.mat")

% computation for the bike sharing network:
str_net = 'bike_net_(13x20x16x8x3)_GroupSort_numGroups10_weights.mat';
load("exlipbab_saved_networks\bike_sharing\bike_net_(13x20x16x8x3)_GroupSort_numGroups10_weights.mat")


% we transform the weights to to double format; otherwise errors occur
%&for ii = 1:length(W)
%    W{ii} = double(W{ii});
%end


%layers = length(W);
layers = length(weights)
clear L_vec L_ResReLU_vec L_SR_vec Ltriv_vec info time info_ResReLU ...
    time_ResReLU info_SR time_SR

type = 'l2';

    
%% LipSDP-GS
disp(['Starting LipSDP-NSR for ' str_net ' for ' type])
[L,status] = LipschitzEstimation(weights,type);

%for ReLU computation:
%disp(['Starting LipSDP for ReLU for ' str_net ' for ' type])
%[L_ResReLU,status_ResReLU] = LipschitzEstimation_ResReLU_8(weights,type); !! use the correct hard coded version !!
    
%% MP bound
Ltriv = 1;
for ii = 1:length(weights)
    Ltriv = norm(weights{ii})*Ltriv;

end

    
%% collect results
L_vec = L
%L_ResReLU_vec(ll) = L_ResReLU
Ltriv_vec = Ltriv

info = status.info
time = status.solvertime
%info_ResReLU{ll} = status_ResReLU.info
%time_ResReLU(ll) = status_ResReLU.solvertime

%save(['results\res_' str_net],'L_vec','L_ResReLU_vec','Ltriv_vec','info','time','info_ResReLU','time_ResReLU')
%save(['results\res_' str_net],'L_vec','Ltriv_vec','info','time')

