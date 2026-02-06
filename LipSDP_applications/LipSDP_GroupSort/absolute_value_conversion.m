% The LipSDP algorithm expects double format and weights an a field called 'weights'
% we load the ReLU networks saved with the exlipbab framework and convert the weights accordingly

close all
clear all
clc

clear W
clear L_vec L_ResReLU_vec L_SR_vec Ltriv_vec info time info_ResReLU ...
        time_ResReLU info_SR time_SR

base_str_net = '[1, 6, 6, 1]ReLUsimul_';
L_vec = [];
time = [];

for nn = 0:19
    str_net = [base_str_net num2str(nn) '_weights.mat'];
    load(['exlipbab_saved_networks\absolute_value\' str_net])
    % we transform the weights to to double format; otherwise errors occur
    for ii = 1:length(W)
        W{ii} = double(W{ii});
    end

    weights = W; % rename W to weights
    save(['exlipbab_saved_networks\absolute_value_converted\absolute_value_net_(1x6x6x1)_ReLU_' num2str(nn) '_weights_converted.mat'],'weights')
end