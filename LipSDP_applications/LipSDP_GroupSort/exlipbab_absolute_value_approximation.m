% In this file, we compute the approximate Lipschitz constant for the networks trained on the absolute value function using LipSDP
% we return a mean Lipschitz constant over all 20 trained networks, as well as a standard deviation
% we also return the mean compute time and standard deviation

close all
clear all
clc

clear W
clear L_vec L_ResReLU_vec L_SR_vec Ltriv_vec info time info_ResReLU ...
        time_ResReLU info_SR time_SR

base_str_net = '[1, 6, 6, 1]GroupSort_numGroups3simul_';
L_vec = [];
time = [];

for nn = 0:19
    str_net = [base_str_net num2str(nn) '_weights.mat'];
    load(['exlipbab_saved_networks\absolute_value\' str_net])
    % we transform the weights to to double format; otherwise errors occur
    for ii = 1:length(W)
        W{ii} = double(W{ii});
    end

    layers = length(W);

    

    type = 'l2';

    %% LipSDP-GS
    disp(['Starting LipSDP-NSR for ' str_net ' for ' type])
    [L,status] = LipschitzEstimation(W,type);

    % append into vectors
    L_vec(nn+1) = L;
    time(nn+1) = status.solvertime;
    L_vec
    L

end

mean_L = mean(L_vec);
std_L = std(L_vec);
mean_time = mean(time);
std_time = std(time);

disp(['Mean Lipschitz constant over 20 networks: ' num2str(mean_L) ' with std: ' num2str(std_L)])
disp(['Mean compute time over 20 networks: ' num2str(mean_time) ' with std: ' num2str(std_time)])

