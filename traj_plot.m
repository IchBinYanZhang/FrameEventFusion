% bird view trajectory plot
% close all
clear;clc;


%% get ground view images and load calibrations
% I_left = imread('/home/yzhang/Videos/SenseEmotion2/calib/IDS/left/ImageSequences/250.png');
% I_right = imread('/home/yzhang/Videos/SenseEmotion2/calib/IDS/right/ImageSequences/250.png');
% load('/home/yzhang/Documents/SenseEmotion/Experiment2A/Calibration_Res/IDS/right-left/calib_params.mat');

% find the ground plane in 3D
% [n, V, P] = ComputePlane(I_right, I_left, stereoParams);
%%% geometry: cam-left and cam-right in experiment2A
V = [0.93603 0.26413; 0.10476 0.42176; -0.33597 0.86738];

%% read the trajectory file and map to ground

traj3D_set = cell(length(7:20),1);
traj2D_set = cell(length(7:20),1);
traj3D_velocity_set = cell(length(7:20),1);
traj2D_velocity_set = cell(length(7:20),1);
filepath = 'E:\\SenseEmotion\\FrameEventFusion\\Tracking3D_VC2015\\x64\\Release\\';
for n =26:26
    traj3Dhomo = importdata([filepath,'baseline_s',num2str(n),'.txt']);
    traj3D = zeros(length(traj3Dhomo(:,1)), 3);
    traj3D(:,1) = traj3Dhomo(:,1)./traj3Dhomo(:,4);
    traj3D(:,2) = traj3Dhomo(:,2)./traj3Dhomo(:,4);
    traj3D(:,3) = traj3Dhomo(:,3)./traj3Dhomo(:,4);
    
    traj2D = traj3D*V;
    traj3D_set{n} = traj3D;
    traj2D_set{n} = traj2D;
    
    traj3D_velocity_set{n} = sqrt(sum(abs(diff(traj3D)).^2,2));
    traj2D_velocity_set{n} = sqrt(sum(abs(diff(traj2D)).^2,2));
end
    
% traj3Dhomo2 = importdata('bin/Debug/traj_3Dhomo.txt');
% traj3D2 = zeros(length(traj3Dhomo2(:,1)), 3);
% traj3D2(:,1) = traj3Dhomo2(:,1)./traj3Dhomo2(:,4);
% traj3D2(:,2) = traj3Dhomo2(:,2)./traj3Dhomo2(:,4);
% traj3D2(:,3) = traj3Dhomo2(:,3)./traj3Dhomo2(:,4);
% 
% 
% traj2D2 = traj3D2*V;



%% birdview trajectory show
for n = 26:26
    figure;
    plot(traj2D_set{n}(:,1), traj2D_set{n}(:,2));hold on;
    scatter(traj2D_set{n}(1:end-1,1), traj2D_set{n}(1:end-1,2),2, traj2D_velocity_set{n});
    grid on;colorbar;

    figure(2)
    plot3(traj3D_set{n}(:,1), traj3D_set{n}(:,2),traj3D_set{n}(:,3) );
end

