clear all; 
close all;
%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 3_2
% by Sheshan Aaron, 11/2014

%% ============================2.1=========================== %%
% load and display signal
load assg3_lep

N = length(lep); %Number of samples
P = 7; % the number of features/variables

%display each channel seperately
for i=1:7
    figure
    plot(time, lep(:,i)); axis tight
    title(sprintf('LEP Channel %i', i)) 
    xlabel('Time (s)')
    ylabel('Amplitude (\muV)')
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
    %print(sprintf('2_1_lep_channel%i', i),'-dpng','-r300');
end

%% ============================2.2=========================== %%
% PCA Decompositon
[Z, mu, sigma] = zscore(lep); % standardize the input data to make it have zero mean and unit variance for each feature/variable
[COEFF,SCORE,latent] = princomp(Z);
% COEFF: a P x P matrix, each column containing coefficients for one principal component
% SCORE: a N x P matrix, each column being a principal component
% latent: a Px1 vector containing the eigenvalues of the covariance matrix of X 
% Scree plot 
figure
plot(latent,'o--')
xlabel('Number of Components')
ylabel('Eigenvalue')
title('Scree Plot')
grid on

set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
%print('2_2_scree_plot','-dpng','-r300');

%% ============================2.3=========================== %%
%reconstruct signal from one principal component
% Reconstruction

P_new = 1; % the new reduced dimension (P_new < P)
Z_rec = (SCORE(:,1:P_new)*COEFF(:,1:P_new)'); % reconstruct signals using P_new principal components
lep_rec = Z_rec .* (ones(N,1)*sigma) + (ones(N,1)*mu);% rescale the reconstructed data to the original scale

figure
for i=1:7 
    subplot(3,3,i)
    plot(time, lep(:,i), 'b'); axis tight
    hold on;
    plot(time, lep_rec(:,i),'r'); axis tight
    hold off;
    title(sprintf('Reconstructed LEP Channel %i', i)) 
    xlabel('Time (s)')
    ylabel('Amplitude (\muV)')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%print('2_3_rec_1_p_lep_channels','-dpng','-r300');

%% ============================2.4=========================== %%
%reconstruct signal from 3 and 5 principal component

% Reconstruction
%reconstruct signal from 3 principal component
P_new = 3; % the new reduced dimension (P_new < P)
Z_rec_3 = (SCORE(:,1:P_new)*COEFF(:,1:P_new)'); % reconstruct signals using P_new principal components
lep_rec_3 = Z_rec_3 .* (ones(N,1)*sigma) + (ones(N,1)*mu);% rescale the reconstructed data to the original scale

figure
for i=1:7 
    subplot(3,3,i)
    plot(time, lep(:,i), 'b'); axis tight
    hold on;
    plot(time, lep_rec_3(:,i),'r'); axis tight
    hold off;
    title(sprintf('Reconstructed LEP Channel %i', i)) 
    xlabel('Time (s)')
    ylabel('Amplitude (\muV)')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%print('2_3_rec_3_p_lep_channels','-dpng','-r300');

%reconstruct signal from 3 and 5 principal component
P_new = 5; % the new reduced dimension (P_new < P)
Z_rec_5 = (SCORE(:,1:P_new)*COEFF(:,1:P_new)'); % reconstruct signals using P_new principal components
lep_rec_5 = Z_rec_5 .* (ones(N,1)*sigma) + (ones(N,1)*mu);% rescale the reconstructed data to the original scale

figure
for i=1:7 
    subplot(3,3,i)
    plot(time, lep(:,i), 'b'); axis tight
    hold on;
    plot(time, lep_rec_5(:,i),'g'); axis tight
    hold off;
    title(sprintf('Reconstructed LEP Channel %i', i)) 
    xlabel('Time (s)')
    ylabel('Amplitude (\muV)')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%print('2_3_rec_5_p_lep_channels','-dpng','-r300');

figure
for i=1:7 
    subplot(3,3,i)
    plot(time, lep(:,i), 'b'); axis tight
    hold on;
    plot(time, lep_rec_3(:,i),'r'); axis tight
    plot(time, lep_rec_5(:,i),'g'); axis tight
    hold off;
    title(sprintf('Reconstructed LEP Channel %i', i)) 
    xlabel('Time (s)')
    ylabel('Amplitude (\muV)')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%print('2_3_rec_3_and_5_p_lep_channels','-dpng','-r300');
