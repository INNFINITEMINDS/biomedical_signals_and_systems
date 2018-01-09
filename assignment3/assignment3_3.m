clear all; 
close all;
%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 3_3
% by Sheshan Aaron, 11/2014

%% ============================3.1=========================== %%
N = 500; % the number of samples
P = 4; % the number of sources
S = zeros(P,N); % P sources, each having N time samples

S(1,:)=sin([1:N]/2); % sinusoid
S(2,:) = square([1:N]/5); % square wave
S(3,:)=((rem([1:N],27)-13)/9); % saw-tooth
S(4,:)=((rand(1,N)<.5)*2-1).*log(rand(1,N)); % impulsive noise 
A = [-0.3 -0.6 -0.3 -0.4; 1.6 -2.0 -1.2 1.0;...
 -0.4 0.0 -1.0 -0.3; 0.2 1.7 0.4 -2.0];

X = A*S; % mixed multivariate signals (P x N)

%plot all original sources
figure
for i=1: 8
    subplot(2,4,i)    
    if(i<5)
        plot([1:N], S(i,:))
        title(sprintf('S_%i', i))
        if(i<4)
            set(gca,'ylim',[-2 2])
        end
    else
        plot([1:N], X(i-4,:))
        title(sprintf('X_%i', i-4))
    end
    set(gca,'xlim',[0 N])
    xlabel('Time Index')
    ylabel('Magnitude')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%print('3_1_original_sources_and_mixed_signals','-dpng','-r300');

%% ============================3.2=========================== %%
% FastICA 
[S_est, A_est, W_est] = fastica(X); % S_est is the estimated independent components and A_est is the estimated mixing matrix
figure
for i=1: 4
    %subplot(2,4,i)
    figure   
    plot([1:N], S_est(i,:))
    title(sprintf('Estimated S_%i', i))
    set(gca,'xlim',[0 N])
    xlabel('Time Index')
    ylabel('Magnitude')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
%% ============================3.3=========================== %%
% Reconstruction
%s1 and s4 are impulisve noise and sine wave respectively, so exclude them
%include s_est[2,3]
X_rec = A_est(:,[3,4])*S_est([3,4],:);

figure
for i=1: 8
    subplot(2,4,i)    
    if(i<5)
        plot([1:N], S_est(i,:))
        title(sprintf('Estimated S_%i', i))
    else
        box on
        hold on
        plot([1:N], X(i-4,:),'b')
        plot([1:N], X_rec(i-4,:),'r')
        hold off
        title(sprintf('Reconstructed X_%i', i-4))
        %legend('original', 'reconstructed')
    end
    set(gca,'xlim',[0 N])
    xlabel('Time Index')
    ylabel('Magnitude')    
    set(gca, 'LineWidth', 1)
    grid on
    set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
    set(gcf, 'PaperSize', [5 5]); %Keep the same paper size    
end
print('3_2_3_estimated_IC_and_rec_mixed_signals','-dpng','-r300');