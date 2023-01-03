clear
close all

%%

load('train_info_2000_3bit.mat')

tr_loss = train_info.tr_loss;
val_loss = train_info.val_loss;
corr = train_info.corr;

%%

figure(1)

p1 = plot(tr_loss);
p1.LineWidth = 1.0;

hold on

p2 = plot(val_loss);
p2.LineWidth = 1.0;

grid on
box on

xlabel('Number of epochs', 'interpreter', 'latex', 'FontSize', 12)
ylabel('MSE loss', 'interpreter', 'latex', 'FontSize', 12)

legend({'Training loss', 'Testing loss'}, ...
    'interpreter', 'latex', ...
    'FontSize', 12)
