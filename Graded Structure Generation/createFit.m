function [fitresult, gof] = createFit(x, y)
%CREATEFIT(X,Y)
%  创建一个拟合。
%
%  要进行 '无标题拟合 1' 拟合的数据:
%      X 输入: x
%      Y 输出: y
%  输出:
%      fitresult: 表示拟合的拟合对象。
%      gof: 带有拟合优度信息的结构体。
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 28-Feb-2024 09:51:24 自动生成


%% 拟合: '无标题拟合 1'。
[xData, yData] = prepareCurveData( x, y );

% 设置 fittype 和选项。
% ft = 'pchipinterp';
% opts = fitoptions( 'Method', 'PchipInterpolant' );
% opts.ExtrapolationMethod = 'pchip';
% opts.Normalize = 'on';
ft = fittype("poly1");

% 对数据进行模型拟合。
% [fitresult, gof] = fit( xData, yData, ft, opts );
[fitresult, gof] = fit( xData, yData, ft);

% % 绘制数据拟合图。
% figure( 'Name', '无标题拟合 1' );
% h = plot( fitresult, xData, yData );
% legend( h, 'y vs. x', '无标题拟合 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % 为坐标区加标签
% xlabel( 'x', 'Interpreter', 'none' );
% ylabel( 'y', 'Interpreter', 'none' );
% grid on


