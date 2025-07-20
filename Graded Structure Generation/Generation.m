clear;
clc;
% finished=[];
% save finished.mat finished;
num=20;
rdnum = 4;
pre = ['Round' num2str(rdnum)];
address=['D:\LiuJiamu\repository\MCTS-AL\Graded Structures Generation\STL' '\'];
filename = 'input matrix.mat';
tic
for l=1:num
    fprintf('Progress：%6.4f  ',l/num);
    toc
    random(l,pre,address,filename);
end
