function [p1,p2]=drawgyroid1(v,sizeofdata,nofv,nofgyroid,data0)
% C为等值面的数值图
% 绘图函数
% v采用函数createv输出值。
% sizeofdatta采用函数createdata输出值。
% nofv,nofgyroid含义与createv，createdata中的保持一致。
% 输出gyroid结构图像，x,y,z三轴等距，默认视角[1,1,1]。

% sizeofdata:sizeofdata0 = [3,3,3];nofv:21;nofgyroid:2;
% data0: 2*data0-1

minxyz=min(data0);%[1,1,1,x] 
maxxyz=max(data0);%[3,3,3,y]
% disp(minxyz);
% disp(maxxyz);
% yspace=linspace(minxyz(1)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(1)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(1)*nofv);
% % 1-(21-1)/2*(2/21), 3+(21-1)/2*(2/21), 63
% xspace=linspace(minxyz(2)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(2)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(2)*nofv);
% zspace=linspace(minxyz(3)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(3)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(3)*nofv);
% [x,y,z]=meshgrid(xspace,yspace,zspace); %63*63*63
[x,y,z] = meshgrid(linspace(0,9,sizeofdata(3)*nofv));
% disp(x);

%gyroid expression, C = 0.9
o = sin(2*pi/nofgyroid*(x)).*cos(2*pi/nofgyroid*y) + sin(2*pi/nofgyroid*y).*cos(2*pi/nofgyroid*z) + sin(2*pi/nofgyroid*z).*cos(2*pi/nofgyroid*(x));
% o = transpose(o);
% fprintf('%1.1d', size(o));
% [xdim, ydim, zdim] = size(o);
% for i=1:zdim
%     o(:,:,i) = transpose(o(:,:,i));
% end

% 边界
% xleft= (y==(minxyz(1)-(nofv-1)/2*(nofgyroid/nofv))) ;
% o(xleft) =0.9;
% xright= (y==(maxxyz(1)+(nofv-1)/2*(nofgyroid/nofv))) ;
% o(xright) =0.9;
% yleft= (x==(minxyz(2)-(nofv-1)/2*(nofgyroid/nofv))) ;
% o(yleft) =0.9;
% yright= (x==(maxxyz(2)+(nofv-1)/2*(nofgyroid/nofv))) ;
% o(yright) =0.9;
% zleft= (z==(minxyz(3)-(nofv-1)/2*(nofgyroid/nofv)) );
% o(zleft) =0.9;
% zright= (z==(maxxyz(3)+(nofv-1)/2*(nofgyroid/nofv))) ;
% o(zright) =0.9;
% 
% xleft= (y==-0.05);
% o(xleft) =0.9;
% xright= (y==6.05);
% o(xright) =0.9;
% yleft= (x==-0.05);
% o(yleft) =0.9;
% yright= (x==6.05);
% o(yright) =0.9;
% zleft= (z==-0.05);
% o(zleft) =0.9;
% zright= (z==6.05);
% o(zright) =0.9;

% zright= (abs(z-8.75)<=0.015) ;
% o(zright) =0.9;
% zleft= (abs(z-1.75)<=0.015) ;
% o(zleft) =0.9;
% % xy1= (abs((x-2.75).*(x-2.75)+(y-2.75).*(y-2.75)-1.5*1.5)<=0.03);
% % o(xy1) =0.9;
% xy2= ((x-2.75).*(x-2.75)+(y-2.75).*(y-2.75)-1.5*1.5>0.03);
% o(xy2) =0;
% zright1= (z-8.75>0.015) ;
% o(zright1) =0;
% zleft1= (z-1.75<-0.015) ;
% o(zleft1) =0;

% xleft= (y==-0.05);
% o(xleft) =0;
% xright= (y==6.05);
% o(xright) =0;
% yleft= (x==-0.05);
% o(yleft) =0;
% yright= (x==6.05);
% o(yright) =0;
% zleft= (z==-0.05);
% o(zleft) =0;
% zright= (z==6.05);
% o(zright) =0;

% xleft= (y==0.05);
% disp(size(xleft))
% A = abs(o(xleft)) - v(xleft);
% A(A>0) = 0;
% o(xleft) = A + v(xleft);
% 
% xright= (y==5.95);
% A = abs(o(xright)) - v(xright);
% A(A>0) = 0;
% o(xright) = A + v(xright);
% 
% yleft= (x==0.05);
% A = abs(o(yleft)) - v(yleft);
% A(A>0) = 0;
% o(yleft) = A + v(yleft);
% 
% yright= (x==5.95);
% A = abs(o(yright)) - v(yright);
% A(A>0) = 0;
% o(yright) = A + v(yright);
% 
% zleft= (z==0.05);
% A = abs(o(zleft)) - v(zleft);
% A(A>0) = 0;
% o(zleft) = A + v(zleft);
% 
% zright= (z==5.95);
% A = abs(o(zright)) - v(zright);
% A(A>0) = 0;
% o(zright) = A + v(zright);

%%%%%%%%%%%%%%%%%%%%%%%
% o = o + v;
% 另一个域
o = v-abs(o);
% o = abs(o)-v;

% o(o>0) = 0;
% disp(o)

%o=o+permute(v,[2 1 3]);

% o=permute(o,[2 1 3]);

R=10/2/pi;

%  x1=x.*(y/R);
%  y1=y;


% p = isosurface(((R+y-R).*cos(x/R)),(R+y-R).*sin(x/R),z+x/R/2/pi*2,o,0.9);       

% p = isosurface(((R+y1-R).*cos(x1./y1)),(R+y1-R).*sin(x1./y1),z,o,0.9);   
x1=x;
y1=y;
p1=isosurface(x1,y1,z,o,0);
p2 = isocaps(x1,y1,z,o,0);
% patch(p,'facecolor','y','edgecolor','none')
% camlight
% axis equal
% view([1,1,1]); 
% xlabel('x/mm')
% ylabel('y/mm')
% zlabel('z/mm')
end