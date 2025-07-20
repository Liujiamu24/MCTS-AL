function [p,p1,p2] = random(l,pre,address,filename)
%% 
load(filename)
%%
n1=3; %3 dimensions
n2=3;
n3=3;
r1=zeros(n1*n2*n3,3); 
% 
mydata = reshape(mydata,[20,3,3,3]);
%% lattice 2 sheet
for a=1:n1
    for b=1:n2
        for c=1:n3
            r1(n2*n3*a-n2*n3+n3*b-n3+c-1+1,:)=[b,a,c];
            r2(n2*n3*a-n2*n3+n3*b-n3+c-1+1,:)=mydata(l,a,b,c); 
        end
    end
end
r2 = (1-r2)/2;
r2=round(100*(1-r2))/10;
data0=[r1 r2];

%% 
x = [0.5326 0.8798 0.5622 0.5979 0.6308 0.6638 0.6971 0.4674 0.3692 0.2009];
y = [1 2 1.1 1.2 1.3 1.4 1.5 0.8 0.5 0];
[fit,~] = createFit(10*x,y);

for i=1:n1*n2*n3
    data0(i,4)=fit(data0(i,4));
end
%% 
b=3;
sizeofdata0=[n1,n2,n3];
accu=21;
v=createv_2(data0,sizeofdata0,accu,b);
v = v - 0.9;

%% 
sizeofgyroid=3;
[p1,p2] = drawgyroid(v,sizeofdata0,accu,sizeofgyroid,sizeofgyroid*data0-1);
nn = p1.faces(:,1);
p1.faces(:,1) = p1.faces(:,2);
p1.faces(:,2) = nn;

nn = p2.faces(:,1);
p2.faces(:,1) = p2.faces(:,2);
p2.faces(:,2) = nn;

p = p1;
p.vertices = [p1.vertices; p2.vertices];
p.faces = [p1.faces; p2.faces + size(p1.vertices,1)];

%% name 
name=[num2str(l) '.stl'];
name=[pre '_' name];
filenm = [num2str(l) '.txt'];
filenm = [pre '_' filenm];
stlwrite(filenm,address,name,p);
end
