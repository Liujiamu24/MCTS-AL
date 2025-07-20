function unitofv=createunitofv(datainput,positon,nofv,dofv)
% create unit of v for a specific position
% 
% position=[i,j,k]
%n of v, d of v
neibourhoods=findneighbour(datainput,positon); %datainput = data = data0, 27*4
unitofv=ones(nofv-2*dofv,nofv-2*dofv,nofv-2*dofv); % 21,3 [15,15,15]

%process of neighbeourhoods unitofv:all 1
if ~(isnan(neibourhoods(2,2,2))) %neibourhoods(2,2,2) cant be nan?1
    unitofv=unitofv*neibourhoods(2,2,2);
else
    unitofv=zeros(nofv,nofv,nofv);
    unitofv(:,:,:)=NaN;
    return
end
if isnan(neibourhoods(3,2,2)) %position are (3,y,z)
    neibourhoods(3,2,2)=neibourhoods(2,2,2);
end
if isnan(neibourhoods(1,2,2))
    neibourhoods(1,2,2)=neibourhoods(2,2,2);
end
if isnan(neibourhoods(2,3,2))
    neibourhoods(2,3,2)=neibourhoods(2,2,2);
end
if isnan(neibourhoods(2,1,2))
    neibourhoods(2,1,2)=neibourhoods(2,2,2);
end
if isnan(neibourhoods(2,2,3))
    neibourhoods(2,2,3)=neibourhoods(2,2,2);
end
if isnan(neibourhoods(2,2,1))
    neibourhoods(2,2,1)=neibourhoods(2,2,2);
end

if isnan(neibourhoods(3,3,2))
    neibourhoods(3,3,2)=(neibourhoods(3,2,2)+neibourhoods(2,3,2))/2;
end
if isnan(neibourhoods(3,1,2))
    neibourhoods(3,1,2)=(neibourhoods(3,2,2)+neibourhoods(2,1,2))/2;
end
if isnan(neibourhoods(1,3,2))
    neibourhoods(1,3,2)=(neibourhoods(1,2,2)+neibourhoods(2,3,2))/2;
end
if isnan(neibourhoods(1,1,2))
    neibourhoods(1,1,2)=(neibourhoods(1,2,2)+neibourhoods(2,1,2))/2;
end
if isnan(neibourhoods(3,2,3))
    neibourhoods(3,2,3)=(neibourhoods(3,2,2)+neibourhoods(2,2,3))/2;
end
if isnan(neibourhoods(3,2,1))
    neibourhoods(3,2,1)=(neibourhoods(3,2,2)+neibourhoods(2,2,1))/2;
end
if isnan(neibourhoods(1,2,3))
    neibourhoods(1,2,3)=(neibourhoods(1,2,2)+neibourhoods(2,2,3))/2;
end
if isnan(neibourhoods(1,2,1))
    neibourhoods(1,2,1)=(neibourhoods(1,2,2)+neibourhoods(2,2,1))/2;
end
if isnan(neibourhoods(2,3,3))
    neibourhoods(2,3,3)=(neibourhoods(2,3,2)+neibourhoods(2,2,3))/2;
end
if isnan(neibourhoods(2,3,1))
    neibourhoods(2,3,1)=(neibourhoods(2,3,2)+neibourhoods(2,2,1))/2;
end
if isnan(neibourhoods(2,1,3))
    neibourhoods(2,1,3)=(neibourhoods(2,1,2)+neibourhoods(2,2,3))/2;
end
if isnan(neibourhoods(2,1,1))
    neibourhoods(2,1,1)=(neibourhoods(2,1,2)+neibourhoods(2,2,1))/2;
end

if isnan(neibourhoods(1,1,1))
    neibourhoods(1,1,1)=(neibourhoods(1,2,2)+neibourhoods(2,1,2)+neibourhoods(2,2,1))/3;
end
if isnan(neibourhoods(3,1,1))
    neibourhoods(3,1,1)=(neibourhoods(3,2,2)+neibourhoods(2,1,2)+neibourhoods(2,2,1))/3;
end
if isnan(neibourhoods(1,3,1))
    neibourhoods(1,3,1)=(neibourhoods(1,2,2)+neibourhoods(2,3,2)+neibourhoods(2,2,1))/3;
end
if isnan(neibourhoods(1,1,3))
    neibourhoods(1,1,3)=(neibourhoods(1,2,2)+neibourhoods(2,1,2)+neibourhoods(2,2,3))/3;
end
if isnan(neibourhoods(1,3,3))
    neibourhoods(1,3,3)=(neibourhoods(1,2,2)+neibourhoods(2,3,2)+neibourhoods(2,2,3))/3;
end
if isnan(neibourhoods(3,1,3))
    neibourhoods(3,1,3)=(neibourhoods(3,2,2)+neibourhoods(2,1,2)+neibourhoods(2,2,3))/3;
end
if isnan(neibourhoods(3,3,1))
    neibourhoods(3,3,1)=(neibourhoods(3,2,2)+neibourhoods(2,3,2)+neibourhoods(2,2,1))/3;
end
if isnan(neibourhoods(3,3,3))
    neibourhoods(3,3,3)=(neibourhoods(3,2,2)+neibourhoods(2,3,2)+neibourhoods(2,2,3))/3;
end

% neibourhoods matrix completed
%添加过渡段

for i=1:dofv %in one case , dofv = 3
    nownumber=neibourhoods(2,2,2)+i*(neibourhoods-neibourhoods(2,2,2))/(2*dofv+1);
    %delta matrix,
    temp=zeros(1,nofv-2*dofv+2*i-2,nofv-2*dofv+2*i-2); %temp:[1,15+2*(i-1),15+2*(i-1)]
    temp(:,:,:)=nownumber(3,2,2);
    unitofv=cat(1,unitofv,temp);%x+ 
    temp(:,:,:)=nownumber(1,2,2);
    unitofv=cat(1,temp,unitofv);%x- %unitofv:[17,15,15]
    temp=zeros(nofv-2*dofv+2*i-2+2,1,nofv-2*dofv+2*i-2); %temp:[15+2*(i-1)+2,1,15+2*(i-1)]
    temp(:,:,:)=nownumber(2,3,2);
    unitofv=cat(2,unitofv,temp);%y+
    temp(:,:,:)=nownumber(2,1,2);
    unitofv=cat(2,temp,unitofv);%y- %unitofv:[17,17,15]
    temp=zeros(nofv-2*dofv+2*i-2+2,nofv-2*dofv+2*i-2+2,1);
    temp(:,:,:)=nownumber(2,2,3);
    unitofv=cat(3,unitofv,temp);%z+
    temp(:,:,:)=nownumber(2,2,1);
    unitofv=cat(3,temp,unitofv);%z- %unitofv:[17,17,17]

    
    unitofv(end,end,:)=nownumber(3,3,2);%x+,y+
    unitofv(1,1,:)=nownumber(1,1,2);%x-,y-
    unitofv(end,1,:)=nownumber(3,1,2);%x+,y-
    unitofv(1,end,:)=nownumber(1,3,2);%x,y+  
    unitofv(end,:,end)=nownumber(3,2,3);
    unitofv(1,:,1)=nownumber(1,2,1);
    unitofv(end,:,1)=nownumber(3,2,1);
    unitofv(1,:,end)=nownumber(1,2,3);    
    unitofv(:,end,end)=nownumber(2,3,3);
    unitofv(:,1,1)=nownumber(2,1,1);
    unitofv(:,end,1)=nownumber(2,3,1);
    unitofv(:,1,end)=nownumber(2,1,3);

    unitofv(end,end,end)=nownumber(3,3,3);
    unitofv(1,end,end)=nownumber(1,3,3);
    unitofv(end,1,end)=nownumber(3,1,3);
    unitofv(end,end,1)=nownumber(3,3,1);
    unitofv(end,1,1)=nownumber(3,1,1);
    unitofv(1,end,1)=nownumber(1,3,1);
    unitofv(1,1,end)=nownumber(1,1,3);
    unitofv(1,1,1)=nownumber(1,1,1);
end

end