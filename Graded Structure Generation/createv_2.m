function v=createv_2(data,sizeofdata,nofv,dofv)
v=[];
% xdata=data(:,1); %size:27*1;
% ydata=data(:,2);
% zdata=data(:,3);
for k=1:sizeofdata(3) 
temp2=[];
    for j=1:sizeofdata(2)
    temp1=[];
        for i=1:sizeofdata(1)
            position=[i,j,k];
            varray=createunitofv(data,position,nofv,dofv); %21,3
            temp1=cat(1,temp1,varray);%[sizeofdata(1)*xdim,ydim,zdim]
        end
        temp2=cat(2,temp2,temp1);%[sizeofdata(1)*xdim,sizeofdata(2)*ydim,zdim]
    end
    v=cat(3,v,temp2);%[sizeofdata(1)*xdim,sizeofdata(2)*ydim,sizeofdata(3)*zdim]
end
end