function neighbourhoods=findneighbour(inputdata,position)
%findneighbour for a specific position
%inputdata:27*4, position:[i,j,k]
neighbourhoods=zeros(3,3,3);
neighbourhoods(:,:,:)=NaN;
[r,~]=size(inputdata); %inputdata = data0; size = [27,4], r = 27
flag=0;
for i=1:r
    if ((inputdata(i,1)==position(1))&(inputdata(i,2)==position(2))&(inputdata(i,3)==position(3))) %if x==x&y==y&z==z
        flag=1;
        id=i; %id of the position
    end
end
%for a specific position
%only for the (2,2,2) one, there is 27 choices
%for the (1,1,1) one, dertax = 0, 1, dertay = 0, 1, dertaz = 0, 1
%neighbourhoods(2/3, 2/3, 2/3) is not nan
%for the (3,3,3), dx = dy = dz = -1,0
%neighbourhoods(1/2, 1/2, 1/2) is not nan
if flag~=0
    for i=1:r
        % every position(i) deviation of the current position
        dertax=inputdata(i,1)-position(1); %delta x, delta y, delta z
        dertay=inputdata(i,2)-position(2);
        dertaz=inputdata(i,3)-position(3);
        if ((abs(dertax))<=1&(abs(dertay))<=1&(abs(dertaz))<=1) % positions near the specific position
            neighbourhoods(dertax+2,dertay+2,dertaz+2)=inputdata(i,4); %if it can find all the DIFFERENT matrix
            %neighbourhoods(position(1)+dertax,position(2)+dertay,position(3)+dertaz)
            %the inputdata(i,4) is from the E_v in 'random.m', and all the
            %probabilities are listed, can it find other ones?
        end
    end
end
end