import numpy as np
import math

#structure process
def findneighbour(inputdata,position):
    neighbourhoods=np.zeros((3,3,3))
    neighbourhoods[:,:,:]=np.nan
    r=len(inputdata)
    flag=0
    for i in range(r):
        if inputdata[i,0]==position[0] and inputdata[i,1]==position[1] and inputdata[i,2]==position[2]:
            flag=1
    if flag!=0:
        for i in range(r):
            dertax=inputdata[i,0]-position[0]
            dertay=inputdata[i,1]-position[1]
            dertaz=inputdata[i,2]-position[2]
            if abs(dertax)<=1 and abs(dertay)<=1 and abs(dertaz)<=1:
                neighbourhoods[int(dertax+1),int(dertay+1),int(dertaz+1)]=inputdata[i,3]
    return neighbourhoods

def createunitofv(datainput,positon,nofv,dofv):
    neibourhoods=findneighbour(datainput,positon)
    unitofv=np.ones((nofv-2*dofv,nofv-2*dofv,nofv-2*dofv))
    if not np.isnan(neibourhoods[1,1,1]):
        unitofv=unitofv*neibourhoods[1,1,1]
    else:
        unitofv=np.zeros((nofv,nofv,nofv))
        unitofv[:,:,:]=np.nan
        return unitofv
    if np.isnan(neibourhoods[2,1,1]):
        neibourhoods[2,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[0,1,1]):
        neibourhoods[0,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,2,1]):
        neibourhoods[1,2,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,0,1]):
        neibourhoods[1,0,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,2]):
        neibourhoods[1,1,2]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,0]):
        neibourhoods[1,1,0]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[2,2,1]):
        neibourhoods[2,2,1]=(neibourhoods[2,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[2,0,1]):
        neibourhoods[2,0,1]=(neibourhoods[2,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[0,2,1]):
        neibourhoods[0,2,1]=(neibourhoods[0,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[0,0,1]):
        neibourhoods[0,0,1]=(neibourhoods[0,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[2,1,2]):
        neibourhoods[2,1,2]=(neibourhoods[2,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[2,1,0]):
        neibourhoods[2,1,0]=(neibourhoods[2,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,1,2]):
        neibourhoods[0,1,2]=(neibourhoods[0,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[0,1,0]):
        neibourhoods[0,1,0]=(neibourhoods[0,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,2,2]):
        neibourhoods[1,2,2]=(neibourhoods[1,2,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,2,0]):
        neibourhoods[1,2,0]=(neibourhoods[1,2,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,0,2]):
        neibourhoods[1,0,2]=(neibourhoods[1,0,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,0,0]):
        neibourhoods[1,0,0]=(neibourhoods[1,0,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,0,0]):
        neibourhoods[0,0,0]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,0,0]):
        neibourhoods[2,0,0]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,2,0]):
        neibourhoods[0,2,0]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,0,2]):
        neibourhoods[0,0,2]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[0,2,2]):
        neibourhoods[0,2,2]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,0,2]):
        neibourhoods[2,0,2]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,2,0]):
        neibourhoods[2,2,0]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,2,2]):
        neibourhoods[2,2,2]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    for i in range(dofv):
        nownumber=neibourhoods[1,1,1]+i*(neibourhoods-neibourhoods[1,1,1])/(2*dofv+1)
        temp=np.zeros((1,nofv-2*dofv+2*i,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[2,1,1]
        unitofv=np.concatenate((unitofv,temp),axis=0)#x+
        temp[:,:,:]=nownumber[0,1,1]
        unitofv=np.concatenate((temp,unitofv),axis=0)#x-
        temp=np.zeros((nofv-2*dofv+2*i+2,1,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[1,2,1]
        unitofv=np.concatenate((unitofv,temp),axis=1)#y+
        temp[:,:,:]=nownumber[1,0,1]
        unitofv=np.concatenate((temp,unitofv),axis=1)#y-
        temp=np.zeros((nofv-2*dofv+2*i+2,nofv-2*dofv+2*i+2,1))
        temp[:,:,:]=nownumber[1,1,2]
        unitofv=np.concatenate((unitofv,temp),axis=2)#z+
        temp[:,:,:]=nownumber[1,1,0]
        unitofv=np.concatenate((temp,unitofv),axis=2)#z-
        unitofv[[-1],[-1],:]=nownumber[2,2,1]#x+,y+
        unitofv[0,0,:]=nownumber[0,0,1]#x-,y-
        unitofv[[-1],0,:]=nownumber[2,0,1]#x+,y-
        unitofv[0,[-1],:]=nownumber[0,2,1]#x,y+
        unitofv[[-1],:,[-1]]=nownumber[2,1,2]
        unitofv[0,:,0]=nownumber[0,1,0]
        unitofv[[-1],:,0]=nownumber[2,1,0]
        unitofv[0,:,[-1]]=nownumber[0,1,2]
        unitofv[:,[-1],[-1]]=nownumber[1,2,2]
        unitofv[:,0,0]=nownumber[1,0,0]
        unitofv[:,[-1],0]=nownumber[1,2,0]
        unitofv[:,0,[-1]]=nownumber[1,0,2]
        unitofv[[-1],[-1],[-1]]=nownumber[2,2,2]
        unitofv[0,[-1],[-1]]=nownumber[0,2,2]
        unitofv[[-1],0,[-1]]=nownumber[2,0,2]
        unitofv[[-1],[-1],0]=nownumber[2,2,0]
        unitofv[[-1],0,0]=nownumber[2,0,0]
        unitofv[0,[-1],0]=nownumber[0,2,0]
        unitofv[0,0,[-1]]=nownumber[0,0,2]
        unitofv[0,0,0]=nownumber[0,0,0]
    return unitofv

def createv_2(data,sizeofdata,nofv,dofv):
    v=[]
    for k in range(sizeofdata[2]):
        temp2=[]
        for j in range(sizeofdata[1]):
            temp1=[]
            for i in range(sizeofdata[0]):
                position=[i,j,k]
                varray=createunitofv(data,position,nofv,dofv)
                if i<1:
                    temp1=varray
                else:
                    temp1=np.concatenate((temp1,varray),axis=0)
            if j<1:
                temp2=temp1
            else:
                temp2=np.concatenate((temp2,temp1),axis=1)
        if k<1:
            v=temp2
        else:
            v=np.concatenate((v,temp2),axis=2)
    return v

def Voxelization(matrix, n_dim = 27):
    r1=np.zeros((n_dim,3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                r1[9*a+3*b+c,0]=a
                r1[9*a+3*b+c,1]=b
                r1[9*a+3*b+c,2]=c
    
    n_size=6
    n_accu=60
    sizeofdata0=[3,3,3]
    accu=20
    x_axis,y_axis,z_axis = np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu)
    x,y,z = np.meshgrid(x_axis, y_axis,z_axis)
    oo = (np.sin(math.pi*x) * np.cos(math.pi*y) + 
          np.sin(math.pi*y) * np.cos(math.pi*z) + 
          np.sin(math.pi*z) * np.cos(math.pi*x))
    
    value=[]
    N=len(matrix)
    finished=(10*(1-matrix).reshape(N,n_dim,1))*0.282-0.469

    for l in range(N):
        r2=finished[l]
        data0=np.concatenate((r1,r2),axis=1)
        v=createv_2(data0,sizeofdata0,accu,3)
        ov=oo+v
        value.append(ov)
    mat_voxelized = np.asarray(value)
    mat_voxelized = np.where(mat_voxelized < 0.9, 1, 0)
    
    return mat_voxelized


