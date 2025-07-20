from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=169.433319091797, 
    height=238.933349609375)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)

for i in range(20):
    i = i+1
    rdnum = 4
    try:
        o1 = session.openOdb(
            name='D:/MCTS-AL/Round'+str(rdnum)+'/odb/'+str(i)+'new.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=o1)
        odb = session.odbs['D:/MCTS-AL/Round'+str(rdnum)+'/odb/'+str(i)+'new.odb']
        session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
            NODAL, ((COMPONENT, 'RF3'), )), ('U', NODAL, ((COMPONENT, 'U3'), )), ), 
            nodeSets=("TOPRP", ))
        xy1 = session.xyDataObjects['U:U3 PI: ASSEMBLY N: 1']
        xy2 = session.xyDataObjects['RF:RF3 PI: ASSEMBLY N: 1']
        xy3 = combine(-xy1, -xy2)
        xy3.setValues(
            sourceDescription='combine ( -"U:U3 PI: ASSEMBLY N: 1",-"RF:RF3 PI: ASSEMBLY N: 1" )')
        tmpName = xy3.name
        session.xyDataObjects.changeKey(tmpName, 'XYData-1')
        x0 = session.xyDataObjects['XYData-1']
        session.writeXYReport(fileName='D:/MCTS-AL/Round'+str(rdnum)+'/rpts/'+str(i)+'new.rpt', xyData=(x0, ))
        
        del session.xyDataObjects['U:U3 PI: ASSEMBLY N: 1']
        del session.xyDataObjects['RF:RF3 PI: ASSEMBLY N: 1']
        del session.xyDataObjects['XYData-1']
    except:
        continue