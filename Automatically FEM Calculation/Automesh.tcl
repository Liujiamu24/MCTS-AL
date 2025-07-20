for {set i 1} {$i < 21} {incr i} {
hm_answernext yes
*deletemodel 
*viewset 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 10 10
*begin "version 2021.0.0.33  5-14-2025  20:18:32"
*createstringarray 17 " 0 penalty value              0.00    0.00    0.80    1.00   10.00" \
  "  1 min length        1 1.0  10.000   9.000   4.000   2.000   1.000    1   59    0" \
  "  2 max length        1 1.0  10.000  12.000  15.000  20.000  30.000    0   39    1" \
  "  3 aspect ratio      1 1.0   1.000   2.000   4.400   5.000  10.000    1   41    2" \
  "  4 warpage           1 1.0   0.000   5.000  13.000  15.000  30.000    1   56    3" \
  "  5 max angle quad    1 1.0  90.000 110.000 134.000 140.000 160.000    1   28    4" \
  "  6 min angle quad    1 1.0  90.000  70.000  46.000  40.000  20.000    1   61    5" \
  "  7 max angle tria    1 1.0  60.000  80.000 112.000 120.000 150.000    1   19    6" \
  "  8 min angle tria    1 1.0  60.000  50.000  34.000  30.000  15.000    1   22    7" \
  "  9 skew              1 1.0   0.000  10.000  34.000  40.000  70.000    1   46    8" \
  " 10 jacobian          1 1.0   1.000   0.900   0.700   0.600   0.300    0   57    9" \
  " 11 chordal dev       0 1.0   0.000   0.300   0.800   1.000   2.000    1   29   10" \
  " 12 taper             1 1.0   0.000   0.200   0.500   0.600   0.900    1   53   11" \
  " 13 % of trias        1 1.0   0.000   6.000  10.000  15.000  20.000    0    0   -1" \
  " 14 QI color legend            32      32       7       6       3           3   -1" \
  " 15 time_step         1      10.000                   0.010            0   59   12" \
  "   Global_solver 1"
*setqualitycriteria 1 17 0
*templatefileset "C:/Program Files/Altair/2021/hwdesktop/templates/feoutput/abaqus/standard.3d"
*menufilterset "*"
*menufilterdisable 
*viewset 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 10 10
*viewset 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 10 10
*menufont 2
*setsolverusessegmentsets 1
*ME_CoreBehaviorAdjust "allowable_actions_policy=TC_lite"
*elementtype 104 7
*elementtype 210 3
*elementtype 213 3
*elementtype 220 3
*elementtype 5 1
*elementtype 55 1
*elementtype 206 1
*elementtype 205 32
*elementtype 208 7
*elementtype 56 2
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*clearmark collections 1
*clearmark collections 2
*clearmark controllers 1
*loaddefaultattributevaluesfromxml 
*createstringarray 23 "Abaqus " "Standard3D " "ALESMOOTHINGS_DISPLAY_SKIP " \
  "EXTRANODES_DISPLAY_SKIP " "ACCELEROMETERS_DISPLAY_SKIP " "SOLVERMASSES_DISPLAY_SKIP " \
  "LOADCOLS_DISPLAY_SKIP " "RETRACTORS_DISPLAY_SKIP " "VECTORCOLS_DISPLAY_SKIP " \
  "SYSTCOLS_DISPLAY_SKIP " "PRIMITIVES_DISPLAY_SKIP " "BLOCKS_DISPLAY_SKIP " \
  "CROSSSECTION_DISPLAY_SKIP " "CONSTRAINEDRIGIDBODY_DISPLAY_SKIP " "ELEMENTCLUSTERS_DISPLAY_SKIP " \
  "RIGIDWALLS_DISPLAY_SKIP " "SLIPRINGS_DISPLAY_SKIP " "CONTACTSURF_DISPLAY_SKIP " \
  "\[DRIVE_MAPPING=\] " "IDRULES_SKIP " "BOM_SUBSYSTEMS_SKIP " "CREATE_PART_HIERARCHY " \
  "IMPORT_MATERIAL_METADATA "
*feinputwithdata2 "\#stl\\stl" "D:/MCTS-AL/Round2/STL/${i}.stl" 0 0 0 0 0 1 23 1 0
*drawlistresetstyle 
*createmark elements 1 "all"
*createmark elements 2
*shrinkwrapmesh elements 1 2 0.08 30 13 1 0.2 0 0 0 0
*clearmark elements 1
*clearmark systems 1
*retainmarkselections 0
*createstringarray 4 "HMBOMCOMMENTS_XML" "HMSUBSYSTEMCOMMENTS_XML" "HMMATCOMMENTS_XML" \
  "EXPORTIDS_SKIP"
hm_answernext yes
*feoutputwithdata "C:/Program Files/Altair/2021/hwdesktop/templates/feoutput/abaqus/standard.3d" "D:/MCTS-AL/Round2/inp_pre/${i}new.inp" 0 0 0 1 4
}
