// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func.func @main(%arg0 : i1) {

    %cst_0 = arc.constant 32837: ui16
    // CHECK-DAG: [[CST0:%[^ ]+]] = arc.constant 32837 : ui16
    %cst_1 = arc.constant 4294967295: ui32
    // CHECK-DAG: [[CST1:%[^ ]+]] = arc.constant 4294967295 : ui32
    %cst_2 = arc.constant -3539969642088554496: si64
    // CHECK-DAG: [[CST2:%[^ ]+]] = arc.constant -3539969642088554496 : si64
    %cst_3 = arc.constant -1: si16
    // CHECK-DAG: [[CST3:%[^ ]+]] = arc.constant -1 : si16
    %cst_4 = arc.constant 187: ui8
    // CHECK-DAG: [[CST4:%[^ ]+]] = arc.constant 187 : ui8
    %cst_5 = arc.constant 2377945821876453376: si64
    // CHECK-DAG: [[CST5:%[^ ]+]] = arc.constant 2377945821876453376 : si64
    %cst_6 = arc.constant -8842: si16
    // CHECK-DAG: [[CST6:%[^ ]+]] = arc.constant -8842 : si16
    %cst_7 = arc.constant -2147483648: si32
    // CHECK-DAG: [[CST7:%[^ ]+]] = arc.constant -2147483648 : si32
    %cst_8 = arc.constant 18: ui8
    // CHECK-DAG: [[CST8:%[^ ]+]] = arc.constant 18 : ui8
    %cst_9 = arc.constant 119: si8
    // CHECK-DAG: [[CST9:%[^ ]+]] = arc.constant 119 : si8
    %cst_10 = arc.constant 1896453325349533696: si64
    // CHECK-DAG: [[CST10:%[^ ]+]] = arc.constant 1896453325349533696 : si64
    %cst_11 = arc.constant 1375401618: ui32
    // CHECK-DAG: [[CST11:%[^ ]+]] = arc.constant 1375401618 : ui32
    %cst_12 = arc.constant -7326918711505242112: si64
    // CHECK-DAG: [[CST12:%[^ ]+]] = arc.constant -7326918711505242112 : si64
    %cst_13 = arc.constant 6941173544696946688: ui64
    // CHECK-DAG: [[CST13:%[^ ]+]] = arc.constant 6941173544696946688 : ui64
    %cst_14 = arc.constant -2651: si16
    // CHECK-DAG: [[CST14:%[^ ]+]] = arc.constant -2651 : si16
    %cst_15 = arc.constant 552779942755491839: ui64
    // CHECK-DAG: [[CST15:%[^ ]+]] = arc.constant 552779942755491839 : ui64
    %cst_16 = arc.constant 0: si64
    // CHECK-DAG: [[CST16:%[^ ]+]] = arc.constant 0 : si64
    %cst_17 = arc.constant 2919565677: ui32
    // CHECK-DAG: [[CST17:%[^ ]+]] = arc.constant 2919565677 : ui32
    %cst_18 = arc.constant 185: ui8
    // CHECK-DAG: [[CST18:%[^ ]+]] = arc.constant 185 : ui8
    %cst_19 = arc.constant -1199241696: si32
    // CHECK-DAG: [[CST19:%[^ ]+]] = arc.constant -1199241696 : si32
    %cst_20 = arc.constant 0: si8
    // CHECK-DAG: [[CST20:%[^ ]+]] = arc.constant 0 : si8
    %cst_21 = arc.constant 7049616374022545408: ui64
    // CHECK-DAG: [[CST21:%[^ ]+]] = arc.constant 7049616374022545408 : ui64
    %cst_22 = arc.constant 140: ui8
    // CHECK-DAG: [[CST22:%[^ ]+]] = arc.constant 140 : ui8
    %cst_23 = arc.constant 3183575273: ui32
    // CHECK-DAG: [[CST23:%[^ ]+]] = arc.constant 3183575273 : ui32
    %cst_24 = arc.constant 123: ui8
    // CHECK-DAG: [[CST24:%[^ ]+]] = arc.constant 123 : ui8
    %cst_25 = arc.constant 1073761281: si32
    // CHECK-DAG: [[CST25:%[^ ]+]] = arc.constant 1073761281 : si32
    %cst_26 = arc.constant 44280: ui16
    // CHECK-DAG: [[CST26:%[^ ]+]] = arc.constant 44280 : ui16
    %cst_27 = arc.constant 49: ui8
    // CHECK-DAG: [[CST27:%[^ ]+]] = arc.constant 49 : ui8
    %cst_28 = arc.constant -1162023820212101120: si64
    // CHECK-DAG: [[CST28:%[^ ]+]] = arc.constant -1162023820212101120 : si64
    %cst_29 = arc.constant 18002406960279658496: ui64
    // CHECK-DAG: [[CST29:%[^ ]+]] = arc.constant 18002406960279658496 : ui64
    %cst_30 = arc.constant -91: si8
    // CHECK-DAG: [[CST30:%[^ ]+]] = arc.constant -91 : si8
    %cst_31 = arc.constant -22393: si16
    // CHECK-DAG: [[CST31:%[^ ]+]] = arc.constant -22393 : si16
    %cst_32 = arc.constant 0: ui16
    // CHECK-DAG: [[CST32:%[^ ]+]] = arc.constant 0 : ui16
    %cst_33 = arc.constant 163: ui8
    // CHECK-DAG: [[CST33:%[^ ]+]] = arc.constant 163 : ui8
    %cst_34 = arc.constant 46164: ui16
    // CHECK-DAG: [[CST34:%[^ ]+]] = arc.constant 46164 : ui16
    %cst_35 = arc.constant 3216506733: ui32
    // CHECK-DAG: [[CST35:%[^ ]+]] = arc.constant 3216506733 : ui32
    %cst_36 = arc.constant 321515960029589504: ui64
    // CHECK-DAG: [[CST36:%[^ ]+]] = arc.constant 321515960029589504 : ui64
    %cst_37 = arc.constant 1075988493: si32
    // CHECK-DAG: [[CST37:%[^ ]+]] = arc.constant 1075988493 : si32
    %cst_38 = arc.constant 251: ui8
    // CHECK-DAG: [[CST38:%[^ ]+]] = arc.constant 251 : ui8
    %cst_39 = arc.constant 26390: ui16
    // CHECK-DAG: [[CST39:%[^ ]+]] = arc.constant 26390 : ui16
    %cst_40 = arc.constant 0: ui64
    // CHECK-DAG: [[CST40:%[^ ]+]] = arc.constant 0 : ui64
    %cst_41 = arc.constant 2188539585: ui32
    // CHECK-DAG: [[CST41:%[^ ]+]] = arc.constant 2188539585 : ui32
    %cst_42 = arc.constant -128: si8
    // CHECK-DAG: [[CST42:%[^ ]+]] = arc.constant -128 : si8
    %cst_43 = arc.constant -1889749006: si32
    // CHECK-DAG: [[CST43:%[^ ]+]] = arc.constant -1889749006 : si32
    %cst_44 = arc.constant 13329: ui16
    // CHECK-DAG: [[CST44:%[^ ]+]] = arc.constant 13329 : ui16
    %cst_45 = arc.constant 18191835084284211200: ui64
    // CHECK-DAG: [[CST45:%[^ ]+]] = arc.constant 18191835084284211200 : ui64
    %cst_46 = arc.constant 47: si8
    // CHECK-DAG: [[CST46:%[^ ]+]] = arc.constant 47 : si8
    %cst_47 = arc.constant 926549452: si32
    // CHECK-DAG: [[CST47:%[^ ]+]] = arc.constant 926549452 : si32
    %cst_48 = arc.constant 16470: si16
    // CHECK-DAG: [[CST48:%[^ ]+]] = arc.constant 16470 : si16
    %cst_49 = arc.constant 105: ui8
    // CHECK-DAG: [[CST49:%[^ ]+]] = arc.constant 105 : ui8
    %cst_50 = arc.constant 937013714: si32
    // CHECK-DAG: [[CST50:%[^ ]+]] = arc.constant 937013714 : si32
    %cst_51 = arc.constant -1: si32
    // CHECK-DAG: [[CST51:%[^ ]+]] = arc.constant -1 : si32
    %cst_52 = arc.constant 3217394157: ui32
    // CHECK-DAG: [[CST52:%[^ ]+]] = arc.constant 3217394157 : ui32
    %cst_53 = arc.constant -8926: si16
    // CHECK-DAG: [[CST53:%[^ ]+]] = arc.constant -8926 : si16
    %cst_54 = arc.constant 39: si8
    // CHECK-DAG: [[CST54:%[^ ]+]] = arc.constant 39 : si8
    %cst_55 = arc.constant 24: ui8
    // CHECK-DAG: [[CST55:%[^ ]+]] = arc.constant 24 : ui8
    %cst_56 = arc.constant -1883307028: si32
    // CHECK-DAG: [[CST56:%[^ ]+]] = arc.constant -1883307028 : si32
    %cst_57 = arc.constant 52207: ui16
    // CHECK-DAG: [[CST57:%[^ ]+]] = arc.constant 52207 : ui16
    %cst_58 = arc.constant 1219585769529212927: si64
    // CHECK-DAG: [[CST58:%[^ ]+]] = arc.constant 1219585769529212927 : si64
    %cst_59 = arc.constant -18957: si16
    // CHECK-DAG: [[CST59:%[^ ]+]] = arc.constant -18957 : si16
    %cst_60 = arc.constant -2435507771193565185: si64
    // CHECK-DAG: [[CST60:%[^ ]+]] = arc.constant -2435507771193565185 : si64
    %cst_61 = arc.constant -16471: si16
    // CHECK-DAG: [[CST61:%[^ ]+]] = arc.constant -16471 : si16
    %cst_62 = arc.constant 80: si8
    // CHECK-DAG: [[CST62:%[^ ]+]] = arc.constant 80 : si8
    %cst_63 = arc.constant 1068758568: ui32
    // CHECK-DAG: [[CST63:%[^ ]+]] = arc.constant 1068758568 : ui32
    %cst_64 = arc.constant 229: ui8
    // CHECK-DAG: [[CST64:%[^ ]+]] = arc.constant 229 : ui8
    %cst_65 = arc.constant 88: si8
    // CHECK-DAG: [[CST65:%[^ ]+]] = arc.constant 88 : si8
    %cst_66 = arc.constant 202: ui8
    // CHECK-DAG: [[CST66:%[^ ]+]] = arc.constant 202 : ui8
    %cst_67 = arc.constant 17154: ui16
    // CHECK-DAG: [[CST67:%[^ ]+]] = arc.constant 17154 : ui16
    %cst_68 = arc.constant 0: si16
    // CHECK-DAG: [[CST68:%[^ ]+]] = arc.constant 0 : si16
    %cst_69 = arc.constant 2650: si16
    // CHECK-DAG: [[CST69:%[^ ]+]] = arc.constant 2650 : si16
    %cst_70 = arc.constant 9223372036854775807: si64
    // CHECK-DAG: [[CST70:%[^ ]+]] = arc.constant 9223372036854775807 : si64
    %cst_71 = arc.constant 17158: ui16
    // CHECK-DAG: [[CST71:%[^ ]+]] = arc.constant 17158 : ui16
    %cst_72 = arc.constant 293898500: ui32
    // CHECK-DAG: [[CST72:%[^ ]+]] = arc.constant 293898500 : ui32
    %cst_73 = arc.constant 2482438085: ui32
    // CHECK-DAG: [[CST73:%[^ ]+]] = arc.constant 2482438085 : ui32
    %cst_74 = arc.constant -38: si8
    // CHECK-DAG: [[CST74:%[^ ]+]] = arc.constant -38 : si8
    %cst_75 = arc.constant 1199241695: si32
    // CHECK-DAG: [[CST75:%[^ ]+]] = arc.constant 1199241695 : si32
    %cst_76 = arc.constant -142934561: si32
    // CHECK-DAG: [[CST76:%[^ ]+]] = arc.constant -142934561 : si32
    %cst_77 = arc.constant 23645006699438080: ui64
    // CHECK-DAG: [[CST77:%[^ ]+]] = arc.constant 23645006699438080 : ui64
    %cst_78 = arc.constant -32768: si16
    // CHECK-DAG: [[CST78:%[^ ]+]] = arc.constant -32768 : si16
    %cst_79 = arc.constant -83: si8
    // CHECK-DAG: [[CST79:%[^ ]+]] = arc.constant -83 : si8
    %cst_80 = arc.constant 30117: si16
    // CHECK-DAG: [[CST80:%[^ ]+]] = arc.constant 30117 : si16
    %cst_81 = arc.constant 82: si8
    // CHECK-DAG: [[CST81:%[^ ]+]] = arc.constant 82 : si8
    %cst_82 = arc.constant 52206: ui16
    // CHECK-DAG: [[CST82:%[^ ]+]] = arc.constant 52206 : ui16
    %cst_83 = arc.constant 8: si8
    // CHECK-DAG: [[CST83:%[^ ]+]] = arc.constant 8 : si8
    %cst_84 = arc.constant 127: si8
    // CHECK-DAG: [[CST84:%[^ ]+]] = arc.constant 127 : si8
    %cst_85 = arc.constant -6787864265661210624: si64
    // CHECK-DAG: [[CST85:%[^ ]+]] = arc.constant -6787864265661210624 : si64
    %cst_86 = arc.constant 0: ui32
    // CHECK-DAG: [[CST86:%[^ ]+]] = arc.constant 0 : ui32
    %cst_87 = arc.constant 28607: ui16
    // CHECK-DAG: [[CST87:%[^ ]+]] = arc.constant 28607 : ui16
    %cst_88 = arc.constant 33818884: ui32
    // CHECK-DAG: [[CST88:%[^ ]+]] = arc.constant 33818884 : ui32
    %cst_89 = arc.constant -4927739357623543808: si64
    // CHECK-DAG: [[CST89:%[^ ]+]] = arc.constant -4927739357623543808 : si64
    %cst_90 = arc.constant 11453: ui16
    // CHECK-DAG: [[CST90:%[^ ]+]] = arc.constant 11453 : ui16
    %cst_91 = arc.constant 1850278074: ui32
    // CHECK-DAG: [[CST91:%[^ ]+]] = arc.constant 1850278074 : ui32
    %cst_92 = arc.constant 17893964130954059776: ui64
    // CHECK-DAG: [[CST92:%[^ ]+]] = arc.constant 17893964130954059776 : ui64
    %cst_93 = arc.constant 3963319931: ui32
    // CHECK-DAG: [[CST93:%[^ ]+]] = arc.constant 3963319931 : ui32
    %cst_94 = arc.constant 0: ui8
    // CHECK-DAG: [[CST94:%[^ ]+]] = arc.constant 0 : ui8
    %cst_95 = arc.constant 725369812184793088: si64
    // CHECK-DAG: [[CST95:%[^ ]+]] = arc.constant 725369812184793088 : si64
    %cst_96 = arc.constant 26: ui8
    // CHECK-DAG: [[CST96:%[^ ]+]] = arc.constant 26 : ui8
    %cst_97 = arc.constant 2147748165: ui32
    // CHECK-DAG: [[CST97:%[^ ]+]] = arc.constant 2147748165 : ui32
    %cst_98 = arc.constant 19371: ui16
    // CHECK-DAG: [[CST98:%[^ ]+]] = arc.constant 19371 : ui16
    %cst_99 = arc.constant 16388: si16
    // CHECK-DAG: [[CST99:%[^ ]+]] = arc.constant 16388 : si16
    %cst_100 = arc.constant 7323703681286311936: ui64
    // CHECK-DAG: [[CST100:%[^ ]+]] = arc.constant 7323703681286311936 : ui64
    %cst_101 = arc.constant 70: ui8
    // CHECK-DAG: [[CST101:%[^ ]+]] = arc.constant 70 : ui8
    %cst_102 = arc.constant 18446744073709551615: ui64
    // CHECK-DAG: [[CST102:%[^ ]+]] = arc.constant 18446744073709551615 : ui64
    %cst_103 = arc.constant -127467969: si32
    // CHECK-DAG: [[CST103:%[^ ]+]] = arc.constant -127467969 : si32
    %cst_104 = arc.constant 19370: ui16
    // CHECK-DAG: [[CST104:%[^ ]+]] = arc.constant 19370 : ui16
    %cst_105 = arc.constant -948241953: si32
    // CHECK-DAG: [[CST105:%[^ ]+]] = arc.constant -948241953 : si32
    %cst_106 = arc.constant 255: ui8
    // CHECK-DAG: [[CST106:%[^ ]+]] = arc.constant 255 : ui8
    %cst_107 = arc.constant 18168190077584773120: ui64
    // CHECK-DAG: [[CST107:%[^ ]+]] = arc.constant 18168190077584773120 : ui64
    %cst_108 = arc.constant -23843: si16
    // CHECK-DAG: [[CST108:%[^ ]+]] = arc.constant -23843 : si16
    %cst_109 = arc.constant -1201229250: si32
    // CHECK-DAG: [[CST109:%[^ ]+]] = arc.constant -1201229250 : si32
    %cst_110 = arc.constant -9223372036854775808: si64
    // CHECK-DAG: [[CST110:%[^ ]+]] = arc.constant -9223372036854775808 : si64
    %cst_111 = arc.constant 0: si32
    // CHECK-DAG: [[CST111:%[^ ]+]] = arc.constant 0 : si32
    %cst_112 = arc.constant -16298: si16
    // CHECK-DAG: [[CST112:%[^ ]+]] = arc.constant -16298 : si16
    %cst_113 = arc.constant 115: ui8
    // CHECK-DAG: [[CST113:%[^ ]+]] = arc.constant 115 : ui8
    %cst_114 = arc.constant -1219585769529212928: si64
    // CHECK-DAG: [[CST114:%[^ ]+]] = arc.constant -1219585769529212928 : si64
    %cst_115 = arc.constant 1210469933: si32
    // CHECK-DAG: [[CST115:%[^ ]+]] = arc.constant 1210469933 : si32
    %cst_116 = arc.constant 2147483647: si32
    // CHECK-DAG: [[CST116:%[^ ]+]] = arc.constant 2147483647 : si32
    %cst_117 = arc.constant 7326918711505242111: si64
    // CHECK-DAG: [[CST117:%[^ ]+]] = arc.constant 7326918711505242111 : si64
    %cst_118 = arc.constant -46: si8
    // CHECK-DAG: [[CST118:%[^ ]+]] = arc.constant -46 : si8
    %cst_119 = arc.constant -16382: si16
    // CHECK-DAG: [[CST119:%[^ ]+]] = arc.constant -16382 : si16
    %cst_120 = arc.constant 39145: ui16
    // CHECK-DAG: [[CST120:%[^ ]+]] = arc.constant 39145 : ui16
    %cst_121 = arc.constant 21792: si16
    // CHECK-DAG: [[CST121:%[^ ]+]] = arc.constant 21792 : si16
    %cst_122 = arc.constant 11061233415582711808: ui64
    // CHECK-DAG: [[CST122:%[^ ]+]] = arc.constant 11061233415582711808 : ui64
    %cst_123 = arc.constant -813760513: si32
    // CHECK-DAG: [[CST123:%[^ ]+]] = arc.constant -813760513 : si32
    %cst_124 = arc.constant 18125228113679962111: ui64
    // CHECK-DAG: [[CST124:%[^ ]+]] = arc.constant 18125228113679962111 : ui64
    %cst_125 = arc.constant -1: si64
    // CHECK-DAG: [[CST125:%[^ ]+]] = arc.constant -1 : si64
    %cst_126 = arc.constant 61438: ui16
    // CHECK-DAG: [[CST126:%[^ ]+]] = arc.constant 61438 : ui16
    %cst_127 = arc.constant 32767: si16
    // CHECK-DAG: [[CST127:%[^ ]+]] = arc.constant 32767 : si16
    %cst_128 = arc.constant 2435507771193565184: si64
    // CHECK-DAG: [[CST128:%[^ ]+]] = arc.constant 2435507771193565184 : si64
    %cst_129 = arc.constant -40: si8
    // CHECK-DAG: [[CST129:%[^ ]+]] = arc.constant -40 : si8
    %cst_130 = arc.constant 65535: ui16
    // CHECK-DAG: [[CST130:%[^ ]+]] = arc.constant 65535 : ui16
    %cst_131 = arc.constant 9174869780490303488: si64
    // CHECK-DAG: [[CST131:%[^ ]+]] = arc.constant 9174869780490303488 : si64
    %cst_132 = arc.constant -2569: si16
    // CHECK-DAG: [[CST132:%[^ ]+]] = arc.constant -2569 : si16
    %cst_133 = arc.constant -9: si8
    // CHECK-DAG: [[CST133:%[^ ]+]] = arc.constant -9 : si8
    %cst_134 = arc.constant -264176621: si32
    // CHECK-DAG: [[CST134:%[^ ]+]] = arc.constant -264176621 : si32
    %cst_135 = arc.constant 18164208655933440: si64
    // CHECK-DAG: [[CST135:%[^ ]+]] = arc.constant 18164208655933440 : si64
    %cst_136 = arc.constant -4909575148967610368: si64
    // CHECK-DAG: [[CST136:%[^ ]+]] = arc.constant -4909575148967610368 : si64
    %cst_137 = arc.constant 7540: si16
    // CHECK-DAG: [[CST137:%[^ ]+]] = arc.constant 7540 : si16
    %cst_138 = arc.constant -118: si8
    // CHECK-DAG: [[CST138:%[^ ]+]] = arc.constant -118 : si8
    %cst_139 = arc.constant -1069484013: si32
    // CHECK-DAG: [[CST139:%[^ ]+]] = arc.constant -1069484013 : si32
    %cst_140 = arc.constant 1883307027: si32
    // CHECK-DAG: [[CST140:%[^ ]+]] = arc.constant 1883307027 : si32
    %cst_141 = arc.constant -1: si8
    // CHECK-DAG: [[CST141:%[^ ]+]] = arc.constant -1 : si8
    %cst_142 = arc.constant 8449499968305510400: si64
    // CHECK-DAG: [[CST142:%[^ ]+]] = arc.constant 8449499968305510400 : si64
    %cst_143 = arc.constant 23842: si16
    // CHECK-DAG: [[CST143:%[^ ]+]] = arc.constant 23842 : si16
    %cst_144 = arc.constant 2444689221: ui32
    // CHECK-DAG: [[CST144:%[^ ]+]] = arc.constant 2444689221 : ui32
    %cst_145 = arc.constant -937013715: si32
    // CHECK-DAG: [[CST145:%[^ ]+]] = arc.constant -937013715 : si32
    %cst_146 = arc.constant 7347418007669223424: ui64
    // CHECK-DAG: [[CST146:%[^ ]+]] = arc.constant 7347418007669223424 : ui64
    %cst_147 = arc.constant -601: si16
    // CHECK-DAG: [[CST147:%[^ ]+]] = arc.constant -601 : si16
    %cst_148 = arc.constant 8003786267325562880: si64
    // CHECK-DAG: [[CST148:%[^ ]+]] = arc.constant 8003786267325562880 : si64
    %cst_149 = arc.constant 11397127699687006207: ui64
    // CHECK-DAG: [[CST149:%[^ ]+]] = arc.constant 11397127699687006207 : ui64
    %cst_150 = arc.constant 23714326382911488: ui64
    // CHECK-DAG: [[CST150:%[^ ]+]] = arc.constant 23714326382911488 : ui64
    %cst_151 = arc.constant 331647364: ui32
    // CHECK-DAG: [[CST151:%[^ ]+]] = arc.constant 331647364 : ui32
    %result_and_96_96 = arc.and %cst_96, %cst_96 : ui8
    "arc.keep"(%result_and_96_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_or_96_96 = arc.or %cst_96, %cst_96 : ui8
    "arc.keep"(%result_or_96_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_xor_96_96 = arc.xor %cst_96, %cst_96 : ui8
    "arc.keep"(%result_xor_96_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_and_96_94 = arc.and %cst_96, %cst_94 : ui8
    "arc.keep"(%result_and_96_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_96_94 = arc.or %cst_96, %cst_94 : ui8
    "arc.keep"(%result_or_96_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_xor_96_94 = arc.xor %cst_96, %cst_94 : ui8
    "arc.keep"(%result_xor_96_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_and_96_113 = arc.and %cst_96, %cst_113 : ui8
    "arc.keep"(%result_and_96_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST8]]) : (ui8) -> ()
    %result_or_96_113 = arc.or %cst_96, %cst_113 : ui8
    "arc.keep"(%result_or_96_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST24]]) : (ui8) -> ()
    %result_xor_96_113 = arc.xor %cst_96, %cst_113 : ui8
    "arc.keep"(%result_xor_96_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST49]]) : (ui8) -> ()
    %result_and_96_18 = arc.and %cst_96, %cst_18 : ui8
    "arc.keep"(%result_and_96_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST55]]) : (ui8) -> ()
    %result_or_96_18 = arc.or %cst_96, %cst_18 : ui8
    "arc.keep"(%result_or_96_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST4]]) : (ui8) -> ()
    %result_xor_96_18 = arc.xor %cst_96, %cst_18 : ui8
    "arc.keep"(%result_xor_96_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST33]]) : (ui8) -> ()
    %result_and_96_106 = arc.and %cst_96, %cst_106 : ui8
    "arc.keep"(%result_and_96_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_or_96_106 = arc.or %cst_96, %cst_106 : ui8
    "arc.keep"(%result_or_96_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_96_106 = arc.xor %cst_96, %cst_106 : ui8
    "arc.keep"(%result_xor_96_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST64]]) : (ui8) -> ()
    %result_and_94_96 = arc.and %cst_94, %cst_96 : ui8
    "arc.keep"(%result_and_94_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_94_96 = arc.or %cst_94, %cst_96 : ui8
    "arc.keep"(%result_or_94_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_xor_94_96 = arc.xor %cst_94, %cst_96 : ui8
    "arc.keep"(%result_xor_94_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_and_94_94 = arc.and %cst_94, %cst_94 : ui8
    "arc.keep"(%result_and_94_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_94_94 = arc.or %cst_94, %cst_94 : ui8
    "arc.keep"(%result_or_94_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_xor_94_94 = arc.xor %cst_94, %cst_94 : ui8
    "arc.keep"(%result_xor_94_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_and_94_113 = arc.and %cst_94, %cst_113 : ui8
    "arc.keep"(%result_and_94_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_94_113 = arc.or %cst_94, %cst_113 : ui8
    "arc.keep"(%result_or_94_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_xor_94_113 = arc.xor %cst_94, %cst_113 : ui8
    "arc.keep"(%result_xor_94_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_and_94_18 = arc.and %cst_94, %cst_18 : ui8
    "arc.keep"(%result_and_94_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_94_18 = arc.or %cst_94, %cst_18 : ui8
    "arc.keep"(%result_or_94_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_xor_94_18 = arc.xor %cst_94, %cst_18 : ui8
    "arc.keep"(%result_xor_94_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_and_94_106 = arc.and %cst_94, %cst_106 : ui8
    "arc.keep"(%result_and_94_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_94_106 = arc.or %cst_94, %cst_106 : ui8
    "arc.keep"(%result_or_94_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_94_106 = arc.xor %cst_94, %cst_106 : ui8
    "arc.keep"(%result_xor_94_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_and_113_96 = arc.and %cst_113, %cst_96 : ui8
    "arc.keep"(%result_and_113_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST8]]) : (ui8) -> ()
    %result_or_113_96 = arc.or %cst_113, %cst_96 : ui8
    "arc.keep"(%result_or_113_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST24]]) : (ui8) -> ()
    %result_xor_113_96 = arc.xor %cst_113, %cst_96 : ui8
    "arc.keep"(%result_xor_113_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST49]]) : (ui8) -> ()
    %result_and_113_94 = arc.and %cst_113, %cst_94 : ui8
    "arc.keep"(%result_and_113_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_113_94 = arc.or %cst_113, %cst_94 : ui8
    "arc.keep"(%result_or_113_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_xor_113_94 = arc.xor %cst_113, %cst_94 : ui8
    "arc.keep"(%result_xor_113_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_and_113_113 = arc.and %cst_113, %cst_113 : ui8
    "arc.keep"(%result_and_113_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_or_113_113 = arc.or %cst_113, %cst_113 : ui8
    "arc.keep"(%result_or_113_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_xor_113_113 = arc.xor %cst_113, %cst_113 : ui8
    "arc.keep"(%result_xor_113_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_and_113_18 = arc.and %cst_113, %cst_18 : ui8
    "arc.keep"(%result_and_113_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST27]]) : (ui8) -> ()
    %result_or_113_18 = arc.or %cst_113, %cst_18 : ui8
    "arc.keep"(%result_or_113_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST38]]) : (ui8) -> ()
    %result_xor_113_18 = arc.xor %cst_113, %cst_18 : ui8
    "arc.keep"(%result_xor_113_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST66]]) : (ui8) -> ()
    %result_and_113_106 = arc.and %cst_113, %cst_106 : ui8
    "arc.keep"(%result_and_113_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_or_113_106 = arc.or %cst_113, %cst_106 : ui8
    "arc.keep"(%result_or_113_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_113_106 = arc.xor %cst_113, %cst_106 : ui8
    "arc.keep"(%result_xor_113_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST22]]) : (ui8) -> ()
    %result_and_18_96 = arc.and %cst_18, %cst_96 : ui8
    "arc.keep"(%result_and_18_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST55]]) : (ui8) -> ()
    %result_or_18_96 = arc.or %cst_18, %cst_96 : ui8
    "arc.keep"(%result_or_18_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST4]]) : (ui8) -> ()
    %result_xor_18_96 = arc.xor %cst_18, %cst_96 : ui8
    "arc.keep"(%result_xor_18_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST33]]) : (ui8) -> ()
    %result_and_18_94 = arc.and %cst_18, %cst_94 : ui8
    "arc.keep"(%result_and_18_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_18_94 = arc.or %cst_18, %cst_94 : ui8
    "arc.keep"(%result_or_18_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_xor_18_94 = arc.xor %cst_18, %cst_94 : ui8
    "arc.keep"(%result_xor_18_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_and_18_113 = arc.and %cst_18, %cst_113 : ui8
    "arc.keep"(%result_and_18_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST27]]) : (ui8) -> ()
    %result_or_18_113 = arc.or %cst_18, %cst_113 : ui8
    "arc.keep"(%result_or_18_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST38]]) : (ui8) -> ()
    %result_xor_18_113 = arc.xor %cst_18, %cst_113 : ui8
    "arc.keep"(%result_xor_18_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST66]]) : (ui8) -> ()
    %result_and_18_18 = arc.and %cst_18, %cst_18 : ui8
    "arc.keep"(%result_and_18_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_or_18_18 = arc.or %cst_18, %cst_18 : ui8
    "arc.keep"(%result_or_18_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_xor_18_18 = arc.xor %cst_18, %cst_18 : ui8
    "arc.keep"(%result_xor_18_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_and_18_106 = arc.and %cst_18, %cst_106 : ui8
    "arc.keep"(%result_and_18_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_or_18_106 = arc.or %cst_18, %cst_106 : ui8
    "arc.keep"(%result_or_18_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_18_106 = arc.xor %cst_18, %cst_106 : ui8
    "arc.keep"(%result_xor_18_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST101]]) : (ui8) -> ()
    %result_and_106_96 = arc.and %cst_106, %cst_96 : ui8
    "arc.keep"(%result_and_106_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST96]]) : (ui8) -> ()
    %result_or_106_96 = arc.or %cst_106, %cst_96 : ui8
    "arc.keep"(%result_or_106_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_106_96 = arc.xor %cst_106, %cst_96 : ui8
    "arc.keep"(%result_xor_106_96) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST64]]) : (ui8) -> ()
    %result_and_106_94 = arc.and %cst_106, %cst_94 : ui8
    "arc.keep"(%result_and_106_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_or_106_94 = arc.or %cst_106, %cst_94 : ui8
    "arc.keep"(%result_or_106_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_106_94 = arc.xor %cst_106, %cst_94 : ui8
    "arc.keep"(%result_xor_106_94) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_and_106_113 = arc.and %cst_106, %cst_113 : ui8
    "arc.keep"(%result_and_106_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST113]]) : (ui8) -> ()
    %result_or_106_113 = arc.or %cst_106, %cst_113 : ui8
    "arc.keep"(%result_or_106_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_106_113 = arc.xor %cst_106, %cst_113 : ui8
    "arc.keep"(%result_xor_106_113) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST22]]) : (ui8) -> ()
    %result_and_106_18 = arc.and %cst_106, %cst_18 : ui8
    "arc.keep"(%result_and_106_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST18]]) : (ui8) -> ()
    %result_or_106_18 = arc.or %cst_106, %cst_18 : ui8
    "arc.keep"(%result_or_106_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_106_18 = arc.xor %cst_106, %cst_18 : ui8
    "arc.keep"(%result_xor_106_18) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST101]]) : (ui8) -> ()
    %result_and_106_106 = arc.and %cst_106, %cst_106 : ui8
    "arc.keep"(%result_and_106_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_or_106_106 = arc.or %cst_106, %cst_106 : ui8
    "arc.keep"(%result_or_106_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST106]]) : (ui8) -> ()
    %result_xor_106_106 = arc.xor %cst_106, %cst_106 : ui8
    "arc.keep"(%result_xor_106_106) : (ui8) -> ()
    // CHECK: "arc.keep"([[CST94]]) : (ui8) -> ()
    %result_and_118_118 = arc.and %cst_118, %cst_118 : si8
    "arc.keep"(%result_and_118_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_or_118_118 = arc.or %cst_118, %cst_118 : si8
    "arc.keep"(%result_or_118_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_xor_118_118 = arc.xor %cst_118, %cst_118 : si8
    "arc.keep"(%result_xor_118_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_118_65 = arc.and %cst_118, %cst_65 : si8
    "arc.keep"(%result_and_118_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST62]]) : (si8) -> ()
    %result_or_118_65 = arc.or %cst_118, %cst_65 : si8
    "arc.keep"(%result_or_118_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST74]]) : (si8) -> ()
    %result_xor_118_65 = arc.xor %cst_118, %cst_65 : si8
    "arc.keep"(%result_xor_118_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST138]]) : (si8) -> ()
    %result_and_118_9 = arc.and %cst_118, %cst_9 : si8
    "arc.keep"(%result_and_118_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_or_118_9 = arc.or %cst_118, %cst_9 : si8
    "arc.keep"(%result_or_118_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_xor_118_9 = arc.xor %cst_118, %cst_9 : si8
    "arc.keep"(%result_xor_118_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST30]]) : (si8) -> ()
    %result_and_118_20 = arc.and %cst_118, %cst_20 : si8
    "arc.keep"(%result_and_118_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_118_20 = arc.or %cst_118, %cst_20 : si8
    "arc.keep"(%result_or_118_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_xor_118_20 = arc.xor %cst_118, %cst_20 : si8
    "arc.keep"(%result_xor_118_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_and_118_42 = arc.and %cst_118, %cst_42 : si8
    "arc.keep"(%result_and_118_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_or_118_42 = arc.or %cst_118, %cst_42 : si8
    "arc.keep"(%result_or_118_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_xor_118_42 = arc.xor %cst_118, %cst_42 : si8
    "arc.keep"(%result_xor_118_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_and_118_84 = arc.and %cst_118, %cst_84 : si8
    "arc.keep"(%result_and_118_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_or_118_84 = arc.or %cst_118, %cst_84 : si8
    "arc.keep"(%result_or_118_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_xor_118_84 = arc.xor %cst_118, %cst_84 : si8
    "arc.keep"(%result_xor_118_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST79]]) : (si8) -> ()
    %result_and_65_118 = arc.and %cst_65, %cst_118 : si8
    "arc.keep"(%result_and_65_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST62]]) : (si8) -> ()
    %result_or_65_118 = arc.or %cst_65, %cst_118 : si8
    "arc.keep"(%result_or_65_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST74]]) : (si8) -> ()
    %result_xor_65_118 = arc.xor %cst_65, %cst_118 : si8
    "arc.keep"(%result_xor_65_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST138]]) : (si8) -> ()
    %result_and_65_65 = arc.and %cst_65, %cst_65 : si8
    "arc.keep"(%result_and_65_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_or_65_65 = arc.or %cst_65, %cst_65 : si8
    "arc.keep"(%result_or_65_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_xor_65_65 = arc.xor %cst_65, %cst_65 : si8
    "arc.keep"(%result_xor_65_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_65_9 = arc.and %cst_65, %cst_9 : si8
    "arc.keep"(%result_and_65_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST62]]) : (si8) -> ()
    %result_or_65_9 = arc.or %cst_65, %cst_9 : si8
    "arc.keep"(%result_or_65_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_65_9 = arc.xor %cst_65, %cst_9 : si8
    "arc.keep"(%result_xor_65_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST46]]) : (si8) -> ()
    %result_and_65_20 = arc.and %cst_65, %cst_20 : si8
    "arc.keep"(%result_and_65_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_65_20 = arc.or %cst_65, %cst_20 : si8
    "arc.keep"(%result_or_65_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_xor_65_20 = arc.xor %cst_65, %cst_20 : si8
    "arc.keep"(%result_xor_65_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_and_65_42 = arc.and %cst_65, %cst_42 : si8
    "arc.keep"(%result_and_65_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_65_42 = arc.or %cst_65, %cst_42 : si8
    "arc.keep"(%result_or_65_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST129]]) : (si8) -> ()
    %result_xor_65_42 = arc.xor %cst_65, %cst_42 : si8
    "arc.keep"(%result_xor_65_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST129]]) : (si8) -> ()
    %result_and_65_84 = arc.and %cst_65, %cst_84 : si8
    "arc.keep"(%result_and_65_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_or_65_84 = arc.or %cst_65, %cst_84 : si8
    "arc.keep"(%result_or_65_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_65_84 = arc.xor %cst_65, %cst_84 : si8
    "arc.keep"(%result_xor_65_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST54]]) : (si8) -> ()
    %result_and_9_118 = arc.and %cst_9, %cst_118 : si8
    "arc.keep"(%result_and_9_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_or_9_118 = arc.or %cst_9, %cst_118 : si8
    "arc.keep"(%result_or_9_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_xor_9_118 = arc.xor %cst_9, %cst_118 : si8
    "arc.keep"(%result_xor_9_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST30]]) : (si8) -> ()
    %result_and_9_65 = arc.and %cst_9, %cst_65 : si8
    "arc.keep"(%result_and_9_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST62]]) : (si8) -> ()
    %result_or_9_65 = arc.or %cst_9, %cst_65 : si8
    "arc.keep"(%result_or_9_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_9_65 = arc.xor %cst_9, %cst_65 : si8
    "arc.keep"(%result_xor_9_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST46]]) : (si8) -> ()
    %result_and_9_9 = arc.and %cst_9, %cst_9 : si8
    "arc.keep"(%result_and_9_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_or_9_9 = arc.or %cst_9, %cst_9 : si8
    "arc.keep"(%result_or_9_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_xor_9_9 = arc.xor %cst_9, %cst_9 : si8
    "arc.keep"(%result_xor_9_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_9_20 = arc.and %cst_9, %cst_20 : si8
    "arc.keep"(%result_and_9_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_9_20 = arc.or %cst_9, %cst_20 : si8
    "arc.keep"(%result_or_9_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_xor_9_20 = arc.xor %cst_9, %cst_20 : si8
    "arc.keep"(%result_xor_9_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_and_9_42 = arc.and %cst_9, %cst_42 : si8
    "arc.keep"(%result_and_9_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_9_42 = arc.or %cst_9, %cst_42 : si8
    "arc.keep"(%result_or_9_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_xor_9_42 = arc.xor %cst_9, %cst_42 : si8
    "arc.keep"(%result_xor_9_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_and_9_84 = arc.and %cst_9, %cst_84 : si8
    "arc.keep"(%result_and_9_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_or_9_84 = arc.or %cst_9, %cst_84 : si8
    "arc.keep"(%result_or_9_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_9_84 = arc.xor %cst_9, %cst_84 : si8
    "arc.keep"(%result_xor_9_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST83]]) : (si8) -> ()
    %result_and_20_118 = arc.and %cst_20, %cst_118 : si8
    "arc.keep"(%result_and_20_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_118 = arc.or %cst_20, %cst_118 : si8
    "arc.keep"(%result_or_20_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_xor_20_118 = arc.xor %cst_20, %cst_118 : si8
    "arc.keep"(%result_xor_20_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_and_20_65 = arc.and %cst_20, %cst_65 : si8
    "arc.keep"(%result_and_20_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_65 = arc.or %cst_20, %cst_65 : si8
    "arc.keep"(%result_or_20_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_xor_20_65 = arc.xor %cst_20, %cst_65 : si8
    "arc.keep"(%result_xor_20_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_and_20_9 = arc.and %cst_20, %cst_9 : si8
    "arc.keep"(%result_and_20_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_9 = arc.or %cst_20, %cst_9 : si8
    "arc.keep"(%result_or_20_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_xor_20_9 = arc.xor %cst_20, %cst_9 : si8
    "arc.keep"(%result_xor_20_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_and_20_20 = arc.and %cst_20, %cst_20 : si8
    "arc.keep"(%result_and_20_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_20 = arc.or %cst_20, %cst_20 : si8
    "arc.keep"(%result_or_20_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_xor_20_20 = arc.xor %cst_20, %cst_20 : si8
    "arc.keep"(%result_xor_20_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_20_42 = arc.and %cst_20, %cst_42 : si8
    "arc.keep"(%result_and_20_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_42 = arc.or %cst_20, %cst_42 : si8
    "arc.keep"(%result_or_20_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_xor_20_42 = arc.xor %cst_20, %cst_42 : si8
    "arc.keep"(%result_xor_20_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_and_20_84 = arc.and %cst_20, %cst_84 : si8
    "arc.keep"(%result_and_20_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_20_84 = arc.or %cst_20, %cst_84 : si8
    "arc.keep"(%result_or_20_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_20_84 = arc.xor %cst_20, %cst_84 : si8
    "arc.keep"(%result_xor_20_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_and_42_118 = arc.and %cst_42, %cst_118 : si8
    "arc.keep"(%result_and_42_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_or_42_118 = arc.or %cst_42, %cst_118 : si8
    "arc.keep"(%result_or_42_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST118]]) : (si8) -> ()
    %result_xor_42_118 = arc.xor %cst_42, %cst_118 : si8
    "arc.keep"(%result_xor_42_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_and_42_65 = arc.and %cst_42, %cst_65 : si8
    "arc.keep"(%result_and_42_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_42_65 = arc.or %cst_42, %cst_65 : si8
    "arc.keep"(%result_or_42_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST129]]) : (si8) -> ()
    %result_xor_42_65 = arc.xor %cst_42, %cst_65 : si8
    "arc.keep"(%result_xor_42_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST129]]) : (si8) -> ()
    %result_and_42_9 = arc.and %cst_42, %cst_9 : si8
    "arc.keep"(%result_and_42_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_42_9 = arc.or %cst_42, %cst_9 : si8
    "arc.keep"(%result_or_42_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_xor_42_9 = arc.xor %cst_42, %cst_9 : si8
    "arc.keep"(%result_xor_42_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST133]]) : (si8) -> ()
    %result_and_42_20 = arc.and %cst_42, %cst_20 : si8
    "arc.keep"(%result_and_42_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_42_20 = arc.or %cst_42, %cst_20 : si8
    "arc.keep"(%result_or_42_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_xor_42_20 = arc.xor %cst_42, %cst_20 : si8
    "arc.keep"(%result_xor_42_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_and_42_42 = arc.and %cst_42, %cst_42 : si8
    "arc.keep"(%result_and_42_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_or_42_42 = arc.or %cst_42, %cst_42 : si8
    "arc.keep"(%result_or_42_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST42]]) : (si8) -> ()
    %result_xor_42_42 = arc.xor %cst_42, %cst_42 : si8
    "arc.keep"(%result_xor_42_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_42_84 = arc.and %cst_42, %cst_84 : si8
    "arc.keep"(%result_and_42_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_42_84 = arc.or %cst_42, %cst_84 : si8
    "arc.keep"(%result_or_42_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_xor_42_84 = arc.xor %cst_42, %cst_84 : si8
    "arc.keep"(%result_xor_42_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_and_84_118 = arc.and %cst_84, %cst_118 : si8
    "arc.keep"(%result_and_84_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST81]]) : (si8) -> ()
    %result_or_84_118 = arc.or %cst_84, %cst_118 : si8
    "arc.keep"(%result_or_84_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_xor_84_118 = arc.xor %cst_84, %cst_118 : si8
    "arc.keep"(%result_xor_84_118) : (si8) -> ()
    // CHECK: "arc.keep"([[CST79]]) : (si8) -> ()
    %result_and_84_65 = arc.and %cst_84, %cst_65 : si8
    "arc.keep"(%result_and_84_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST65]]) : (si8) -> ()
    %result_or_84_65 = arc.or %cst_84, %cst_65 : si8
    "arc.keep"(%result_or_84_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_84_65 = arc.xor %cst_84, %cst_65 : si8
    "arc.keep"(%result_xor_84_65) : (si8) -> ()
    // CHECK: "arc.keep"([[CST54]]) : (si8) -> ()
    %result_and_84_9 = arc.and %cst_84, %cst_9 : si8
    "arc.keep"(%result_and_84_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST9]]) : (si8) -> ()
    %result_or_84_9 = arc.or %cst_84, %cst_9 : si8
    "arc.keep"(%result_or_84_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_84_9 = arc.xor %cst_84, %cst_9 : si8
    "arc.keep"(%result_xor_84_9) : (si8) -> ()
    // CHECK: "arc.keep"([[CST83]]) : (si8) -> ()
    %result_and_84_20 = arc.and %cst_84, %cst_20 : si8
    "arc.keep"(%result_and_84_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_84_20 = arc.or %cst_84, %cst_20 : si8
    "arc.keep"(%result_or_84_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_84_20 = arc.xor %cst_84, %cst_20 : si8
    "arc.keep"(%result_xor_84_20) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_and_84_42 = arc.and %cst_84, %cst_42 : si8
    "arc.keep"(%result_and_84_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_or_84_42 = arc.or %cst_84, %cst_42 : si8
    "arc.keep"(%result_or_84_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_xor_84_42 = arc.xor %cst_84, %cst_42 : si8
    "arc.keep"(%result_xor_84_42) : (si8) -> ()
    // CHECK: "arc.keep"([[CST141]]) : (si8) -> ()
    %result_and_84_84 = arc.and %cst_84, %cst_84 : si8
    "arc.keep"(%result_and_84_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_or_84_84 = arc.or %cst_84, %cst_84 : si8
    "arc.keep"(%result_or_84_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST84]]) : (si8) -> ()
    %result_xor_84_84 = arc.xor %cst_84, %cst_84 : si8
    "arc.keep"(%result_xor_84_84) : (si8) -> ()
    // CHECK: "arc.keep"([[CST20]]) : (si8) -> ()
    %result_and_32_32 = arc.and %cst_32, %cst_32 : ui16
    "arc.keep"(%result_and_32_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_32_32 = arc.or %cst_32, %cst_32 : ui16
    "arc.keep"(%result_or_32_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_xor_32_32 = arc.xor %cst_32, %cst_32 : ui16
    "arc.keep"(%result_xor_32_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_and_32_130 = arc.and %cst_32, %cst_130 : ui16
    "arc.keep"(%result_and_32_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_32_130 = arc.or %cst_32, %cst_130 : ui16
    "arc.keep"(%result_or_32_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_32_130 = arc.xor %cst_32, %cst_130 : ui16
    "arc.keep"(%result_xor_32_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_and_32_39 = arc.and %cst_32, %cst_39 : ui16
    "arc.keep"(%result_and_32_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_32_39 = arc.or %cst_32, %cst_39 : ui16
    "arc.keep"(%result_or_32_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_xor_32_39 = arc.xor %cst_32, %cst_39 : ui16
    "arc.keep"(%result_xor_32_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_and_32_98 = arc.and %cst_32, %cst_98 : ui16
    "arc.keep"(%result_and_32_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_32_98 = arc.or %cst_32, %cst_98 : ui16
    "arc.keep"(%result_or_32_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_xor_32_98 = arc.xor %cst_32, %cst_98 : ui16
    "arc.keep"(%result_xor_32_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_and_32_82 = arc.and %cst_32, %cst_82 : ui16
    "arc.keep"(%result_and_32_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_32_82 = arc.or %cst_32, %cst_82 : ui16
    "arc.keep"(%result_or_32_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_xor_32_82 = arc.xor %cst_32, %cst_82 : ui16
    "arc.keep"(%result_xor_32_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_and_130_32 = arc.and %cst_130, %cst_32 : ui16
    "arc.keep"(%result_and_130_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_130_32 = arc.or %cst_130, %cst_32 : ui16
    "arc.keep"(%result_or_130_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_130_32 = arc.xor %cst_130, %cst_32 : ui16
    "arc.keep"(%result_xor_130_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_and_130_130 = arc.and %cst_130, %cst_130 : ui16
    "arc.keep"(%result_and_130_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_or_130_130 = arc.or %cst_130, %cst_130 : ui16
    "arc.keep"(%result_or_130_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_130_130 = arc.xor %cst_130, %cst_130 : ui16
    "arc.keep"(%result_xor_130_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_and_130_39 = arc.and %cst_130, %cst_39 : ui16
    "arc.keep"(%result_and_130_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_or_130_39 = arc.or %cst_130, %cst_39 : ui16
    "arc.keep"(%result_or_130_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_130_39 = arc.xor %cst_130, %cst_39 : ui16
    "arc.keep"(%result_xor_130_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST120]]) : (ui16) -> ()
    %result_and_130_98 = arc.and %cst_130, %cst_98 : ui16
    "arc.keep"(%result_and_130_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_or_130_98 = arc.or %cst_130, %cst_98 : ui16
    "arc.keep"(%result_or_130_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_130_98 = arc.xor %cst_130, %cst_98 : ui16
    "arc.keep"(%result_xor_130_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST34]]) : (ui16) -> ()
    %result_and_130_82 = arc.and %cst_130, %cst_82 : ui16
    "arc.keep"(%result_and_130_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_or_130_82 = arc.or %cst_130, %cst_82 : ui16
    "arc.keep"(%result_or_130_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_130_82 = arc.xor %cst_130, %cst_82 : ui16
    "arc.keep"(%result_xor_130_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST44]]) : (ui16) -> ()
    %result_and_39_32 = arc.and %cst_39, %cst_32 : ui16
    "arc.keep"(%result_and_39_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_39_32 = arc.or %cst_39, %cst_32 : ui16
    "arc.keep"(%result_or_39_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_xor_39_32 = arc.xor %cst_39, %cst_32 : ui16
    "arc.keep"(%result_xor_39_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_and_39_130 = arc.and %cst_39, %cst_130 : ui16
    "arc.keep"(%result_and_39_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_or_39_130 = arc.or %cst_39, %cst_130 : ui16
    "arc.keep"(%result_or_39_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_39_130 = arc.xor %cst_39, %cst_130 : ui16
    "arc.keep"(%result_xor_39_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST120]]) : (ui16) -> ()
    %result_and_39_39 = arc.and %cst_39, %cst_39 : ui16
    "arc.keep"(%result_and_39_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_or_39_39 = arc.or %cst_39, %cst_39 : ui16
    "arc.keep"(%result_or_39_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST39]]) : (ui16) -> ()
    %result_xor_39_39 = arc.xor %cst_39, %cst_39 : ui16
    "arc.keep"(%result_xor_39_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_and_39_98 = arc.and %cst_39, %cst_98 : ui16
    "arc.keep"(%result_and_39_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST67]]) : (ui16) -> ()
    %result_or_39_98 = arc.or %cst_39, %cst_98 : ui16
    "arc.keep"(%result_or_39_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST87]]) : (ui16) -> ()
    %result_xor_39_98 = arc.xor %cst_39, %cst_98 : ui16
    "arc.keep"(%result_xor_39_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST90]]) : (ui16) -> ()
    %result_and_39_82 = arc.and %cst_39, %cst_82 : ui16
    "arc.keep"(%result_and_39_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST71]]) : (ui16) -> ()
    %result_or_39_82 = arc.or %cst_39, %cst_82 : ui16
    "arc.keep"(%result_or_39_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST126]]) : (ui16) -> ()
    %result_xor_39_82 = arc.xor %cst_39, %cst_82 : ui16
    "arc.keep"(%result_xor_39_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST26]]) : (ui16) -> ()
    %result_and_98_32 = arc.and %cst_98, %cst_32 : ui16
    "arc.keep"(%result_and_98_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_98_32 = arc.or %cst_98, %cst_32 : ui16
    "arc.keep"(%result_or_98_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_xor_98_32 = arc.xor %cst_98, %cst_32 : ui16
    "arc.keep"(%result_xor_98_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_and_98_130 = arc.and %cst_98, %cst_130 : ui16
    "arc.keep"(%result_and_98_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_or_98_130 = arc.or %cst_98, %cst_130 : ui16
    "arc.keep"(%result_or_98_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_98_130 = arc.xor %cst_98, %cst_130 : ui16
    "arc.keep"(%result_xor_98_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST34]]) : (ui16) -> ()
    %result_and_98_39 = arc.and %cst_98, %cst_39 : ui16
    "arc.keep"(%result_and_98_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST67]]) : (ui16) -> ()
    %result_or_98_39 = arc.or %cst_98, %cst_39 : ui16
    "arc.keep"(%result_or_98_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST87]]) : (ui16) -> ()
    %result_xor_98_39 = arc.xor %cst_98, %cst_39 : ui16
    "arc.keep"(%result_xor_98_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST90]]) : (ui16) -> ()
    %result_and_98_98 = arc.and %cst_98, %cst_98 : ui16
    "arc.keep"(%result_and_98_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_or_98_98 = arc.or %cst_98, %cst_98 : ui16
    "arc.keep"(%result_or_98_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST98]]) : (ui16) -> ()
    %result_xor_98_98 = arc.xor %cst_98, %cst_98 : ui16
    "arc.keep"(%result_xor_98_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_and_98_82 = arc.and %cst_98, %cst_82 : ui16
    "arc.keep"(%result_and_98_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST104]]) : (ui16) -> ()
    %result_or_98_82 = arc.or %cst_98, %cst_82 : ui16
    "arc.keep"(%result_or_98_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST57]]) : (ui16) -> ()
    %result_xor_98_82 = arc.xor %cst_98, %cst_82 : ui16
    "arc.keep"(%result_xor_98_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST0]]) : (ui16) -> ()
    %result_and_82_32 = arc.and %cst_82, %cst_32 : ui16
    "arc.keep"(%result_and_82_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_or_82_32 = arc.or %cst_82, %cst_32 : ui16
    "arc.keep"(%result_or_82_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_xor_82_32 = arc.xor %cst_82, %cst_32 : ui16
    "arc.keep"(%result_xor_82_32) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_and_82_130 = arc.and %cst_82, %cst_130 : ui16
    "arc.keep"(%result_and_82_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_or_82_130 = arc.or %cst_82, %cst_130 : ui16
    "arc.keep"(%result_or_82_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST130]]) : (ui16) -> ()
    %result_xor_82_130 = arc.xor %cst_82, %cst_130 : ui16
    "arc.keep"(%result_xor_82_130) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST44]]) : (ui16) -> ()
    %result_and_82_39 = arc.and %cst_82, %cst_39 : ui16
    "arc.keep"(%result_and_82_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST71]]) : (ui16) -> ()
    %result_or_82_39 = arc.or %cst_82, %cst_39 : ui16
    "arc.keep"(%result_or_82_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST126]]) : (ui16) -> ()
    %result_xor_82_39 = arc.xor %cst_82, %cst_39 : ui16
    "arc.keep"(%result_xor_82_39) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST26]]) : (ui16) -> ()
    %result_and_82_98 = arc.and %cst_82, %cst_98 : ui16
    "arc.keep"(%result_and_82_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST104]]) : (ui16) -> ()
    %result_or_82_98 = arc.or %cst_82, %cst_98 : ui16
    "arc.keep"(%result_or_82_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST57]]) : (ui16) -> ()
    %result_xor_82_98 = arc.xor %cst_82, %cst_98 : ui16
    "arc.keep"(%result_xor_82_98) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST0]]) : (ui16) -> ()
    %result_and_82_82 = arc.and %cst_82, %cst_82 : ui16
    "arc.keep"(%result_and_82_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_or_82_82 = arc.or %cst_82, %cst_82 : ui16
    "arc.keep"(%result_or_82_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST82]]) : (ui16) -> ()
    %result_xor_82_82 = arc.xor %cst_82, %cst_82 : ui16
    "arc.keep"(%result_xor_82_82) : (ui16) -> ()
    // CHECK: "arc.keep"([[CST32]]) : (ui16) -> ()
    %result_and_68_68 = arc.and %cst_68, %cst_68 : si16
    "arc.keep"(%result_and_68_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_68 = arc.or %cst_68, %cst_68 : si16
    "arc.keep"(%result_or_68_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_xor_68_68 = arc.xor %cst_68, %cst_68 : si16
    "arc.keep"(%result_xor_68_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_68_78 = arc.and %cst_68, %cst_78 : si16
    "arc.keep"(%result_and_68_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_78 = arc.or %cst_68, %cst_78 : si16
    "arc.keep"(%result_or_68_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_xor_68_78 = arc.xor %cst_68, %cst_78 : si16
    "arc.keep"(%result_xor_68_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_and_68_53 = arc.and %cst_68, %cst_53 : si16
    "arc.keep"(%result_and_68_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_53 = arc.or %cst_68, %cst_53 : si16
    "arc.keep"(%result_or_68_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_xor_68_53 = arc.xor %cst_68, %cst_53 : si16
    "arc.keep"(%result_xor_68_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_and_68_127 = arc.and %cst_68, %cst_127 : si16
    "arc.keep"(%result_and_68_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_127 = arc.or %cst_68, %cst_127 : si16
    "arc.keep"(%result_or_68_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_xor_68_127 = arc.xor %cst_68, %cst_127 : si16
    "arc.keep"(%result_xor_68_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_and_68_112 = arc.and %cst_68, %cst_112 : si16
    "arc.keep"(%result_and_68_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_112 = arc.or %cst_68, %cst_112 : si16
    "arc.keep"(%result_or_68_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_xor_68_112 = arc.xor %cst_68, %cst_112 : si16
    "arc.keep"(%result_xor_68_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_and_68_80 = arc.and %cst_68, %cst_80 : si16
    "arc.keep"(%result_and_68_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_68_80 = arc.or %cst_68, %cst_80 : si16
    "arc.keep"(%result_or_68_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_xor_68_80 = arc.xor %cst_68, %cst_80 : si16
    "arc.keep"(%result_xor_68_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_and_78_68 = arc.and %cst_78, %cst_68 : si16
    "arc.keep"(%result_and_78_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_78_68 = arc.or %cst_78, %cst_68 : si16
    "arc.keep"(%result_or_78_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_xor_78_68 = arc.xor %cst_78, %cst_68 : si16
    "arc.keep"(%result_xor_78_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_and_78_78 = arc.and %cst_78, %cst_78 : si16
    "arc.keep"(%result_and_78_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_or_78_78 = arc.or %cst_78, %cst_78 : si16
    "arc.keep"(%result_or_78_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_xor_78_78 = arc.xor %cst_78, %cst_78 : si16
    "arc.keep"(%result_xor_78_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_78_53 = arc.and %cst_78, %cst_53 : si16
    "arc.keep"(%result_and_78_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_or_78_53 = arc.or %cst_78, %cst_53 : si16
    "arc.keep"(%result_or_78_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_xor_78_53 = arc.xor %cst_78, %cst_53 : si16
    "arc.keep"(%result_xor_78_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST143]]) : (si16) -> ()
    %result_and_78_127 = arc.and %cst_78, %cst_127 : si16
    "arc.keep"(%result_and_78_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_78_127 = arc.or %cst_78, %cst_127 : si16
    "arc.keep"(%result_or_78_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_78_127 = arc.xor %cst_78, %cst_127 : si16
    "arc.keep"(%result_xor_78_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_and_78_112 = arc.and %cst_78, %cst_112 : si16
    "arc.keep"(%result_and_78_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_or_78_112 = arc.or %cst_78, %cst_112 : si16
    "arc.keep"(%result_or_78_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_xor_78_112 = arc.xor %cst_78, %cst_112 : si16
    "arc.keep"(%result_xor_78_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST48]]) : (si16) -> ()
    %result_and_78_80 = arc.and %cst_78, %cst_80 : si16
    "arc.keep"(%result_and_78_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_78_80 = arc.or %cst_78, %cst_80 : si16
    "arc.keep"(%result_or_78_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST14]]) : (si16) -> ()
    %result_xor_78_80 = arc.xor %cst_78, %cst_80 : si16
    "arc.keep"(%result_xor_78_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST14]]) : (si16) -> ()
    %result_and_53_68 = arc.and %cst_53, %cst_68 : si16
    "arc.keep"(%result_and_53_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_53_68 = arc.or %cst_53, %cst_68 : si16
    "arc.keep"(%result_or_53_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_xor_53_68 = arc.xor %cst_53, %cst_68 : si16
    "arc.keep"(%result_xor_53_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_and_53_78 = arc.and %cst_53, %cst_78 : si16
    "arc.keep"(%result_and_53_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_or_53_78 = arc.or %cst_53, %cst_78 : si16
    "arc.keep"(%result_or_53_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_xor_53_78 = arc.xor %cst_53, %cst_78 : si16
    "arc.keep"(%result_xor_53_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST143]]) : (si16) -> ()
    %result_and_53_53 = arc.and %cst_53, %cst_53 : si16
    "arc.keep"(%result_and_53_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_or_53_53 = arc.or %cst_53, %cst_53 : si16
    "arc.keep"(%result_or_53_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST53]]) : (si16) -> ()
    %result_xor_53_53 = arc.xor %cst_53, %cst_53 : si16
    "arc.keep"(%result_xor_53_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_53_127 = arc.and %cst_53, %cst_127 : si16
    "arc.keep"(%result_and_53_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST143]]) : (si16) -> ()
    %result_or_53_127 = arc.or %cst_53, %cst_127 : si16
    "arc.keep"(%result_or_53_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_53_127 = arc.xor %cst_53, %cst_127 : si16
    "arc.keep"(%result_xor_53_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST108]]) : (si16) -> ()
    %result_and_53_112 = arc.and %cst_53, %cst_112 : si16
    "arc.keep"(%result_and_53_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST119]]) : (si16) -> ()
    %result_or_53_112 = arc.or %cst_53, %cst_112 : si16
    "arc.keep"(%result_or_53_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST6]]) : (si16) -> ()
    %result_xor_53_112 = arc.xor %cst_53, %cst_112 : si16
    "arc.keep"(%result_xor_53_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST137]]) : (si16) -> ()
    %result_and_53_80 = arc.and %cst_53, %cst_80 : si16
    "arc.keep"(%result_and_53_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST121]]) : (si16) -> ()
    %result_or_53_80 = arc.or %cst_53, %cst_80 : si16
    "arc.keep"(%result_or_53_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST147]]) : (si16) -> ()
    %result_xor_53_80 = arc.xor %cst_53, %cst_80 : si16
    "arc.keep"(%result_xor_53_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST31]]) : (si16) -> ()
    %result_and_127_68 = arc.and %cst_127, %cst_68 : si16
    "arc.keep"(%result_and_127_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_127_68 = arc.or %cst_127, %cst_68 : si16
    "arc.keep"(%result_or_127_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_xor_127_68 = arc.xor %cst_127, %cst_68 : si16
    "arc.keep"(%result_xor_127_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_and_127_78 = arc.and %cst_127, %cst_78 : si16
    "arc.keep"(%result_and_127_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_127_78 = arc.or %cst_127, %cst_78 : si16
    "arc.keep"(%result_or_127_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_127_78 = arc.xor %cst_127, %cst_78 : si16
    "arc.keep"(%result_xor_127_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_and_127_53 = arc.and %cst_127, %cst_53 : si16
    "arc.keep"(%result_and_127_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST143]]) : (si16) -> ()
    %result_or_127_53 = arc.or %cst_127, %cst_53 : si16
    "arc.keep"(%result_or_127_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_127_53 = arc.xor %cst_127, %cst_53 : si16
    "arc.keep"(%result_xor_127_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST108]]) : (si16) -> ()
    %result_and_127_127 = arc.and %cst_127, %cst_127 : si16
    "arc.keep"(%result_and_127_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_or_127_127 = arc.or %cst_127, %cst_127 : si16
    "arc.keep"(%result_or_127_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_xor_127_127 = arc.xor %cst_127, %cst_127 : si16
    "arc.keep"(%result_xor_127_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_127_112 = arc.and %cst_127, %cst_112 : si16
    "arc.keep"(%result_and_127_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST48]]) : (si16) -> ()
    %result_or_127_112 = arc.or %cst_127, %cst_112 : si16
    "arc.keep"(%result_or_127_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_127_112 = arc.xor %cst_127, %cst_112 : si16
    "arc.keep"(%result_xor_127_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST61]]) : (si16) -> ()
    %result_and_127_80 = arc.and %cst_127, %cst_80 : si16
    "arc.keep"(%result_and_127_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_or_127_80 = arc.or %cst_127, %cst_80 : si16
    "arc.keep"(%result_or_127_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_xor_127_80 = arc.xor %cst_127, %cst_80 : si16
    "arc.keep"(%result_xor_127_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST69]]) : (si16) -> ()
    %result_and_112_68 = arc.and %cst_112, %cst_68 : si16
    "arc.keep"(%result_and_112_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_112_68 = arc.or %cst_112, %cst_68 : si16
    "arc.keep"(%result_or_112_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_xor_112_68 = arc.xor %cst_112, %cst_68 : si16
    "arc.keep"(%result_xor_112_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_and_112_78 = arc.and %cst_112, %cst_78 : si16
    "arc.keep"(%result_and_112_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST78]]) : (si16) -> ()
    %result_or_112_78 = arc.or %cst_112, %cst_78 : si16
    "arc.keep"(%result_or_112_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_xor_112_78 = arc.xor %cst_112, %cst_78 : si16
    "arc.keep"(%result_xor_112_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST48]]) : (si16) -> ()
    %result_and_112_53 = arc.and %cst_112, %cst_53 : si16
    "arc.keep"(%result_and_112_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST119]]) : (si16) -> ()
    %result_or_112_53 = arc.or %cst_112, %cst_53 : si16
    "arc.keep"(%result_or_112_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST6]]) : (si16) -> ()
    %result_xor_112_53 = arc.xor %cst_112, %cst_53 : si16
    "arc.keep"(%result_xor_112_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST137]]) : (si16) -> ()
    %result_and_112_127 = arc.and %cst_112, %cst_127 : si16
    "arc.keep"(%result_and_112_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST48]]) : (si16) -> ()
    %result_or_112_127 = arc.or %cst_112, %cst_127 : si16
    "arc.keep"(%result_or_112_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST3]]) : (si16) -> ()
    %result_xor_112_127 = arc.xor %cst_112, %cst_127 : si16
    "arc.keep"(%result_xor_112_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST61]]) : (si16) -> ()
    %result_and_112_112 = arc.and %cst_112, %cst_112 : si16
    "arc.keep"(%result_and_112_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_or_112_112 = arc.or %cst_112, %cst_112 : si16
    "arc.keep"(%result_or_112_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST112]]) : (si16) -> ()
    %result_xor_112_112 = arc.xor %cst_112, %cst_112 : si16
    "arc.keep"(%result_xor_112_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_112_80 = arc.and %cst_112, %cst_80 : si16
    "arc.keep"(%result_and_112_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST99]]) : (si16) -> ()
    %result_or_112_80 = arc.or %cst_112, %cst_80 : si16
    "arc.keep"(%result_or_112_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST132]]) : (si16) -> ()
    %result_xor_112_80 = arc.xor %cst_112, %cst_80 : si16
    "arc.keep"(%result_xor_112_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST59]]) : (si16) -> ()
    %result_and_80_68 = arc.and %cst_80, %cst_68 : si16
    "arc.keep"(%result_and_80_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_80_68 = arc.or %cst_80, %cst_68 : si16
    "arc.keep"(%result_or_80_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_xor_80_68 = arc.xor %cst_80, %cst_68 : si16
    "arc.keep"(%result_xor_80_68) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_and_80_78 = arc.and %cst_80, %cst_78 : si16
    "arc.keep"(%result_and_80_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_or_80_78 = arc.or %cst_80, %cst_78 : si16
    "arc.keep"(%result_or_80_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST14]]) : (si16) -> ()
    %result_xor_80_78 = arc.xor %cst_80, %cst_78 : si16
    "arc.keep"(%result_xor_80_78) : (si16) -> ()
    // CHECK: "arc.keep"([[CST14]]) : (si16) -> ()
    %result_and_80_53 = arc.and %cst_80, %cst_53 : si16
    "arc.keep"(%result_and_80_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST121]]) : (si16) -> ()
    %result_or_80_53 = arc.or %cst_80, %cst_53 : si16
    "arc.keep"(%result_or_80_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST147]]) : (si16) -> ()
    %result_xor_80_53 = arc.xor %cst_80, %cst_53 : si16
    "arc.keep"(%result_xor_80_53) : (si16) -> ()
    // CHECK: "arc.keep"([[CST31]]) : (si16) -> ()
    %result_and_80_127 = arc.and %cst_80, %cst_127 : si16
    "arc.keep"(%result_and_80_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_or_80_127 = arc.or %cst_80, %cst_127 : si16
    "arc.keep"(%result_or_80_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST127]]) : (si16) -> ()
    %result_xor_80_127 = arc.xor %cst_80, %cst_127 : si16
    "arc.keep"(%result_xor_80_127) : (si16) -> ()
    // CHECK: "arc.keep"([[CST69]]) : (si16) -> ()
    %result_and_80_112 = arc.and %cst_80, %cst_112 : si16
    "arc.keep"(%result_and_80_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST99]]) : (si16) -> ()
    %result_or_80_112 = arc.or %cst_80, %cst_112 : si16
    "arc.keep"(%result_or_80_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST132]]) : (si16) -> ()
    %result_xor_80_112 = arc.xor %cst_80, %cst_112 : si16
    "arc.keep"(%result_xor_80_112) : (si16) -> ()
    // CHECK: "arc.keep"([[CST59]]) : (si16) -> ()
    %result_and_80_80 = arc.and %cst_80, %cst_80 : si16
    "arc.keep"(%result_and_80_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_or_80_80 = arc.or %cst_80, %cst_80 : si16
    "arc.keep"(%result_or_80_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST80]]) : (si16) -> ()
    %result_xor_80_80 = arc.xor %cst_80, %cst_80 : si16
    "arc.keep"(%result_xor_80_80) : (si16) -> ()
    // CHECK: "arc.keep"([[CST68]]) : (si16) -> ()
    %result_and_144_144 = arc.and %cst_144, %cst_144 : ui32
    "arc.keep"(%result_and_144_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_or_144_144 = arc.or %cst_144, %cst_144 : ui32
    "arc.keep"(%result_or_144_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_xor_144_144 = arc.xor %cst_144, %cst_144 : ui32
    "arc.keep"(%result_xor_144_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_and_144_17 = arc.and %cst_144, %cst_17 : ui32
    "arc.keep"(%result_and_144_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST97]]) : (ui32) -> ()
    %result_or_144_17 = arc.or %cst_144, %cst_17 : ui32
    "arc.keep"(%result_or_144_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST35]]) : (ui32) -> ()
    %result_xor_144_17 = arc.xor %cst_144, %cst_17 : ui32
    "arc.keep"(%result_xor_144_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST63]]) : (ui32) -> ()
    %result_and_144_151 = arc.and %cst_144, %cst_151 : ui32
    "arc.keep"(%result_and_144_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST72]]) : (ui32) -> ()
    %result_or_144_151 = arc.or %cst_144, %cst_151 : ui32
    "arc.keep"(%result_or_144_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST73]]) : (ui32) -> ()
    %result_xor_144_151 = arc.xor %cst_144, %cst_151 : ui32
    "arc.keep"(%result_xor_144_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST41]]) : (ui32) -> ()
    %result_and_144_86 = arc.and %cst_144, %cst_86 : ui32
    "arc.keep"(%result_and_144_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_144_86 = arc.or %cst_144, %cst_86 : ui32
    "arc.keep"(%result_or_144_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_xor_144_86 = arc.xor %cst_144, %cst_86 : ui32
    "arc.keep"(%result_xor_144_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_and_144_1 = arc.and %cst_144, %cst_1 : ui32
    "arc.keep"(%result_and_144_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_or_144_1 = arc.or %cst_144, %cst_1 : ui32
    "arc.keep"(%result_or_144_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_144_1 = arc.xor %cst_144, %cst_1 : ui32
    "arc.keep"(%result_xor_144_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST91]]) : (ui32) -> ()
    %result_and_17_144 = arc.and %cst_17, %cst_144 : ui32
    "arc.keep"(%result_and_17_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST97]]) : (ui32) -> ()
    %result_or_17_144 = arc.or %cst_17, %cst_144 : ui32
    "arc.keep"(%result_or_17_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST35]]) : (ui32) -> ()
    %result_xor_17_144 = arc.xor %cst_17, %cst_144 : ui32
    "arc.keep"(%result_xor_17_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST63]]) : (ui32) -> ()
    %result_and_17_17 = arc.and %cst_17, %cst_17 : ui32
    "arc.keep"(%result_and_17_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_or_17_17 = arc.or %cst_17, %cst_17 : ui32
    "arc.keep"(%result_or_17_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_xor_17_17 = arc.xor %cst_17, %cst_17 : ui32
    "arc.keep"(%result_xor_17_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_and_17_151 = arc.and %cst_17, %cst_151 : ui32
    "arc.keep"(%result_and_17_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST88]]) : (ui32) -> ()
    %result_or_17_151 = arc.or %cst_17, %cst_151 : ui32
    "arc.keep"(%result_or_17_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST52]]) : (ui32) -> ()
    %result_xor_17_151 = arc.xor %cst_17, %cst_151 : ui32
    "arc.keep"(%result_xor_17_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST23]]) : (ui32) -> ()
    %result_and_17_86 = arc.and %cst_17, %cst_86 : ui32
    "arc.keep"(%result_and_17_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_17_86 = arc.or %cst_17, %cst_86 : ui32
    "arc.keep"(%result_or_17_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_xor_17_86 = arc.xor %cst_17, %cst_86 : ui32
    "arc.keep"(%result_xor_17_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_and_17_1 = arc.and %cst_17, %cst_1 : ui32
    "arc.keep"(%result_and_17_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_or_17_1 = arc.or %cst_17, %cst_1 : ui32
    "arc.keep"(%result_or_17_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_17_1 = arc.xor %cst_17, %cst_1 : ui32
    "arc.keep"(%result_xor_17_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST11]]) : (ui32) -> ()
    %result_and_151_144 = arc.and %cst_151, %cst_144 : ui32
    "arc.keep"(%result_and_151_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST72]]) : (ui32) -> ()
    %result_or_151_144 = arc.or %cst_151, %cst_144 : ui32
    "arc.keep"(%result_or_151_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST73]]) : (ui32) -> ()
    %result_xor_151_144 = arc.xor %cst_151, %cst_144 : ui32
    "arc.keep"(%result_xor_151_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST41]]) : (ui32) -> ()
    %result_and_151_17 = arc.and %cst_151, %cst_17 : ui32
    "arc.keep"(%result_and_151_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST88]]) : (ui32) -> ()
    %result_or_151_17 = arc.or %cst_151, %cst_17 : ui32
    "arc.keep"(%result_or_151_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST52]]) : (ui32) -> ()
    %result_xor_151_17 = arc.xor %cst_151, %cst_17 : ui32
    "arc.keep"(%result_xor_151_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST23]]) : (ui32) -> ()
    %result_and_151_151 = arc.and %cst_151, %cst_151 : ui32
    "arc.keep"(%result_and_151_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_or_151_151 = arc.or %cst_151, %cst_151 : ui32
    "arc.keep"(%result_or_151_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_xor_151_151 = arc.xor %cst_151, %cst_151 : ui32
    "arc.keep"(%result_xor_151_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_and_151_86 = arc.and %cst_151, %cst_86 : ui32
    "arc.keep"(%result_and_151_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_151_86 = arc.or %cst_151, %cst_86 : ui32
    "arc.keep"(%result_or_151_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_xor_151_86 = arc.xor %cst_151, %cst_86 : ui32
    "arc.keep"(%result_xor_151_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_and_151_1 = arc.and %cst_151, %cst_1 : ui32
    "arc.keep"(%result_and_151_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_or_151_1 = arc.or %cst_151, %cst_1 : ui32
    "arc.keep"(%result_or_151_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_151_1 = arc.xor %cst_151, %cst_1 : ui32
    "arc.keep"(%result_xor_151_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST93]]) : (ui32) -> ()
    %result_and_86_144 = arc.and %cst_86, %cst_144 : ui32
    "arc.keep"(%result_and_86_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_86_144 = arc.or %cst_86, %cst_144 : ui32
    "arc.keep"(%result_or_86_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_xor_86_144 = arc.xor %cst_86, %cst_144 : ui32
    "arc.keep"(%result_xor_86_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_and_86_17 = arc.and %cst_86, %cst_17 : ui32
    "arc.keep"(%result_and_86_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_86_17 = arc.or %cst_86, %cst_17 : ui32
    "arc.keep"(%result_or_86_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_xor_86_17 = arc.xor %cst_86, %cst_17 : ui32
    "arc.keep"(%result_xor_86_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_and_86_151 = arc.and %cst_86, %cst_151 : ui32
    "arc.keep"(%result_and_86_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_86_151 = arc.or %cst_86, %cst_151 : ui32
    "arc.keep"(%result_or_86_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_xor_86_151 = arc.xor %cst_86, %cst_151 : ui32
    "arc.keep"(%result_xor_86_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_and_86_86 = arc.and %cst_86, %cst_86 : ui32
    "arc.keep"(%result_and_86_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_86_86 = arc.or %cst_86, %cst_86 : ui32
    "arc.keep"(%result_or_86_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_xor_86_86 = arc.xor %cst_86, %cst_86 : ui32
    "arc.keep"(%result_xor_86_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_and_86_1 = arc.and %cst_86, %cst_1 : ui32
    "arc.keep"(%result_and_86_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_86_1 = arc.or %cst_86, %cst_1 : ui32
    "arc.keep"(%result_or_86_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_86_1 = arc.xor %cst_86, %cst_1 : ui32
    "arc.keep"(%result_xor_86_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_and_1_144 = arc.and %cst_1, %cst_144 : ui32
    "arc.keep"(%result_and_1_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST144]]) : (ui32) -> ()
    %result_or_1_144 = arc.or %cst_1, %cst_144 : ui32
    "arc.keep"(%result_or_1_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_1_144 = arc.xor %cst_1, %cst_144 : ui32
    "arc.keep"(%result_xor_1_144) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST91]]) : (ui32) -> ()
    %result_and_1_17 = arc.and %cst_1, %cst_17 : ui32
    "arc.keep"(%result_and_1_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST17]]) : (ui32) -> ()
    %result_or_1_17 = arc.or %cst_1, %cst_17 : ui32
    "arc.keep"(%result_or_1_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_1_17 = arc.xor %cst_1, %cst_17 : ui32
    "arc.keep"(%result_xor_1_17) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST11]]) : (ui32) -> ()
    %result_and_1_151 = arc.and %cst_1, %cst_151 : ui32
    "arc.keep"(%result_and_1_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST151]]) : (ui32) -> ()
    %result_or_1_151 = arc.or %cst_1, %cst_151 : ui32
    "arc.keep"(%result_or_1_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_1_151 = arc.xor %cst_1, %cst_151 : ui32
    "arc.keep"(%result_xor_1_151) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST93]]) : (ui32) -> ()
    %result_and_1_86 = arc.and %cst_1, %cst_86 : ui32
    "arc.keep"(%result_and_1_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_or_1_86 = arc.or %cst_1, %cst_86 : ui32
    "arc.keep"(%result_or_1_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_1_86 = arc.xor %cst_1, %cst_86 : ui32
    "arc.keep"(%result_xor_1_86) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_and_1_1 = arc.and %cst_1, %cst_1 : ui32
    "arc.keep"(%result_and_1_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_or_1_1 = arc.or %cst_1, %cst_1 : ui32
    "arc.keep"(%result_or_1_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST1]]) : (ui32) -> ()
    %result_xor_1_1 = arc.xor %cst_1, %cst_1 : ui32
    "arc.keep"(%result_xor_1_1) : (ui32) -> ()
    // CHECK: "arc.keep"([[CST86]]) : (ui32) -> ()
    %result_and_111_111 = arc.and %cst_111, %cst_111 : si32
    "arc.keep"(%result_and_111_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_111 = arc.or %cst_111, %cst_111 : si32
    "arc.keep"(%result_or_111_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_xor_111_111 = arc.xor %cst_111, %cst_111 : si32
    "arc.keep"(%result_xor_111_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_111_7 = arc.and %cst_111, %cst_7 : si32
    "arc.keep"(%result_and_111_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_7 = arc.or %cst_111, %cst_7 : si32
    "arc.keep"(%result_or_111_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_xor_111_7 = arc.xor %cst_111, %cst_7 : si32
    "arc.keep"(%result_xor_111_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_and_111_115 = arc.and %cst_111, %cst_115 : si32
    "arc.keep"(%result_and_111_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_115 = arc.or %cst_111, %cst_115 : si32
    "arc.keep"(%result_or_111_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_xor_111_115 = arc.xor %cst_111, %cst_115 : si32
    "arc.keep"(%result_xor_111_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_and_111_116 = arc.and %cst_111, %cst_116 : si32
    "arc.keep"(%result_and_111_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_116 = arc.or %cst_111, %cst_116 : si32
    "arc.keep"(%result_or_111_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_xor_111_116 = arc.xor %cst_111, %cst_116 : si32
    "arc.keep"(%result_xor_111_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_and_111_105 = arc.and %cst_111, %cst_105 : si32
    "arc.keep"(%result_and_111_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_105 = arc.or %cst_111, %cst_105 : si32
    "arc.keep"(%result_or_111_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_xor_111_105 = arc.xor %cst_111, %cst_105 : si32
    "arc.keep"(%result_xor_111_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_and_111_134 = arc.and %cst_111, %cst_134 : si32
    "arc.keep"(%result_and_111_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_111_134 = arc.or %cst_111, %cst_134 : si32
    "arc.keep"(%result_or_111_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_xor_111_134 = arc.xor %cst_111, %cst_134 : si32
    "arc.keep"(%result_xor_111_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_and_7_111 = arc.and %cst_7, %cst_111 : si32
    "arc.keep"(%result_and_7_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_7_111 = arc.or %cst_7, %cst_111 : si32
    "arc.keep"(%result_or_7_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_xor_7_111 = arc.xor %cst_7, %cst_111 : si32
    "arc.keep"(%result_xor_7_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_and_7_7 = arc.and %cst_7, %cst_7 : si32
    "arc.keep"(%result_and_7_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_or_7_7 = arc.or %cst_7, %cst_7 : si32
    "arc.keep"(%result_or_7_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_xor_7_7 = arc.xor %cst_7, %cst_7 : si32
    "arc.keep"(%result_xor_7_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_7_115 = arc.and %cst_7, %cst_115 : si32
    "arc.keep"(%result_and_7_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_7_115 = arc.or %cst_7, %cst_115 : si32
    "arc.keep"(%result_or_7_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST145]]) : (si32) -> ()
    %result_xor_7_115 = arc.xor %cst_7, %cst_115 : si32
    "arc.keep"(%result_xor_7_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST145]]) : (si32) -> ()
    %result_and_7_116 = arc.and %cst_7, %cst_116 : si32
    "arc.keep"(%result_and_7_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_7_116 = arc.or %cst_7, %cst_116 : si32
    "arc.keep"(%result_or_7_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_7_116 = arc.xor %cst_7, %cst_116 : si32
    "arc.keep"(%result_xor_7_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_and_7_105 = arc.and %cst_7, %cst_105 : si32
    "arc.keep"(%result_and_7_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_or_7_105 = arc.or %cst_7, %cst_105 : si32
    "arc.keep"(%result_or_7_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_xor_7_105 = arc.xor %cst_7, %cst_105 : si32
    "arc.keep"(%result_xor_7_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST75]]) : (si32) -> ()
    %result_and_7_134 = arc.and %cst_7, %cst_134 : si32
    "arc.keep"(%result_and_7_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_or_7_134 = arc.or %cst_7, %cst_134 : si32
    "arc.keep"(%result_or_7_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_xor_7_134 = arc.xor %cst_7, %cst_134 : si32
    "arc.keep"(%result_xor_7_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST140]]) : (si32) -> ()
    %result_and_115_111 = arc.and %cst_115, %cst_111 : si32
    "arc.keep"(%result_and_115_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_115_111 = arc.or %cst_115, %cst_111 : si32
    "arc.keep"(%result_or_115_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_xor_115_111 = arc.xor %cst_115, %cst_111 : si32
    "arc.keep"(%result_xor_115_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_and_115_7 = arc.and %cst_115, %cst_7 : si32
    "arc.keep"(%result_and_115_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_115_7 = arc.or %cst_115, %cst_7 : si32
    "arc.keep"(%result_or_115_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST145]]) : (si32) -> ()
    %result_xor_115_7 = arc.xor %cst_115, %cst_7 : si32
    "arc.keep"(%result_xor_115_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST145]]) : (si32) -> ()
    %result_and_115_115 = arc.and %cst_115, %cst_115 : si32
    "arc.keep"(%result_and_115_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_or_115_115 = arc.or %cst_115, %cst_115 : si32
    "arc.keep"(%result_or_115_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_xor_115_115 = arc.xor %cst_115, %cst_115 : si32
    "arc.keep"(%result_xor_115_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_115_116 = arc.and %cst_115, %cst_116 : si32
    "arc.keep"(%result_and_115_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_or_115_116 = arc.or %cst_115, %cst_116 : si32
    "arc.keep"(%result_or_115_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_xor_115_116 = arc.xor %cst_115, %cst_116 : si32
    "arc.keep"(%result_xor_115_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST50]]) : (si32) -> ()
    %result_and_115_105 = arc.and %cst_115, %cst_105 : si32
    "arc.keep"(%result_and_115_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST37]]) : (si32) -> ()
    %result_or_115_105 = arc.or %cst_115, %cst_105 : si32
    "arc.keep"(%result_or_115_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST123]]) : (si32) -> ()
    %result_xor_115_105 = arc.xor %cst_115, %cst_105 : si32
    "arc.keep"(%result_xor_115_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST43]]) : (si32) -> ()
    %result_and_115_134 = arc.and %cst_115, %cst_134 : si32
    "arc.keep"(%result_and_115_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST25]]) : (si32) -> ()
    %result_or_115_134 = arc.or %cst_115, %cst_134 : si32
    "arc.keep"(%result_or_115_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST103]]) : (si32) -> ()
    %result_xor_115_134 = arc.xor %cst_115, %cst_134 : si32
    "arc.keep"(%result_xor_115_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST109]]) : (si32) -> ()
    %result_and_116_111 = arc.and %cst_116, %cst_111 : si32
    "arc.keep"(%result_and_116_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_116_111 = arc.or %cst_116, %cst_111 : si32
    "arc.keep"(%result_or_116_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_xor_116_111 = arc.xor %cst_116, %cst_111 : si32
    "arc.keep"(%result_xor_116_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_and_116_7 = arc.and %cst_116, %cst_7 : si32
    "arc.keep"(%result_and_116_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_116_7 = arc.or %cst_116, %cst_7 : si32
    "arc.keep"(%result_or_116_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_116_7 = arc.xor %cst_116, %cst_7 : si32
    "arc.keep"(%result_xor_116_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_and_116_115 = arc.and %cst_116, %cst_115 : si32
    "arc.keep"(%result_and_116_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST115]]) : (si32) -> ()
    %result_or_116_115 = arc.or %cst_116, %cst_115 : si32
    "arc.keep"(%result_or_116_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_xor_116_115 = arc.xor %cst_116, %cst_115 : si32
    "arc.keep"(%result_xor_116_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST50]]) : (si32) -> ()
    %result_and_116_116 = arc.and %cst_116, %cst_116 : si32
    "arc.keep"(%result_and_116_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_or_116_116 = arc.or %cst_116, %cst_116 : si32
    "arc.keep"(%result_or_116_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST116]]) : (si32) -> ()
    %result_xor_116_116 = arc.xor %cst_116, %cst_116 : si32
    "arc.keep"(%result_xor_116_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_116_105 = arc.and %cst_116, %cst_105 : si32
    "arc.keep"(%result_and_116_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST75]]) : (si32) -> ()
    %result_or_116_105 = arc.or %cst_116, %cst_105 : si32
    "arc.keep"(%result_or_116_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_116_105 = arc.xor %cst_116, %cst_105 : si32
    "arc.keep"(%result_xor_116_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST19]]) : (si32) -> ()
    %result_and_116_134 = arc.and %cst_116, %cst_134 : si32
    "arc.keep"(%result_and_116_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST140]]) : (si32) -> ()
    %result_or_116_134 = arc.or %cst_116, %cst_134 : si32
    "arc.keep"(%result_or_116_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_116_134 = arc.xor %cst_116, %cst_134 : si32
    "arc.keep"(%result_xor_116_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST56]]) : (si32) -> ()
    %result_and_105_111 = arc.and %cst_105, %cst_111 : si32
    "arc.keep"(%result_and_105_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_105_111 = arc.or %cst_105, %cst_111 : si32
    "arc.keep"(%result_or_105_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_xor_105_111 = arc.xor %cst_105, %cst_111 : si32
    "arc.keep"(%result_xor_105_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_and_105_7 = arc.and %cst_105, %cst_7 : si32
    "arc.keep"(%result_and_105_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_or_105_7 = arc.or %cst_105, %cst_7 : si32
    "arc.keep"(%result_or_105_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_xor_105_7 = arc.xor %cst_105, %cst_7 : si32
    "arc.keep"(%result_xor_105_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST75]]) : (si32) -> ()
    %result_and_105_115 = arc.and %cst_105, %cst_115 : si32
    "arc.keep"(%result_and_105_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST37]]) : (si32) -> ()
    %result_or_105_115 = arc.or %cst_105, %cst_115 : si32
    "arc.keep"(%result_or_105_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST123]]) : (si32) -> ()
    %result_xor_105_115 = arc.xor %cst_105, %cst_115 : si32
    "arc.keep"(%result_xor_105_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST43]]) : (si32) -> ()
    %result_and_105_116 = arc.and %cst_105, %cst_116 : si32
    "arc.keep"(%result_and_105_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST75]]) : (si32) -> ()
    %result_or_105_116 = arc.or %cst_105, %cst_116 : si32
    "arc.keep"(%result_or_105_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_105_116 = arc.xor %cst_105, %cst_116 : si32
    "arc.keep"(%result_xor_105_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST19]]) : (si32) -> ()
    %result_and_105_105 = arc.and %cst_105, %cst_105 : si32
    "arc.keep"(%result_and_105_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_or_105_105 = arc.or %cst_105, %cst_105 : si32
    "arc.keep"(%result_or_105_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST105]]) : (si32) -> ()
    %result_xor_105_105 = arc.xor %cst_105, %cst_105 : si32
    "arc.keep"(%result_xor_105_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_105_134 = arc.and %cst_105, %cst_134 : si32
    "arc.keep"(%result_and_105_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST139]]) : (si32) -> ()
    %result_or_105_134 = arc.or %cst_105, %cst_134 : si32
    "arc.keep"(%result_or_105_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST76]]) : (si32) -> ()
    %result_xor_105_134 = arc.xor %cst_105, %cst_134 : si32
    "arc.keep"(%result_xor_105_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST47]]) : (si32) -> ()
    %result_and_134_111 = arc.and %cst_134, %cst_111 : si32
    "arc.keep"(%result_and_134_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_or_134_111 = arc.or %cst_134, %cst_111 : si32
    "arc.keep"(%result_or_134_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_xor_134_111 = arc.xor %cst_134, %cst_111 : si32
    "arc.keep"(%result_xor_134_111) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_and_134_7 = arc.and %cst_134, %cst_7 : si32
    "arc.keep"(%result_and_134_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST7]]) : (si32) -> ()
    %result_or_134_7 = arc.or %cst_134, %cst_7 : si32
    "arc.keep"(%result_or_134_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_xor_134_7 = arc.xor %cst_134, %cst_7 : si32
    "arc.keep"(%result_xor_134_7) : (si32) -> ()
    // CHECK: "arc.keep"([[CST140]]) : (si32) -> ()
    %result_and_134_115 = arc.and %cst_134, %cst_115 : si32
    "arc.keep"(%result_and_134_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST25]]) : (si32) -> ()
    %result_or_134_115 = arc.or %cst_134, %cst_115 : si32
    "arc.keep"(%result_or_134_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST103]]) : (si32) -> ()
    %result_xor_134_115 = arc.xor %cst_134, %cst_115 : si32
    "arc.keep"(%result_xor_134_115) : (si32) -> ()
    // CHECK: "arc.keep"([[CST109]]) : (si32) -> ()
    %result_and_134_116 = arc.and %cst_134, %cst_116 : si32
    "arc.keep"(%result_and_134_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST140]]) : (si32) -> ()
    %result_or_134_116 = arc.or %cst_134, %cst_116 : si32
    "arc.keep"(%result_or_134_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST51]]) : (si32) -> ()
    %result_xor_134_116 = arc.xor %cst_134, %cst_116 : si32
    "arc.keep"(%result_xor_134_116) : (si32) -> ()
    // CHECK: "arc.keep"([[CST56]]) : (si32) -> ()
    %result_and_134_105 = arc.and %cst_134, %cst_105 : si32
    "arc.keep"(%result_and_134_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST139]]) : (si32) -> ()
    %result_or_134_105 = arc.or %cst_134, %cst_105 : si32
    "arc.keep"(%result_or_134_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST76]]) : (si32) -> ()
    %result_xor_134_105 = arc.xor %cst_134, %cst_105 : si32
    "arc.keep"(%result_xor_134_105) : (si32) -> ()
    // CHECK: "arc.keep"([[CST47]]) : (si32) -> ()
    %result_and_134_134 = arc.and %cst_134, %cst_134 : si32
    "arc.keep"(%result_and_134_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_or_134_134 = arc.or %cst_134, %cst_134 : si32
    "arc.keep"(%result_or_134_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST134]]) : (si32) -> ()
    %result_xor_134_134 = arc.xor %cst_134, %cst_134 : si32
    "arc.keep"(%result_xor_134_134) : (si32) -> ()
    // CHECK: "arc.keep"([[CST111]]) : (si32) -> ()
    %result_and_92_92 = arc.and %cst_92, %cst_92 : ui64
    "arc.keep"(%result_and_92_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_or_92_92 = arc.or %cst_92, %cst_92 : ui64
    "arc.keep"(%result_or_92_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_xor_92_92 = arc.xor %cst_92, %cst_92 : ui64
    "arc.keep"(%result_xor_92_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_and_92_102 = arc.and %cst_92, %cst_102 : ui64
    "arc.keep"(%result_and_92_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_or_92_102 = arc.or %cst_92, %cst_102 : ui64
    "arc.keep"(%result_or_92_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_92_102 = arc.xor %cst_92, %cst_102 : ui64
    "arc.keep"(%result_xor_92_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST15]]) : (ui64) -> ()
    %result_and_92_40 = arc.and %cst_92, %cst_40 : ui64
    "arc.keep"(%result_and_92_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_92_40 = arc.or %cst_92, %cst_40 : ui64
    "arc.keep"(%result_or_92_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_xor_92_40 = arc.xor %cst_92, %cst_40 : ui64
    "arc.keep"(%result_xor_92_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_and_92_36 = arc.and %cst_92, %cst_36 : ui64
    "arc.keep"(%result_and_92_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST77]]) : (ui64) -> ()
    %result_or_92_36 = arc.or %cst_92, %cst_36 : ui64
    "arc.keep"(%result_or_92_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST45]]) : (ui64) -> ()
    %result_xor_92_36 = arc.xor %cst_92, %cst_36 : ui64
    "arc.keep"(%result_xor_92_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST107]]) : (ui64) -> ()
    %result_and_92_21 = arc.and %cst_92, %cst_21 : ui64
    "arc.keep"(%result_and_92_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST13]]) : (ui64) -> ()
    %result_or_92_21 = arc.or %cst_92, %cst_21 : ui64
    "arc.keep"(%result_or_92_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST29]]) : (ui64) -> ()
    %result_xor_92_21 = arc.xor %cst_92, %cst_21 : ui64
    "arc.keep"(%result_xor_92_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST122]]) : (ui64) -> ()
    %result_and_102_92 = arc.and %cst_102, %cst_92 : ui64
    "arc.keep"(%result_and_102_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_or_102_92 = arc.or %cst_102, %cst_92 : ui64
    "arc.keep"(%result_or_102_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_102_92 = arc.xor %cst_102, %cst_92 : ui64
    "arc.keep"(%result_xor_102_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST15]]) : (ui64) -> ()
    %result_and_102_102 = arc.and %cst_102, %cst_102 : ui64
    "arc.keep"(%result_and_102_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_or_102_102 = arc.or %cst_102, %cst_102 : ui64
    "arc.keep"(%result_or_102_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_102_102 = arc.xor %cst_102, %cst_102 : ui64
    "arc.keep"(%result_xor_102_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_and_102_40 = arc.and %cst_102, %cst_40 : ui64
    "arc.keep"(%result_and_102_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_102_40 = arc.or %cst_102, %cst_40 : ui64
    "arc.keep"(%result_or_102_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_102_40 = arc.xor %cst_102, %cst_40 : ui64
    "arc.keep"(%result_xor_102_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_and_102_36 = arc.and %cst_102, %cst_36 : ui64
    "arc.keep"(%result_and_102_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_or_102_36 = arc.or %cst_102, %cst_36 : ui64
    "arc.keep"(%result_or_102_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_102_36 = arc.xor %cst_102, %cst_36 : ui64
    "arc.keep"(%result_xor_102_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST124]]) : (ui64) -> ()
    %result_and_102_21 = arc.and %cst_102, %cst_21 : ui64
    "arc.keep"(%result_and_102_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_or_102_21 = arc.or %cst_102, %cst_21 : ui64
    "arc.keep"(%result_or_102_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_102_21 = arc.xor %cst_102, %cst_21 : ui64
    "arc.keep"(%result_xor_102_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST149]]) : (ui64) -> ()
    %result_and_40_92 = arc.and %cst_40, %cst_92 : ui64
    "arc.keep"(%result_and_40_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_40_92 = arc.or %cst_40, %cst_92 : ui64
    "arc.keep"(%result_or_40_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_xor_40_92 = arc.xor %cst_40, %cst_92 : ui64
    "arc.keep"(%result_xor_40_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST92]]) : (ui64) -> ()
    %result_and_40_102 = arc.and %cst_40, %cst_102 : ui64
    "arc.keep"(%result_and_40_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_40_102 = arc.or %cst_40, %cst_102 : ui64
    "arc.keep"(%result_or_40_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_40_102 = arc.xor %cst_40, %cst_102 : ui64
    "arc.keep"(%result_xor_40_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_and_40_40 = arc.and %cst_40, %cst_40 : ui64
    "arc.keep"(%result_and_40_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_40_40 = arc.or %cst_40, %cst_40 : ui64
    "arc.keep"(%result_or_40_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_xor_40_40 = arc.xor %cst_40, %cst_40 : ui64
    "arc.keep"(%result_xor_40_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_and_40_36 = arc.and %cst_40, %cst_36 : ui64
    "arc.keep"(%result_and_40_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_40_36 = arc.or %cst_40, %cst_36 : ui64
    "arc.keep"(%result_or_40_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_xor_40_36 = arc.xor %cst_40, %cst_36 : ui64
    "arc.keep"(%result_xor_40_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_and_40_21 = arc.and %cst_40, %cst_21 : ui64
    "arc.keep"(%result_and_40_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_40_21 = arc.or %cst_40, %cst_21 : ui64
    "arc.keep"(%result_or_40_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_xor_40_21 = arc.xor %cst_40, %cst_21 : ui64
    "arc.keep"(%result_xor_40_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_and_36_92 = arc.and %cst_36, %cst_92 : ui64
    "arc.keep"(%result_and_36_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST77]]) : (ui64) -> ()
    %result_or_36_92 = arc.or %cst_36, %cst_92 : ui64
    "arc.keep"(%result_or_36_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST45]]) : (ui64) -> ()
    %result_xor_36_92 = arc.xor %cst_36, %cst_92 : ui64
    "arc.keep"(%result_xor_36_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST107]]) : (ui64) -> ()
    %result_and_36_102 = arc.and %cst_36, %cst_102 : ui64
    "arc.keep"(%result_and_36_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_or_36_102 = arc.or %cst_36, %cst_102 : ui64
    "arc.keep"(%result_or_36_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_36_102 = arc.xor %cst_36, %cst_102 : ui64
    "arc.keep"(%result_xor_36_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST124]]) : (ui64) -> ()
    %result_and_36_40 = arc.and %cst_36, %cst_40 : ui64
    "arc.keep"(%result_and_36_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_36_40 = arc.or %cst_36, %cst_40 : ui64
    "arc.keep"(%result_or_36_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_xor_36_40 = arc.xor %cst_36, %cst_40 : ui64
    "arc.keep"(%result_xor_36_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_and_36_36 = arc.and %cst_36, %cst_36 : ui64
    "arc.keep"(%result_and_36_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_or_36_36 = arc.or %cst_36, %cst_36 : ui64
    "arc.keep"(%result_or_36_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST36]]) : (ui64) -> ()
    %result_xor_36_36 = arc.xor %cst_36, %cst_36 : ui64
    "arc.keep"(%result_xor_36_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_and_36_21 = arc.and %cst_36, %cst_21 : ui64
    "arc.keep"(%result_and_36_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST150]]) : (ui64) -> ()
    %result_or_36_21 = arc.or %cst_36, %cst_21 : ui64
    "arc.keep"(%result_or_36_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST146]]) : (ui64) -> ()
    %result_xor_36_21 = arc.xor %cst_36, %cst_21 : ui64
    "arc.keep"(%result_xor_36_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST100]]) : (ui64) -> ()
    %result_and_21_92 = arc.and %cst_21, %cst_92 : ui64
    "arc.keep"(%result_and_21_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST13]]) : (ui64) -> ()
    %result_or_21_92 = arc.or %cst_21, %cst_92 : ui64
    "arc.keep"(%result_or_21_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST29]]) : (ui64) -> ()
    %result_xor_21_92 = arc.xor %cst_21, %cst_92 : ui64
    "arc.keep"(%result_xor_21_92) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST122]]) : (ui64) -> ()
    %result_and_21_102 = arc.and %cst_21, %cst_102 : ui64
    "arc.keep"(%result_and_21_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_or_21_102 = arc.or %cst_21, %cst_102 : ui64
    "arc.keep"(%result_or_21_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST102]]) : (ui64) -> ()
    %result_xor_21_102 = arc.xor %cst_21, %cst_102 : ui64
    "arc.keep"(%result_xor_21_102) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST149]]) : (ui64) -> ()
    %result_and_21_40 = arc.and %cst_21, %cst_40 : ui64
    "arc.keep"(%result_and_21_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_or_21_40 = arc.or %cst_21, %cst_40 : ui64
    "arc.keep"(%result_or_21_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_xor_21_40 = arc.xor %cst_21, %cst_40 : ui64
    "arc.keep"(%result_xor_21_40) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_and_21_36 = arc.and %cst_21, %cst_36 : ui64
    "arc.keep"(%result_and_21_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST150]]) : (ui64) -> ()
    %result_or_21_36 = arc.or %cst_21, %cst_36 : ui64
    "arc.keep"(%result_or_21_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST146]]) : (ui64) -> ()
    %result_xor_21_36 = arc.xor %cst_21, %cst_36 : ui64
    "arc.keep"(%result_xor_21_36) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST100]]) : (ui64) -> ()
    %result_and_21_21 = arc.and %cst_21, %cst_21 : ui64
    "arc.keep"(%result_and_21_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_or_21_21 = arc.or %cst_21, %cst_21 : ui64
    "arc.keep"(%result_or_21_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST21]]) : (ui64) -> ()
    %result_xor_21_21 = arc.xor %cst_21, %cst_21 : ui64
    "arc.keep"(%result_xor_21_21) : (ui64) -> ()
    // CHECK: "arc.keep"([[CST40]]) : (ui64) -> ()
    %result_and_70_70 = arc.and %cst_70, %cst_70 : si64
    "arc.keep"(%result_and_70_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_or_70_70 = arc.or %cst_70, %cst_70 : si64
    "arc.keep"(%result_or_70_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_70_70 = arc.xor %cst_70, %cst_70 : si64
    "arc.keep"(%result_xor_70_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_and_70_10 = arc.and %cst_70, %cst_10 : si64
    "arc.keep"(%result_and_70_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_or_70_10 = arc.or %cst_70, %cst_10 : si64
    "arc.keep"(%result_or_70_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_70_10 = arc.xor %cst_70, %cst_10 : si64
    "arc.keep"(%result_xor_70_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST117]]) : (si64) -> ()
    %result_and_70_85 = arc.and %cst_70, %cst_85 : si64
    "arc.keep"(%result_and_70_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST128]]) : (si64) -> ()
    %result_or_70_85 = arc.or %cst_70, %cst_85 : si64
    "arc.keep"(%result_or_70_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_xor_70_85 = arc.xor %cst_70, %cst_85 : si64
    "arc.keep"(%result_xor_70_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST60]]) : (si64) -> ()
    %result_and_70_16 = arc.and %cst_70, %cst_16 : si64
    "arc.keep"(%result_and_70_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_70_16 = arc.or %cst_70, %cst_16 : si64
    "arc.keep"(%result_or_70_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_70_16 = arc.xor %cst_70, %cst_16 : si64
    "arc.keep"(%result_xor_70_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_and_70_110 = arc.and %cst_70, %cst_110 : si64
    "arc.keep"(%result_and_70_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_70_110 = arc.or %cst_70, %cst_110 : si64
    "arc.keep"(%result_or_70_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_xor_70_110 = arc.xor %cst_70, %cst_110 : si64
    "arc.keep"(%result_xor_70_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_and_70_148 = arc.and %cst_70, %cst_148 : si64
    "arc.keep"(%result_and_70_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_or_70_148 = arc.or %cst_70, %cst_148 : si64
    "arc.keep"(%result_or_70_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_70_148 = arc.xor %cst_70, %cst_148 : si64
    "arc.keep"(%result_xor_70_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST58]]) : (si64) -> ()
    %result_and_10_70 = arc.and %cst_10, %cst_70 : si64
    "arc.keep"(%result_and_10_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_or_10_70 = arc.or %cst_10, %cst_70 : si64
    "arc.keep"(%result_or_10_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_10_70 = arc.xor %cst_10, %cst_70 : si64
    "arc.keep"(%result_xor_10_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST117]]) : (si64) -> ()
    %result_and_10_10 = arc.and %cst_10, %cst_10 : si64
    "arc.keep"(%result_and_10_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_or_10_10 = arc.or %cst_10, %cst_10 : si64
    "arc.keep"(%result_or_10_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_xor_10_10 = arc.xor %cst_10, %cst_10 : si64
    "arc.keep"(%result_xor_10_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_and_10_85 = arc.and %cst_10, %cst_85 : si64
    "arc.keep"(%result_and_10_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST135]]) : (si64) -> ()
    %result_or_10_85 = arc.or %cst_10, %cst_85 : si64
    "arc.keep"(%result_or_10_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST136]]) : (si64) -> ()
    %result_xor_10_85 = arc.xor %cst_10, %cst_85 : si64
    "arc.keep"(%result_xor_10_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST89]]) : (si64) -> ()
    %result_and_10_16 = arc.and %cst_10, %cst_16 : si64
    "arc.keep"(%result_and_10_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_10_16 = arc.or %cst_10, %cst_16 : si64
    "arc.keep"(%result_or_10_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_xor_10_16 = arc.xor %cst_10, %cst_16 : si64
    "arc.keep"(%result_xor_10_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_and_10_110 = arc.and %cst_10, %cst_110 : si64
    "arc.keep"(%result_and_10_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_10_110 = arc.or %cst_10, %cst_110 : si64
    "arc.keep"(%result_or_10_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST12]]) : (si64) -> ()
    %result_xor_10_110 = arc.xor %cst_10, %cst_110 : si64
    "arc.keep"(%result_xor_10_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST12]]) : (si64) -> ()
    %result_and_10_148 = arc.and %cst_10, %cst_148 : si64
    "arc.keep"(%result_and_10_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST95]]) : (si64) -> ()
    %result_or_10_148 = arc.or %cst_10, %cst_148 : si64
    "arc.keep"(%result_or_10_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST131]]) : (si64) -> ()
    %result_xor_10_148 = arc.xor %cst_10, %cst_148 : si64
    "arc.keep"(%result_xor_10_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST142]]) : (si64) -> ()
    %result_and_85_70 = arc.and %cst_85, %cst_70 : si64
    "arc.keep"(%result_and_85_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST128]]) : (si64) -> ()
    %result_or_85_70 = arc.or %cst_85, %cst_70 : si64
    "arc.keep"(%result_or_85_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_xor_85_70 = arc.xor %cst_85, %cst_70 : si64
    "arc.keep"(%result_xor_85_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST60]]) : (si64) -> ()
    %result_and_85_10 = arc.and %cst_85, %cst_10 : si64
    "arc.keep"(%result_and_85_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST135]]) : (si64) -> ()
    %result_or_85_10 = arc.or %cst_85, %cst_10 : si64
    "arc.keep"(%result_or_85_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST136]]) : (si64) -> ()
    %result_xor_85_10 = arc.xor %cst_85, %cst_10 : si64
    "arc.keep"(%result_xor_85_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST89]]) : (si64) -> ()
    %result_and_85_85 = arc.and %cst_85, %cst_85 : si64
    "arc.keep"(%result_and_85_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_or_85_85 = arc.or %cst_85, %cst_85 : si64
    "arc.keep"(%result_or_85_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_xor_85_85 = arc.xor %cst_85, %cst_85 : si64
    "arc.keep"(%result_xor_85_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_and_85_16 = arc.and %cst_85, %cst_16 : si64
    "arc.keep"(%result_and_85_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_85_16 = arc.or %cst_85, %cst_16 : si64
    "arc.keep"(%result_or_85_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_xor_85_16 = arc.xor %cst_85, %cst_16 : si64
    "arc.keep"(%result_xor_85_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_and_85_110 = arc.and %cst_85, %cst_110 : si64
    "arc.keep"(%result_and_85_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_or_85_110 = arc.or %cst_85, %cst_110 : si64
    "arc.keep"(%result_or_85_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_xor_85_110 = arc.xor %cst_85, %cst_110 : si64
    "arc.keep"(%result_xor_85_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST128]]) : (si64) -> ()
    %result_and_85_148 = arc.and %cst_85, %cst_148 : si64
    "arc.keep"(%result_and_85_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST5]]) : (si64) -> ()
    %result_or_85_148 = arc.or %cst_85, %cst_148 : si64
    "arc.keep"(%result_or_85_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST28]]) : (si64) -> ()
    %result_xor_85_148 = arc.xor %cst_85, %cst_148 : si64
    "arc.keep"(%result_xor_85_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST2]]) : (si64) -> ()
    %result_and_16_70 = arc.and %cst_16, %cst_70 : si64
    "arc.keep"(%result_and_16_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_70 = arc.or %cst_16, %cst_70 : si64
    "arc.keep"(%result_or_16_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_16_70 = arc.xor %cst_16, %cst_70 : si64
    "arc.keep"(%result_xor_16_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_and_16_10 = arc.and %cst_16, %cst_10 : si64
    "arc.keep"(%result_and_16_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_10 = arc.or %cst_16, %cst_10 : si64
    "arc.keep"(%result_or_16_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_xor_16_10 = arc.xor %cst_16, %cst_10 : si64
    "arc.keep"(%result_xor_16_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST10]]) : (si64) -> ()
    %result_and_16_85 = arc.and %cst_16, %cst_85 : si64
    "arc.keep"(%result_and_16_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_85 = arc.or %cst_16, %cst_85 : si64
    "arc.keep"(%result_or_16_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_xor_16_85 = arc.xor %cst_16, %cst_85 : si64
    "arc.keep"(%result_xor_16_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_and_16_16 = arc.and %cst_16, %cst_16 : si64
    "arc.keep"(%result_and_16_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_16 = arc.or %cst_16, %cst_16 : si64
    "arc.keep"(%result_or_16_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_xor_16_16 = arc.xor %cst_16, %cst_16 : si64
    "arc.keep"(%result_xor_16_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_and_16_110 = arc.and %cst_16, %cst_110 : si64
    "arc.keep"(%result_and_16_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_110 = arc.or %cst_16, %cst_110 : si64
    "arc.keep"(%result_or_16_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_xor_16_110 = arc.xor %cst_16, %cst_110 : si64
    "arc.keep"(%result_xor_16_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_and_16_148 = arc.and %cst_16, %cst_148 : si64
    "arc.keep"(%result_and_16_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_16_148 = arc.or %cst_16, %cst_148 : si64
    "arc.keep"(%result_or_16_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_xor_16_148 = arc.xor %cst_16, %cst_148 : si64
    "arc.keep"(%result_xor_16_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_and_110_70 = arc.and %cst_110, %cst_70 : si64
    "arc.keep"(%result_and_110_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_110_70 = arc.or %cst_110, %cst_70 : si64
    "arc.keep"(%result_or_110_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_xor_110_70 = arc.xor %cst_110, %cst_70 : si64
    "arc.keep"(%result_xor_110_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST125]]) : (si64) -> ()
    %result_and_110_10 = arc.and %cst_110, %cst_10 : si64
    "arc.keep"(%result_and_110_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_110_10 = arc.or %cst_110, %cst_10 : si64
    "arc.keep"(%result_or_110_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST12]]) : (si64) -> ()
    %result_xor_110_10 = arc.xor %cst_110, %cst_10 : si64
    "arc.keep"(%result_xor_110_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST12]]) : (si64) -> ()
    %result_and_110_85 = arc.and %cst_110, %cst_85 : si64
    "arc.keep"(%result_and_110_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_or_110_85 = arc.or %cst_110, %cst_85 : si64
    "arc.keep"(%result_or_110_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST85]]) : (si64) -> ()
    %result_xor_110_85 = arc.xor %cst_110, %cst_85 : si64
    "arc.keep"(%result_xor_110_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST128]]) : (si64) -> ()
    %result_and_110_16 = arc.and %cst_110, %cst_16 : si64
    "arc.keep"(%result_and_110_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_110_16 = arc.or %cst_110, %cst_16 : si64
    "arc.keep"(%result_or_110_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_xor_110_16 = arc.xor %cst_110, %cst_16 : si64
    "arc.keep"(%result_xor_110_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_and_110_110 = arc.and %cst_110, %cst_110 : si64
    "arc.keep"(%result_and_110_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_or_110_110 = arc.or %cst_110, %cst_110 : si64
    "arc.keep"(%result_or_110_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST110]]) : (si64) -> ()
    %result_xor_110_110 = arc.xor %cst_110, %cst_110 : si64
    "arc.keep"(%result_xor_110_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_and_110_148 = arc.and %cst_110, %cst_148 : si64
    "arc.keep"(%result_and_110_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_110_148 = arc.or %cst_110, %cst_148 : si64
    "arc.keep"(%result_or_110_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST114]]) : (si64) -> ()
    %result_xor_110_148 = arc.xor %cst_110, %cst_148 : si64
    "arc.keep"(%result_xor_110_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST114]]) : (si64) -> ()
    %result_and_148_70 = arc.and %cst_148, %cst_70 : si64
    "arc.keep"(%result_and_148_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_or_148_70 = arc.or %cst_148, %cst_70 : si64
    "arc.keep"(%result_or_148_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST70]]) : (si64) -> ()
    %result_xor_148_70 = arc.xor %cst_148, %cst_70 : si64
    "arc.keep"(%result_xor_148_70) : (si64) -> ()
    // CHECK: "arc.keep"([[CST58]]) : (si64) -> ()
    %result_and_148_10 = arc.and %cst_148, %cst_10 : si64
    "arc.keep"(%result_and_148_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST95]]) : (si64) -> ()
    %result_or_148_10 = arc.or %cst_148, %cst_10 : si64
    "arc.keep"(%result_or_148_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST131]]) : (si64) -> ()
    %result_xor_148_10 = arc.xor %cst_148, %cst_10 : si64
    "arc.keep"(%result_xor_148_10) : (si64) -> ()
    // CHECK: "arc.keep"([[CST142]]) : (si64) -> ()
    %result_and_148_85 = arc.and %cst_148, %cst_85 : si64
    "arc.keep"(%result_and_148_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST5]]) : (si64) -> ()
    %result_or_148_85 = arc.or %cst_148, %cst_85 : si64
    "arc.keep"(%result_or_148_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST28]]) : (si64) -> ()
    %result_xor_148_85 = arc.xor %cst_148, %cst_85 : si64
    "arc.keep"(%result_xor_148_85) : (si64) -> ()
    // CHECK: "arc.keep"([[CST2]]) : (si64) -> ()
    %result_and_148_16 = arc.and %cst_148, %cst_16 : si64
    "arc.keep"(%result_and_148_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_148_16 = arc.or %cst_148, %cst_16 : si64
    "arc.keep"(%result_or_148_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_xor_148_16 = arc.xor %cst_148, %cst_16 : si64
    "arc.keep"(%result_xor_148_16) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_and_148_110 = arc.and %cst_148, %cst_110 : si64
    "arc.keep"(%result_and_148_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    %result_or_148_110 = arc.or %cst_148, %cst_110 : si64
    "arc.keep"(%result_or_148_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST114]]) : (si64) -> ()
    %result_xor_148_110 = arc.xor %cst_148, %cst_110 : si64
    "arc.keep"(%result_xor_148_110) : (si64) -> ()
    // CHECK: "arc.keep"([[CST114]]) : (si64) -> ()
    %result_and_148_148 = arc.and %cst_148, %cst_148 : si64
    "arc.keep"(%result_and_148_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_or_148_148 = arc.or %cst_148, %cst_148 : si64
    "arc.keep"(%result_or_148_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST148]]) : (si64) -> ()
    %result_xor_148_148 = arc.xor %cst_148, %cst_148 : si64
    "arc.keep"(%result_xor_148_148) : (si64) -> ()
    // CHECK: "arc.keep"([[CST16]]) : (si64) -> ()
    return
  }
}
