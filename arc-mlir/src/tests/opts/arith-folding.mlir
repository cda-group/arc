// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func @main(%arg0 : i1) {

    %cst_si16n32768 = arc.constant -32768: si16
    // CHECK-DAG: [[CSTsi16n32768:%[^ ]+]] = arc.constant -32768 : si16
    %cst_si16n32767 = arc.constant -32767: si16
    // CHECK-DAG: [[CSTsi16n32767:%[^ ]+]] = arc.constant -32767 : si16
    %cst_si16n32547 = arc.constant -32547: si16
    // CHECK-DAG: [[CSTsi16n32547:%[^ ]+]] = arc.constant -32547 : si16
    %cst_si16n22282 = arc.constant -22282: si16
    // CHECK-DAG: [[CSTsi16n22282:%[^ ]+]] = arc.constant -22282 : si16
    %cst_si16n22281 = arc.constant -22281: si16
    // CHECK-DAG: [[CSTsi16n22281:%[^ ]+]] = arc.constant -22281 : si16
    %cst_si16n22061 = arc.constant -22061: si16
    // CHECK-DAG: [[CSTsi16n22061:%[^ ]+]] = arc.constant -22061 : si16
    %cst_si16n16254 = arc.constant -16254: si16
    // CHECK-DAG: [[CSTsi16n16254:%[^ ]+]] = arc.constant -16254 : si16
    %cst_si16n16253 = arc.constant -16253: si16
    // CHECK-DAG: [[CSTsi16n16253:%[^ ]+]] = arc.constant -16253 : si16
    %cst_si16n16033 = arc.constant -16033: si16
    // CHECK-DAG: [[CSTsi16n16033:%[^ ]+]] = arc.constant -16033 : si16
    %cst_si16n2 = arc.constant -2: si16
    // CHECK-DAG: [[CSTsi16n2:%[^ ]+]] = arc.constant -2 : si16
    %cst_si16n1 = arc.constant -1: si16
    // CHECK-DAG: [[CSTsi16n1:%[^ ]+]] = arc.constant -1 : si16
    %cst_si160 = arc.constant 0: si16
    // CHECK-DAG: [[CSTsi160:%[^ ]+]] = arc.constant 0 : si16
    %cst_si16219 = arc.constant 219: si16
    // CHECK-DAG: [[CSTsi16219:%[^ ]+]] = arc.constant 219 : si16
    %cst_si16220 = arc.constant 220: si16
    // CHECK-DAG: [[CSTsi16220:%[^ ]+]] = arc.constant 220 : si16
    %cst_si1610486 = arc.constant 10486: si16
    // CHECK-DAG: [[CSTsi1610486:%[^ ]+]] = arc.constant 10486 : si16
    %cst_si1616514 = arc.constant 16514: si16
    // CHECK-DAG: [[CSTsi1616514:%[^ ]+]] = arc.constant 16514 : si16
    %cst_si1620972 = arc.constant 20972: si16
    // CHECK-DAG: [[CSTsi1620972:%[^ ]+]] = arc.constant 20972 : si16
    %cst_si1627000 = arc.constant 27000: si16
    // CHECK-DAG: [[CSTsi1627000:%[^ ]+]] = arc.constant 27000 : si16
    %cst_si1632766 = arc.constant 32766: si16
    // CHECK-DAG: [[CSTsi1632766:%[^ ]+]] = arc.constant 32766 : si16
    %cst_si1632767 = arc.constant 32767: si16
    // CHECK-DAG: [[CSTsi1632767:%[^ ]+]] = arc.constant 32767 : si16
    %cst_si32n2147483648 = arc.constant -2147483648: si32
    // CHECK-DAG: [[CSTsi32n2147483648:%[^ ]+]] = arc.constant -2147483648 : si32
    %cst_si32n2147483647 = arc.constant -2147483647: si32
    // CHECK-DAG: [[CSTsi32n2147483647:%[^ ]+]] = arc.constant -2147483647 : si32
    %cst_si32n2070811526 = arc.constant -2070811526: si32
    // CHECK-DAG: [[CSTsi32n2070811526:%[^ ]+]] = arc.constant -2070811526 : si32
    %cst_si32n1713183800 = arc.constant -1713183800: si32
    // CHECK-DAG: [[CSTsi32n1713183800:%[^ ]+]] = arc.constant -1713183800 : si32
    %cst_si32n1252582164 = arc.constant -1252582164: si32
    // CHECK-DAG: [[CSTsi32n1252582164:%[^ ]+]] = arc.constant -1252582164 : si32
    %cst_si32n1035405763 = arc.constant -1035405763: si32
    // CHECK-DAG: [[CSTsi32n1035405763:%[^ ]+]] = arc.constant -1035405763 : si32
    %cst_si32n2 = arc.constant -2: si32
    // CHECK-DAG: [[CSTsi32n2:%[^ ]+]] = arc.constant -2 : si32
    %cst_si32n1 = arc.constant -1: si32
    // CHECK-DAG: [[CSTsi32n1:%[^ ]+]] = arc.constant -1 : si32
    %cst_si320 = arc.constant 0: si32
    // CHECK-DAG: [[CSTsi320:%[^ ]+]] = arc.constant 0 : si32
    %cst_si32434299846 = arc.constant 434299846: si32
    // CHECK-DAG: [[CSTsi32434299846:%[^ ]+]] = arc.constant 434299846 : si32
    %cst_si32434299847 = arc.constant 434299847: si32
    // CHECK-DAG: [[CSTsi32434299847:%[^ ]+]] = arc.constant 434299847 : si32
    %cst_si32894901482 = arc.constant 894901482: si32
    // CHECK-DAG: [[CSTsi32894901482:%[^ ]+]] = arc.constant 894901482 : si32
    %cst_si32894901483 = arc.constant 894901483: si32
    // CHECK-DAG: [[CSTsi32894901483:%[^ ]+]] = arc.constant 894901483 : si32
    %cst_si321112077883 = arc.constant 1112077883: si32
    // CHECK-DAG: [[CSTsi321112077883:%[^ ]+]] = arc.constant 1112077883 : si32
    %cst_si321112077884 = arc.constant 1112077884: si32
    // CHECK-DAG: [[CSTsi321112077884:%[^ ]+]] = arc.constant 1112077884 : si32
    %cst_si322147483646 = arc.constant 2147483646: si32
    // CHECK-DAG: [[CSTsi322147483646:%[^ ]+]] = arc.constant 2147483646 : si32
    %cst_si322147483647 = arc.constant 2147483647: si32
    // CHECK-DAG: [[CSTsi322147483647:%[^ ]+]] = arc.constant 2147483647 : si32
    %cst_si64n9223372036854775808 = arc.constant -9223372036854775808: si64
    // CHECK-DAG: [[CSTsi64n9223372036854775808:%[^ ]+]] = arc.constant -9223372036854775808 : si64
    %cst_si64n9223372036854775807 = arc.constant -9223372036854775807: si64
    // CHECK-DAG: [[CSTsi64n9223372036854775807:%[^ ]+]] = arc.constant -9223372036854775807 : si64
    %cst_si64n3646223071723659264 = arc.constant -3646223071723659264: si64
    // CHECK-DAG: [[CSTsi64n3646223071723659264:%[^ ]+]] = arc.constant -3646223071723659264 : si64
    %cst_si64n3646223071723659263 = arc.constant -3646223071723659263: si64
    // CHECK-DAG: [[CSTsi64n3646223071723659263:%[^ ]+]] = arc.constant -3646223071723659263 : si64
    %cst_si64n3483854430320017408 = arc.constant -3483854430320017408: si64
    // CHECK-DAG: [[CSTsi64n3483854430320017408:%[^ ]+]] = arc.constant -3483854430320017408 : si64
    %cst_si64n3070198554514583552 = arc.constant -3070198554514583552: si64
    // CHECK-DAG: [[CSTsi64n3070198554514583552:%[^ ]+]] = arc.constant -3070198554514583552 : si64
    %cst_si64n2656542678709149696 = arc.constant -2656542678709149696: si64
    // CHECK-DAG: [[CSTsi64n2656542678709149696:%[^ ]+]] = arc.constant -2656542678709149696 : si64
    %cst_si64n1741927215160008704 = arc.constant -1741927215160008704: si64
    // CHECK-DAG: [[CSTsi64n1741927215160008704:%[^ ]+]] = arc.constant -1741927215160008704 : si64
    %cst_si64n1328271339354574848 = arc.constant -1328271339354574848: si64
    // CHECK-DAG: [[CSTsi64n1328271339354574848:%[^ ]+]] = arc.constant -1328271339354574848 : si64
    %cst_si64n2 = arc.constant -2: si64
    // CHECK-DAG: [[CSTsi64n2:%[^ ]+]] = arc.constant -2 : si64
    %cst_si64n1 = arc.constant -1: si64
    // CHECK-DAG: [[CSTsi64n1:%[^ ]+]] = arc.constant -1 : si64
    %cst_si640 = arc.constant 0: si64
    // CHECK-DAG: [[CSTsi640:%[^ ]+]] = arc.constant 0 : si64
    %cst_si643835221749971107840 = arc.constant 3835221749971107840: si64
    // CHECK-DAG: [[CSTsi643835221749971107840:%[^ ]+]] = arc.constant 3835221749971107840 : si64
    %cst_si644248877625776541696 = arc.constant 4248877625776541696: si64
    // CHECK-DAG: [[CSTsi644248877625776541696:%[^ ]+]] = arc.constant 4248877625776541696 : si64
    %cst_si645577148965131116544 = arc.constant 5577148965131116544: si64
    // CHECK-DAG: [[CSTsi645577148965131116544:%[^ ]+]] = arc.constant 5577148965131116544 : si64
    %cst_si647481444821694767102 = arc.constant 7481444821694767102: si64
    // CHECK-DAG: [[CSTsi647481444821694767102:%[^ ]+]] = arc.constant 7481444821694767102 : si64
    %cst_si647481444821694767103 = arc.constant 7481444821694767103: si64
    // CHECK-DAG: [[CSTsi647481444821694767103:%[^ ]+]] = arc.constant 7481444821694767103 : si64
    %cst_si647895100697500200958 = arc.constant 7895100697500200958: si64
    // CHECK-DAG: [[CSTsi647895100697500200958:%[^ ]+]] = arc.constant 7895100697500200958 : si64
    %cst_si647895100697500200959 = arc.constant 7895100697500200959: si64
    // CHECK-DAG: [[CSTsi647895100697500200959:%[^ ]+]] = arc.constant 7895100697500200959 : si64
    %cst_si649223372036854775806 = arc.constant 9223372036854775806: si64
    // CHECK-DAG: [[CSTsi649223372036854775806:%[^ ]+]] = arc.constant 9223372036854775806 : si64
    %cst_si649223372036854775807 = arc.constant 9223372036854775807: si64
    // CHECK-DAG: [[CSTsi649223372036854775807:%[^ ]+]] = arc.constant 9223372036854775807 : si64
    %cst_si8n128 = arc.constant -128: si8
    // CHECK-DAG: [[CSTsi8n128:%[^ ]+]] = arc.constant -128 : si8
    %cst_si8n127 = arc.constant -127: si8
    // CHECK-DAG: [[CSTsi8n127:%[^ ]+]] = arc.constant -127 : si8
    %cst_si8n126 = arc.constant -126: si8
    // CHECK-DAG: [[CSTsi8n126:%[^ ]+]] = arc.constant -126 : si8
    %cst_si8n112 = arc.constant -112: si8
    // CHECK-DAG: [[CSTsi8n112:%[^ ]+]] = arc.constant -112 : si8
    %cst_si8n111 = arc.constant -111: si8
    // CHECK-DAG: [[CSTsi8n111:%[^ ]+]] = arc.constant -111 : si8
    %cst_si8n2 = arc.constant -2: si8
    // CHECK-DAG: [[CSTsi8n2:%[^ ]+]] = arc.constant -2 : si8
    %cst_si8n1 = arc.constant -1: si8
    // CHECK-DAG: [[CSTsi8n1:%[^ ]+]] = arc.constant -1 : si8
    %cst_si80 = arc.constant 0: si8
    // CHECK-DAG: [[CSTsi80:%[^ ]+]] = arc.constant 0 : si8
    %cst_si81 = arc.constant 1: si8
    // CHECK-DAG: [[CSTsi81:%[^ ]+]] = arc.constant 1 : si8
    %cst_si82 = arc.constant 2: si8
    // CHECK-DAG: [[CSTsi82:%[^ ]+]] = arc.constant 2 : si8
    %cst_si816 = arc.constant 16: si8
    // CHECK-DAG: [[CSTsi816:%[^ ]+]] = arc.constant 16 : si8
    %cst_si817 = arc.constant 17: si8
    // CHECK-DAG: [[CSTsi817:%[^ ]+]] = arc.constant 17 : si8
    %cst_si832 = arc.constant 32: si8
    // CHECK-DAG: [[CSTsi832:%[^ ]+]] = arc.constant 32 : si8
    %cst_si8126 = arc.constant 126: si8
    // CHECK-DAG: [[CSTsi8126:%[^ ]+]] = arc.constant 126 : si8
    %cst_si8127 = arc.constant 127: si8
    // CHECK-DAG: [[CSTsi8127:%[^ ]+]] = arc.constant 127 : si8
    %cst_ui160 = arc.constant 0: ui16
    // CHECK-DAG: [[CSTui160:%[^ ]+]] = arc.constant 0 : ui16
    %cst_ui161 = arc.constant 1: ui16
    // CHECK-DAG: [[CSTui161:%[^ ]+]] = arc.constant 1 : ui16
    %cst_ui162 = arc.constant 2: ui16
    // CHECK-DAG: [[CSTui162:%[^ ]+]] = arc.constant 2 : ui16
    %cst_ui161717 = arc.constant 1717: ui16
    // CHECK-DAG: [[CSTui161717:%[^ ]+]] = arc.constant 1717 : ui16
    %cst_ui161718 = arc.constant 1718: ui16
    // CHECK-DAG: [[CSTui161718:%[^ ]+]] = arc.constant 1718 : ui16
    %cst_ui163434 = arc.constant 3434: ui16
    // CHECK-DAG: [[CSTui163434:%[^ ]+]] = arc.constant 3434 : ui16
    %cst_ui1617988 = arc.constant 17988: ui16
    // CHECK-DAG: [[CSTui1617988:%[^ ]+]] = arc.constant 17988 : ui16
    %cst_ui1617989 = arc.constant 17989: ui16
    // CHECK-DAG: [[CSTui1617989:%[^ ]+]] = arc.constant 17989 : ui16
    %cst_ui1619705 = arc.constant 19705: ui16
    // CHECK-DAG: [[CSTui1619705:%[^ ]+]] = arc.constant 19705 : ui16
    %cst_ui1635976 = arc.constant 35976: ui16
    // CHECK-DAG: [[CSTui1635976:%[^ ]+]] = arc.constant 35976 : ui16
    %cst_ui1665096 = arc.constant 65096: ui16
    // CHECK-DAG: [[CSTui1665096:%[^ ]+]] = arc.constant 65096 : ui16
    %cst_ui1665097 = arc.constant 65097: ui16
    // CHECK-DAG: [[CSTui1665097:%[^ ]+]] = arc.constant 65097 : ui16
    %cst_ui1665534 = arc.constant 65534: ui16
    // CHECK-DAG: [[CSTui1665534:%[^ ]+]] = arc.constant 65534 : ui16
    %cst_ui1665535 = arc.constant 65535: ui16
    // CHECK-DAG: [[CSTui1665535:%[^ ]+]] = arc.constant 65535 : ui16
    %cst_ui320 = arc.constant 0: ui32
    // CHECK-DAG: [[CSTui320:%[^ ]+]] = arc.constant 0 : ui32
    %cst_ui321 = arc.constant 1: ui32
    // CHECK-DAG: [[CSTui321:%[^ ]+]] = arc.constant 1 : ui32
    %cst_ui322 = arc.constant 2: ui32
    // CHECK-DAG: [[CSTui322:%[^ ]+]] = arc.constant 2 : ui32
    %cst_ui322119154652 = arc.constant 2119154652: ui32
    // CHECK-DAG: [[CSTui322119154652:%[^ ]+]] = arc.constant 2119154652 : ui32
    %cst_ui322119154653 = arc.constant 2119154653: ui32
    // CHECK-DAG: [[CSTui322119154653:%[^ ]+]] = arc.constant 2119154653 : ui32
    %cst_ui323002788344 = arc.constant 3002788344: ui32
    // CHECK-DAG: [[CSTui323002788344:%[^ ]+]] = arc.constant 3002788344 : ui32
    %cst_ui323002788345 = arc.constant 3002788345: ui32
    // CHECK-DAG: [[CSTui323002788345:%[^ ]+]] = arc.constant 3002788345 : ui32
    %cst_ui323482297128 = arc.constant 3482297128: ui32
    // CHECK-DAG: [[CSTui323482297128:%[^ ]+]] = arc.constant 3482297128 : ui32
    %cst_ui323482297129 = arc.constant 3482297129: ui32
    // CHECK-DAG: [[CSTui323482297129:%[^ ]+]] = arc.constant 3482297129 : ui32
    %cst_ui324238309304 = arc.constant 4238309304: ui32
    // CHECK-DAG: [[CSTui324238309304:%[^ ]+]] = arc.constant 4238309304 : ui32
    %cst_ui324294967294 = arc.constant 4294967294: ui32
    // CHECK-DAG: [[CSTui324294967294:%[^ ]+]] = arc.constant 4294967294 : ui32
    %cst_ui324294967295 = arc.constant 4294967295: ui32
    // CHECK-DAG: [[CSTui324294967295:%[^ ]+]] = arc.constant 4294967295 : ui32
    %cst_ui640 = arc.constant 0: ui64
    // CHECK-DAG: [[CSTui640:%[^ ]+]] = arc.constant 0 : ui64
    %cst_ui641 = arc.constant 1: ui64
    // CHECK-DAG: [[CSTui641:%[^ ]+]] = arc.constant 1 : ui64
    %cst_ui642 = arc.constant 2: ui64
    // CHECK-DAG: [[CSTui642:%[^ ]+]] = arc.constant 2 : ui64
    %cst_ui64191084152064409600 = arc.constant 191084152064409600: ui64
    // CHECK-DAG: [[CSTui64191084152064409600:%[^ ]+]] = arc.constant 191084152064409600 : ui64
    %cst_ui64191084152064409601 = arc.constant 191084152064409601: ui64
    // CHECK-DAG: [[CSTui64191084152064409601:%[^ ]+]] = arc.constant 191084152064409601 : ui64
    %cst_ui64382168304128819200 = arc.constant 382168304128819200: ui64
    // CHECK-DAG: [[CSTui64382168304128819200:%[^ ]+]] = arc.constant 382168304128819200 : ui64
    %cst_ui6411015955194427482112 = arc.constant 11015955194427482112: ui64
    // CHECK-DAG: [[CSTui6411015955194427482112:%[^ ]+]] = arc.constant 11015955194427482112 : ui64
    %cst_ui6411015955194427482113 = arc.constant 11015955194427482113: ui64
    // CHECK-DAG: [[CSTui6411015955194427482113:%[^ ]+]] = arc.constant 11015955194427482113 : ui64
    %cst_ui6411207039346491891712 = arc.constant 11207039346491891712: ui64
    // CHECK-DAG: [[CSTui6411207039346491891712:%[^ ]+]] = arc.constant 11207039346491891712 : ui64
    %cst_ui6416990600415051759616 = arc.constant 16990600415051759616: ui64
    // CHECK-DAG: [[CSTui6416990600415051759616:%[^ ]+]] = arc.constant 16990600415051759616 : ui64
    %cst_ui6416990600415051759617 = arc.constant 16990600415051759617: ui64
    // CHECK-DAG: [[CSTui6416990600415051759617:%[^ ]+]] = arc.constant 16990600415051759617 : ui64
    %cst_ui6417181684567116169216 = arc.constant 17181684567116169216: ui64
    // CHECK-DAG: [[CSTui6417181684567116169216:%[^ ]+]] = arc.constant 17181684567116169216 : ui64
    %cst_ui6418446744073709551614 = arc.constant 18446744073709551614: ui64
    // CHECK-DAG: [[CSTui6418446744073709551614:%[^ ]+]] = arc.constant 18446744073709551614 : ui64
    %cst_ui6418446744073709551615 = arc.constant 18446744073709551615: ui64
    // CHECK-DAG: [[CSTui6418446744073709551615:%[^ ]+]] = arc.constant 18446744073709551615 : ui64
    %cst_ui80 = arc.constant 0: ui8
    // CHECK-DAG: [[CSTui80:%[^ ]+]] = arc.constant 0 : ui8
    %cst_ui81 = arc.constant 1: ui8
    // CHECK-DAG: [[CSTui81:%[^ ]+]] = arc.constant 1 : ui8
    %cst_ui82 = arc.constant 2: ui8
    // CHECK-DAG: [[CSTui82:%[^ ]+]] = arc.constant 2 : ui8
    %cst_ui872 = arc.constant 72: ui8
    // CHECK-DAG: [[CSTui872:%[^ ]+]] = arc.constant 72 : ui8
    %cst_ui873 = arc.constant 73: ui8
    // CHECK-DAG: [[CSTui873:%[^ ]+]] = arc.constant 73 : ui8
    %cst_ui8100 = arc.constant 100: ui8
    // CHECK-DAG: [[CSTui8100:%[^ ]+]] = arc.constant 100 : ui8
    %cst_ui8101 = arc.constant 101: ui8
    // CHECK-DAG: [[CSTui8101:%[^ ]+]] = arc.constant 101 : ui8
    %cst_ui8144 = arc.constant 144: ui8
    // CHECK-DAG: [[CSTui8144:%[^ ]+]] = arc.constant 144 : ui8
    %cst_ui8162 = arc.constant 162: ui8
    // CHECK-DAG: [[CSTui8162:%[^ ]+]] = arc.constant 162 : ui8
    %cst_ui8163 = arc.constant 163: ui8
    // CHECK-DAG: [[CSTui8163:%[^ ]+]] = arc.constant 163 : ui8
    %cst_ui8172 = arc.constant 172: ui8
    // CHECK-DAG: [[CSTui8172:%[^ ]+]] = arc.constant 172 : ui8
    %cst_ui8200 = arc.constant 200: ui8
    // CHECK-DAG: [[CSTui8200:%[^ ]+]] = arc.constant 200 : ui8
    %cst_ui8234 = arc.constant 234: ui8
    // CHECK-DAG: [[CSTui8234:%[^ ]+]] = arc.constant 234 : ui8
    %cst_ui8254 = arc.constant 254: ui8
    // CHECK-DAG: [[CSTui8254:%[^ ]+]] = arc.constant 254 : ui8
    %cst_ui8255 = arc.constant 255: ui8
    // CHECK-DAG: [[CSTui8255:%[^ ]+]] = arc.constant 255 : ui8
    // addi -32768, 0 -> -32768
    %result_addi_si16n32768_si160 = arc.addi %cst_si16n32768, %cst_si160 : si16
    "arc.keep"(%result_addi_si16n32768_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32768]]) : (si16) -> ()

    // addi 0, -32768 -> -32768
    %result_addi_si160_si16n32768 = arc.addi %cst_si160, %cst_si16n32768 : si16
    "arc.keep"(%result_addi_si160_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32768]]) : (si16) -> ()

    // addi -32767, 0 -> -32767
    %result_addi_si16n32767_si160 = arc.addi %cst_si16n32767, %cst_si160 : si16
    "arc.keep"(%result_addi_si16n32767_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32767]]) : (si16) -> ()

    // addi 0, -32767 -> -32767
    %result_addi_si160_si16n32767 = arc.addi %cst_si160, %cst_si16n32767 : si16
    "arc.keep"(%result_addi_si160_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32767]]) : (si16) -> ()

    // addi -32547, 0 -> -32547
    %result_addi_si16n32547_si160 = arc.addi %cst_si16n32547, %cst_si160 : si16
    "arc.keep"(%result_addi_si16n32547_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32547]]) : (si16) -> ()

    // addi 0, -32547 -> -32547
    %result_addi_si160_si16n32547 = arc.addi %cst_si160, %cst_si16n32547 : si16
    "arc.keep"(%result_addi_si160_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n32547]]) : (si16) -> ()

    // addi -32768, 10486 -> -22282
    %result_addi_si16n32768_si1610486 = arc.addi %cst_si16n32768, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si16n32768_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22282]]) : (si16) -> ()

    // addi 10486, -32768 -> -22282
    %result_addi_si1610486_si16n32768 = arc.addi %cst_si1610486, %cst_si16n32768 : si16
    "arc.keep"(%result_addi_si1610486_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22282]]) : (si16) -> ()

    // addi -32767, 10486 -> -22281
    %result_addi_si16n32767_si1610486 = arc.addi %cst_si16n32767, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si16n32767_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22281]]) : (si16) -> ()

    // addi 10486, -32767 -> -22281
    %result_addi_si1610486_si16n32767 = arc.addi %cst_si1610486, %cst_si16n32767 : si16
    "arc.keep"(%result_addi_si1610486_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22281]]) : (si16) -> ()

    // addi -32547, 10486 -> -22061
    %result_addi_si16n32547_si1610486 = arc.addi %cst_si16n32547, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si16n32547_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22061]]) : (si16) -> ()

    // addi 10486, -32547 -> -22061
    %result_addi_si1610486_si16n32547 = arc.addi %cst_si1610486, %cst_si16n32547 : si16
    "arc.keep"(%result_addi_si1610486_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n22061]]) : (si16) -> ()

    // addi -32768, 16514 -> -16254
    %result_addi_si16n32768_si1616514 = arc.addi %cst_si16n32768, %cst_si1616514 : si16
    "arc.keep"(%result_addi_si16n32768_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16254]]) : (si16) -> ()

    // addi 16514, -32768 -> -16254
    %result_addi_si1616514_si16n32768 = arc.addi %cst_si1616514, %cst_si16n32768 : si16
    "arc.keep"(%result_addi_si1616514_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16254]]) : (si16) -> ()

    // addi -32767, 16514 -> -16253
    %result_addi_si16n32767_si1616514 = arc.addi %cst_si16n32767, %cst_si1616514 : si16
    "arc.keep"(%result_addi_si16n32767_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16253]]) : (si16) -> ()

    // addi 16514, -32767 -> -16253
    %result_addi_si1616514_si16n32767 = arc.addi %cst_si1616514, %cst_si16n32767 : si16
    "arc.keep"(%result_addi_si1616514_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16253]]) : (si16) -> ()

    // addi -32547, 16514 -> -16033
    %result_addi_si16n32547_si1616514 = arc.addi %cst_si16n32547, %cst_si1616514 : si16
    "arc.keep"(%result_addi_si16n32547_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16033]]) : (si16) -> ()

    // addi 16514, -32547 -> -16033
    %result_addi_si1616514_si16n32547 = arc.addi %cst_si1616514, %cst_si16n32547 : si16
    "arc.keep"(%result_addi_si1616514_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n16033]]) : (si16) -> ()

    // addi -32768, 32766 -> -2
    %result_addi_si16n32768_si1632766 = arc.addi %cst_si16n32768, %cst_si1632766 : si16
    "arc.keep"(%result_addi_si16n32768_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n2]]) : (si16) -> ()

    // addi 32766, -32768 -> -2
    %result_addi_si1632766_si16n32768 = arc.addi %cst_si1632766, %cst_si16n32768 : si16
    "arc.keep"(%result_addi_si1632766_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n2]]) : (si16) -> ()

    // addi -32768, 32767 -> -1
    %result_addi_si16n32768_si1632767 = arc.addi %cst_si16n32768, %cst_si1632767 : si16
    "arc.keep"(%result_addi_si16n32768_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n1]]) : (si16) -> ()

    // addi -32767, 32766 -> -1
    %result_addi_si16n32767_si1632766 = arc.addi %cst_si16n32767, %cst_si1632766 : si16
    "arc.keep"(%result_addi_si16n32767_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n1]]) : (si16) -> ()

    // addi 32766, -32767 -> -1
    %result_addi_si1632766_si16n32767 = arc.addi %cst_si1632766, %cst_si16n32767 : si16
    "arc.keep"(%result_addi_si1632766_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n1]]) : (si16) -> ()

    // addi 32767, -32768 -> -1
    %result_addi_si1632767_si16n32768 = arc.addi %cst_si1632767, %cst_si16n32768 : si16
    "arc.keep"(%result_addi_si1632767_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16n1]]) : (si16) -> ()

    // addi -32767, 32767 -> 0
    %result_addi_si16n32767_si1632767 = arc.addi %cst_si16n32767, %cst_si1632767 : si16
    "arc.keep"(%result_addi_si16n32767_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi160]]) : (si16) -> ()

    // addi 0, 0 -> 0
    %result_addi_si160_si160 = arc.addi %cst_si160, %cst_si160 : si16
    "arc.keep"(%result_addi_si160_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi160]]) : (si16) -> ()

    // addi 32767, -32767 -> 0
    %result_addi_si1632767_si16n32767 = arc.addi %cst_si1632767, %cst_si16n32767 : si16
    "arc.keep"(%result_addi_si1632767_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi160]]) : (si16) -> ()

    // addi -32547, 32766 -> 219
    %result_addi_si16n32547_si1632766 = arc.addi %cst_si16n32547, %cst_si1632766 : si16
    "arc.keep"(%result_addi_si16n32547_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16219]]) : (si16) -> ()

    // addi 32766, -32547 -> 219
    %result_addi_si1632766_si16n32547 = arc.addi %cst_si1632766, %cst_si16n32547 : si16
    "arc.keep"(%result_addi_si1632766_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16219]]) : (si16) -> ()

    // addi -32547, 32767 -> 220
    %result_addi_si16n32547_si1632767 = arc.addi %cst_si16n32547, %cst_si1632767 : si16
    "arc.keep"(%result_addi_si16n32547_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16220]]) : (si16) -> ()

    // addi 32767, -32547 -> 220
    %result_addi_si1632767_si16n32547 = arc.addi %cst_si1632767, %cst_si16n32547 : si16
    "arc.keep"(%result_addi_si1632767_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi16220]]) : (si16) -> ()

    // addi 0, 10486 -> 10486
    %result_addi_si160_si1610486 = arc.addi %cst_si160, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si160_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1610486]]) : (si16) -> ()

    // addi 10486, 0 -> 10486
    %result_addi_si1610486_si160 = arc.addi %cst_si1610486, %cst_si160 : si16
    "arc.keep"(%result_addi_si1610486_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1610486]]) : (si16) -> ()

    // addi 0, 16514 -> 16514
    %result_addi_si160_si1616514 = arc.addi %cst_si160, %cst_si1616514 : si16
    "arc.keep"(%result_addi_si160_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1616514]]) : (si16) -> ()

    // addi 16514, 0 -> 16514
    %result_addi_si1616514_si160 = arc.addi %cst_si1616514, %cst_si160 : si16
    "arc.keep"(%result_addi_si1616514_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1616514]]) : (si16) -> ()

    // addi 10486, 10486 -> 20972
    %result_addi_si1610486_si1610486 = arc.addi %cst_si1610486, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si1610486_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1620972]]) : (si16) -> ()

    // addi 10486, 16514 -> 27000
    %result_addi_si1610486_si1616514 = arc.addi %cst_si1610486, %cst_si1616514 : si16
    "arc.keep"(%result_addi_si1610486_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1627000]]) : (si16) -> ()

    // addi 16514, 10486 -> 27000
    %result_addi_si1616514_si1610486 = arc.addi %cst_si1616514, %cst_si1610486 : si16
    "arc.keep"(%result_addi_si1616514_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1627000]]) : (si16) -> ()

    // addi 0, 32766 -> 32766
    %result_addi_si160_si1632766 = arc.addi %cst_si160, %cst_si1632766 : si16
    "arc.keep"(%result_addi_si160_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1632766]]) : (si16) -> ()

    // addi 32766, 0 -> 32766
    %result_addi_si1632766_si160 = arc.addi %cst_si1632766, %cst_si160 : si16
    "arc.keep"(%result_addi_si1632766_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1632766]]) : (si16) -> ()

    // addi 0, 32767 -> 32767
    %result_addi_si160_si1632767 = arc.addi %cst_si160, %cst_si1632767 : si16
    "arc.keep"(%result_addi_si160_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1632767]]) : (si16) -> ()

    // addi 32767, 0 -> 32767
    %result_addi_si1632767_si160 = arc.addi %cst_si1632767, %cst_si160 : si16
    "arc.keep"(%result_addi_si1632767_si160) : (si16) -> ()
    // CHECK: "arc.keep"([[CSTsi1632767]]) : (si16) -> ()

    // addi -32768, -32768 -> no-fold
    %result_addi_si16n32768_si16n32768 = arc.addi %cst_si16n32768, %cst_si16n32768 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32768_si16n32768:%[^ ]+]] = arc.addi [[CSTsi16n32768]], [[CSTsi16n32768]] : si16
    "arc.keep"(%result_addi_si16n32768_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32768_si16n32768]]) : (si16) -> ()

    // addi -32768, -32767 -> no-fold
    %result_addi_si16n32768_si16n32767 = arc.addi %cst_si16n32768, %cst_si16n32767 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32768_si16n32767:%[^ ]+]] = arc.addi [[CSTsi16n32768]], [[CSTsi16n32767]] : si16
    "arc.keep"(%result_addi_si16n32768_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32768_si16n32767]]) : (si16) -> ()

    // addi -32768, -32547 -> no-fold
    %result_addi_si16n32768_si16n32547 = arc.addi %cst_si16n32768, %cst_si16n32547 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32768_si16n32547:%[^ ]+]] = arc.addi [[CSTsi16n32768]], [[CSTsi16n32547]] : si16
    "arc.keep"(%result_addi_si16n32768_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32768_si16n32547]]) : (si16) -> ()

    // addi -32767, -32768 -> no-fold
    %result_addi_si16n32767_si16n32768 = arc.addi %cst_si16n32767, %cst_si16n32768 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32767_si16n32768:%[^ ]+]] = arc.addi [[CSTsi16n32767]], [[CSTsi16n32768]] : si16
    "arc.keep"(%result_addi_si16n32767_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32767_si16n32768]]) : (si16) -> ()

    // addi -32767, -32767 -> no-fold
    %result_addi_si16n32767_si16n32767 = arc.addi %cst_si16n32767, %cst_si16n32767 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32767_si16n32767:%[^ ]+]] = arc.addi [[CSTsi16n32767]], [[CSTsi16n32767]] : si16
    "arc.keep"(%result_addi_si16n32767_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32767_si16n32767]]) : (si16) -> ()

    // addi -32767, -32547 -> no-fold
    %result_addi_si16n32767_si16n32547 = arc.addi %cst_si16n32767, %cst_si16n32547 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32767_si16n32547:%[^ ]+]] = arc.addi [[CSTsi16n32767]], [[CSTsi16n32547]] : si16
    "arc.keep"(%result_addi_si16n32767_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32767_si16n32547]]) : (si16) -> ()

    // addi -32547, -32768 -> no-fold
    %result_addi_si16n32547_si16n32768 = arc.addi %cst_si16n32547, %cst_si16n32768 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32547_si16n32768:%[^ ]+]] = arc.addi [[CSTsi16n32547]], [[CSTsi16n32768]] : si16
    "arc.keep"(%result_addi_si16n32547_si16n32768) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32547_si16n32768]]) : (si16) -> ()

    // addi -32547, -32767 -> no-fold
    %result_addi_si16n32547_si16n32767 = arc.addi %cst_si16n32547, %cst_si16n32767 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32547_si16n32767:%[^ ]+]] = arc.addi [[CSTsi16n32547]], [[CSTsi16n32767]] : si16
    "arc.keep"(%result_addi_si16n32547_si16n32767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32547_si16n32767]]) : (si16) -> ()

    // addi -32547, -32547 -> no-fold
    %result_addi_si16n32547_si16n32547 = arc.addi %cst_si16n32547, %cst_si16n32547 : si16
    // CHECK-DAG: [[RESULT_addi_si16n32547_si16n32547:%[^ ]+]] = arc.addi [[CSTsi16n32547]], [[CSTsi16n32547]] : si16
    "arc.keep"(%result_addi_si16n32547_si16n32547) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si16n32547_si16n32547]]) : (si16) -> ()

    // addi 10486, 32766 -> no-fold
    %result_addi_si1610486_si1632766 = arc.addi %cst_si1610486, %cst_si1632766 : si16
    // CHECK-DAG: [[RESULT_addi_si1610486_si1632766:%[^ ]+]] = arc.addi [[CSTsi1610486]], [[CSTsi1632766]] : si16
    "arc.keep"(%result_addi_si1610486_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1610486_si1632766]]) : (si16) -> ()

    // addi 10486, 32767 -> no-fold
    %result_addi_si1610486_si1632767 = arc.addi %cst_si1610486, %cst_si1632767 : si16
    // CHECK-DAG: [[RESULT_addi_si1610486_si1632767:%[^ ]+]] = arc.addi [[CSTsi1610486]], [[CSTsi1632767]] : si16
    "arc.keep"(%result_addi_si1610486_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1610486_si1632767]]) : (si16) -> ()

    // addi 16514, 16514 -> no-fold
    %result_addi_si1616514_si1616514 = arc.addi %cst_si1616514, %cst_si1616514 : si16
    // CHECK-DAG: [[RESULT_addi_si1616514_si1616514:%[^ ]+]] = arc.addi [[CSTsi1616514]], [[CSTsi1616514]] : si16
    "arc.keep"(%result_addi_si1616514_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1616514_si1616514]]) : (si16) -> ()

    // addi 16514, 32766 -> no-fold
    %result_addi_si1616514_si1632766 = arc.addi %cst_si1616514, %cst_si1632766 : si16
    // CHECK-DAG: [[RESULT_addi_si1616514_si1632766:%[^ ]+]] = arc.addi [[CSTsi1616514]], [[CSTsi1632766]] : si16
    "arc.keep"(%result_addi_si1616514_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1616514_si1632766]]) : (si16) -> ()

    // addi 16514, 32767 -> no-fold
    %result_addi_si1616514_si1632767 = arc.addi %cst_si1616514, %cst_si1632767 : si16
    // CHECK-DAG: [[RESULT_addi_si1616514_si1632767:%[^ ]+]] = arc.addi [[CSTsi1616514]], [[CSTsi1632767]] : si16
    "arc.keep"(%result_addi_si1616514_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1616514_si1632767]]) : (si16) -> ()

    // addi 32766, 10486 -> no-fold
    %result_addi_si1632766_si1610486 = arc.addi %cst_si1632766, %cst_si1610486 : si16
    // CHECK-DAG: [[RESULT_addi_si1632766_si1610486:%[^ ]+]] = arc.addi [[CSTsi1632766]], [[CSTsi1610486]] : si16
    "arc.keep"(%result_addi_si1632766_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632766_si1610486]]) : (si16) -> ()

    // addi 32766, 16514 -> no-fold
    %result_addi_si1632766_si1616514 = arc.addi %cst_si1632766, %cst_si1616514 : si16
    // CHECK-DAG: [[RESULT_addi_si1632766_si1616514:%[^ ]+]] = arc.addi [[CSTsi1632766]], [[CSTsi1616514]] : si16
    "arc.keep"(%result_addi_si1632766_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632766_si1616514]]) : (si16) -> ()

    // addi 32766, 32766 -> no-fold
    %result_addi_si1632766_si1632766 = arc.addi %cst_si1632766, %cst_si1632766 : si16
    // CHECK-DAG: [[RESULT_addi_si1632766_si1632766:%[^ ]+]] = arc.addi [[CSTsi1632766]], [[CSTsi1632766]] : si16
    "arc.keep"(%result_addi_si1632766_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632766_si1632766]]) : (si16) -> ()

    // addi 32766, 32767 -> no-fold
    %result_addi_si1632766_si1632767 = arc.addi %cst_si1632766, %cst_si1632767 : si16
    // CHECK-DAG: [[RESULT_addi_si1632766_si1632767:%[^ ]+]] = arc.addi [[CSTsi1632766]], [[CSTsi1632767]] : si16
    "arc.keep"(%result_addi_si1632766_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632766_si1632767]]) : (si16) -> ()

    // addi 32767, 10486 -> no-fold
    %result_addi_si1632767_si1610486 = arc.addi %cst_si1632767, %cst_si1610486 : si16
    // CHECK-DAG: [[RESULT_addi_si1632767_si1610486:%[^ ]+]] = arc.addi [[CSTsi1632767]], [[CSTsi1610486]] : si16
    "arc.keep"(%result_addi_si1632767_si1610486) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632767_si1610486]]) : (si16) -> ()

    // addi 32767, 16514 -> no-fold
    %result_addi_si1632767_si1616514 = arc.addi %cst_si1632767, %cst_si1616514 : si16
    // CHECK-DAG: [[RESULT_addi_si1632767_si1616514:%[^ ]+]] = arc.addi [[CSTsi1632767]], [[CSTsi1616514]] : si16
    "arc.keep"(%result_addi_si1632767_si1616514) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632767_si1616514]]) : (si16) -> ()

    // addi 32767, 32766 -> no-fold
    %result_addi_si1632767_si1632766 = arc.addi %cst_si1632767, %cst_si1632766 : si16
    // CHECK-DAG: [[RESULT_addi_si1632767_si1632766:%[^ ]+]] = arc.addi [[CSTsi1632767]], [[CSTsi1632766]] : si16
    "arc.keep"(%result_addi_si1632767_si1632766) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632767_si1632766]]) : (si16) -> ()

    // addi 32767, 32767 -> no-fold
    %result_addi_si1632767_si1632767 = arc.addi %cst_si1632767, %cst_si1632767 : si16
    // CHECK-DAG: [[RESULT_addi_si1632767_si1632767:%[^ ]+]] = arc.addi [[CSTsi1632767]], [[CSTsi1632767]] : si16
    "arc.keep"(%result_addi_si1632767_si1632767) : (si16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si1632767_si1632767]]) : (si16) -> ()

    // addi -2147483648, 0 -> -2147483648
    %result_addi_si32n2147483648_si320 = arc.addi %cst_si32n2147483648, %cst_si320 : si32
    "arc.keep"(%result_addi_si32n2147483648_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2147483648]]) : (si32) -> ()

    // addi 0, -2147483648 -> -2147483648
    %result_addi_si320_si32n2147483648 = arc.addi %cst_si320, %cst_si32n2147483648 : si32
    "arc.keep"(%result_addi_si320_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2147483648]]) : (si32) -> ()

    // addi -2147483647, 0 -> -2147483647
    %result_addi_si32n2147483647_si320 = arc.addi %cst_si32n2147483647, %cst_si320 : si32
    "arc.keep"(%result_addi_si32n2147483647_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2147483647]]) : (si32) -> ()

    // addi 0, -2147483647 -> -2147483647
    %result_addi_si320_si32n2147483647 = arc.addi %cst_si320, %cst_si32n2147483647 : si32
    "arc.keep"(%result_addi_si320_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2147483647]]) : (si32) -> ()

    // addi -1035405763, -1035405763 -> -2070811526
    %result_addi_si32n1035405763_si32n1035405763 = arc.addi %cst_si32n1035405763, %cst_si32n1035405763 : si32
    "arc.keep"(%result_addi_si32n1035405763_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2070811526]]) : (si32) -> ()

    // addi -1713183800, 0 -> -1713183800
    %result_addi_si32n1713183800_si320 = arc.addi %cst_si32n1713183800, %cst_si320 : si32
    "arc.keep"(%result_addi_si32n1713183800_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1713183800]]) : (si32) -> ()

    // addi 0, -1713183800 -> -1713183800
    %result_addi_si320_si32n1713183800 = arc.addi %cst_si320, %cst_si32n1713183800 : si32
    "arc.keep"(%result_addi_si320_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1713183800]]) : (si32) -> ()

    // addi -1252582164, 0 -> -1252582164
    %result_addi_si32n1252582164_si320 = arc.addi %cst_si32n1252582164, %cst_si320 : si32
    "arc.keep"(%result_addi_si32n1252582164_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1252582164]]) : (si32) -> ()

    // addi 0, -1252582164 -> -1252582164
    %result_addi_si320_si32n1252582164 = arc.addi %cst_si320, %cst_si32n1252582164 : si32
    "arc.keep"(%result_addi_si320_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1252582164]]) : (si32) -> ()

    // addi -1035405763, 0 -> -1035405763
    %result_addi_si32n1035405763_si320 = arc.addi %cst_si32n1035405763, %cst_si320 : si32
    "arc.keep"(%result_addi_si32n1035405763_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1035405763]]) : (si32) -> ()

    // addi 0, -1035405763 -> -1035405763
    %result_addi_si320_si32n1035405763 = arc.addi %cst_si320, %cst_si32n1035405763 : si32
    "arc.keep"(%result_addi_si320_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1035405763]]) : (si32) -> ()

    // addi -2147483648, 2147483646 -> -2
    %result_addi_si32n2147483648_si322147483646 = arc.addi %cst_si32n2147483648, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si32n2147483648_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2]]) : (si32) -> ()

    // addi 2147483646, -2147483648 -> -2
    %result_addi_si322147483646_si32n2147483648 = arc.addi %cst_si322147483646, %cst_si32n2147483648 : si32
    "arc.keep"(%result_addi_si322147483646_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n2]]) : (si32) -> ()

    // addi -2147483648, 2147483647 -> -1
    %result_addi_si32n2147483648_si322147483647 = arc.addi %cst_si32n2147483648, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si32n2147483648_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1]]) : (si32) -> ()

    // addi -2147483647, 2147483646 -> -1
    %result_addi_si32n2147483647_si322147483646 = arc.addi %cst_si32n2147483647, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si32n2147483647_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1]]) : (si32) -> ()

    // addi 2147483646, -2147483647 -> -1
    %result_addi_si322147483646_si32n2147483647 = arc.addi %cst_si322147483646, %cst_si32n2147483647 : si32
    "arc.keep"(%result_addi_si322147483646_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1]]) : (si32) -> ()

    // addi 2147483647, -2147483648 -> -1
    %result_addi_si322147483647_si32n2147483648 = arc.addi %cst_si322147483647, %cst_si32n2147483648 : si32
    "arc.keep"(%result_addi_si322147483647_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32n1]]) : (si32) -> ()

    // addi -2147483647, 2147483647 -> 0
    %result_addi_si32n2147483647_si322147483647 = arc.addi %cst_si32n2147483647, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si32n2147483647_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi320]]) : (si32) -> ()

    // addi 0, 0 -> 0
    %result_addi_si320_si320 = arc.addi %cst_si320, %cst_si320 : si32
    "arc.keep"(%result_addi_si320_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi320]]) : (si32) -> ()

    // addi 2147483647, -2147483647 -> 0
    %result_addi_si322147483647_si32n2147483647 = arc.addi %cst_si322147483647, %cst_si32n2147483647 : si32
    "arc.keep"(%result_addi_si322147483647_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi320]]) : (si32) -> ()

    // addi -1713183800, 2147483646 -> 434299846
    %result_addi_si32n1713183800_si322147483646 = arc.addi %cst_si32n1713183800, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si32n1713183800_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32434299846]]) : (si32) -> ()

    // addi 2147483646, -1713183800 -> 434299846
    %result_addi_si322147483646_si32n1713183800 = arc.addi %cst_si322147483646, %cst_si32n1713183800 : si32
    "arc.keep"(%result_addi_si322147483646_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32434299846]]) : (si32) -> ()

    // addi -1713183800, 2147483647 -> 434299847
    %result_addi_si32n1713183800_si322147483647 = arc.addi %cst_si32n1713183800, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si32n1713183800_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32434299847]]) : (si32) -> ()

    // addi 2147483647, -1713183800 -> 434299847
    %result_addi_si322147483647_si32n1713183800 = arc.addi %cst_si322147483647, %cst_si32n1713183800 : si32
    "arc.keep"(%result_addi_si322147483647_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32434299847]]) : (si32) -> ()

    // addi -1252582164, 2147483646 -> 894901482
    %result_addi_si32n1252582164_si322147483646 = arc.addi %cst_si32n1252582164, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si32n1252582164_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32894901482]]) : (si32) -> ()

    // addi 2147483646, -1252582164 -> 894901482
    %result_addi_si322147483646_si32n1252582164 = arc.addi %cst_si322147483646, %cst_si32n1252582164 : si32
    "arc.keep"(%result_addi_si322147483646_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32894901482]]) : (si32) -> ()

    // addi -1252582164, 2147483647 -> 894901483
    %result_addi_si32n1252582164_si322147483647 = arc.addi %cst_si32n1252582164, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si32n1252582164_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32894901483]]) : (si32) -> ()

    // addi 2147483647, -1252582164 -> 894901483
    %result_addi_si322147483647_si32n1252582164 = arc.addi %cst_si322147483647, %cst_si32n1252582164 : si32
    "arc.keep"(%result_addi_si322147483647_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi32894901483]]) : (si32) -> ()

    // addi -1035405763, 2147483646 -> 1112077883
    %result_addi_si32n1035405763_si322147483646 = arc.addi %cst_si32n1035405763, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si32n1035405763_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi321112077883]]) : (si32) -> ()

    // addi 2147483646, -1035405763 -> 1112077883
    %result_addi_si322147483646_si32n1035405763 = arc.addi %cst_si322147483646, %cst_si32n1035405763 : si32
    "arc.keep"(%result_addi_si322147483646_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi321112077883]]) : (si32) -> ()

    // addi -1035405763, 2147483647 -> 1112077884
    %result_addi_si32n1035405763_si322147483647 = arc.addi %cst_si32n1035405763, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si32n1035405763_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi321112077884]]) : (si32) -> ()

    // addi 2147483647, -1035405763 -> 1112077884
    %result_addi_si322147483647_si32n1035405763 = arc.addi %cst_si322147483647, %cst_si32n1035405763 : si32
    "arc.keep"(%result_addi_si322147483647_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi321112077884]]) : (si32) -> ()

    // addi 0, 2147483646 -> 2147483646
    %result_addi_si320_si322147483646 = arc.addi %cst_si320, %cst_si322147483646 : si32
    "arc.keep"(%result_addi_si320_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi322147483646]]) : (si32) -> ()

    // addi 2147483646, 0 -> 2147483646
    %result_addi_si322147483646_si320 = arc.addi %cst_si322147483646, %cst_si320 : si32
    "arc.keep"(%result_addi_si322147483646_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi322147483646]]) : (si32) -> ()

    // addi 0, 2147483647 -> 2147483647
    %result_addi_si320_si322147483647 = arc.addi %cst_si320, %cst_si322147483647 : si32
    "arc.keep"(%result_addi_si320_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi322147483647]]) : (si32) -> ()

    // addi 2147483647, 0 -> 2147483647
    %result_addi_si322147483647_si320 = arc.addi %cst_si322147483647, %cst_si320 : si32
    "arc.keep"(%result_addi_si322147483647_si320) : (si32) -> ()
    // CHECK: "arc.keep"([[CSTsi322147483647]]) : (si32) -> ()

    // addi -2147483648, -2147483648 -> no-fold
    %result_addi_si32n2147483648_si32n2147483648 = arc.addi %cst_si32n2147483648, %cst_si32n2147483648 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483648_si32n2147483648:%[^ ]+]] = arc.addi [[CSTsi32n2147483648]], [[CSTsi32n2147483648]] : si32
    "arc.keep"(%result_addi_si32n2147483648_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483648_si32n2147483648]]) : (si32) -> ()

    // addi -2147483648, -2147483647 -> no-fold
    %result_addi_si32n2147483648_si32n2147483647 = arc.addi %cst_si32n2147483648, %cst_si32n2147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483648_si32n2147483647:%[^ ]+]] = arc.addi [[CSTsi32n2147483648]], [[CSTsi32n2147483647]] : si32
    "arc.keep"(%result_addi_si32n2147483648_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483648_si32n2147483647]]) : (si32) -> ()

    // addi -2147483648, -1713183800 -> no-fold
    %result_addi_si32n2147483648_si32n1713183800 = arc.addi %cst_si32n2147483648, %cst_si32n1713183800 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483648_si32n1713183800:%[^ ]+]] = arc.addi [[CSTsi32n2147483648]], [[CSTsi32n1713183800]] : si32
    "arc.keep"(%result_addi_si32n2147483648_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483648_si32n1713183800]]) : (si32) -> ()

    // addi -2147483648, -1252582164 -> no-fold
    %result_addi_si32n2147483648_si32n1252582164 = arc.addi %cst_si32n2147483648, %cst_si32n1252582164 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483648_si32n1252582164:%[^ ]+]] = arc.addi [[CSTsi32n2147483648]], [[CSTsi32n1252582164]] : si32
    "arc.keep"(%result_addi_si32n2147483648_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483648_si32n1252582164]]) : (si32) -> ()

    // addi -2147483648, -1035405763 -> no-fold
    %result_addi_si32n2147483648_si32n1035405763 = arc.addi %cst_si32n2147483648, %cst_si32n1035405763 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483648_si32n1035405763:%[^ ]+]] = arc.addi [[CSTsi32n2147483648]], [[CSTsi32n1035405763]] : si32
    "arc.keep"(%result_addi_si32n2147483648_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483648_si32n1035405763]]) : (si32) -> ()

    // addi -2147483647, -2147483648 -> no-fold
    %result_addi_si32n2147483647_si32n2147483648 = arc.addi %cst_si32n2147483647, %cst_si32n2147483648 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483647_si32n2147483648:%[^ ]+]] = arc.addi [[CSTsi32n2147483647]], [[CSTsi32n2147483648]] : si32
    "arc.keep"(%result_addi_si32n2147483647_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483647_si32n2147483648]]) : (si32) -> ()

    // addi -2147483647, -2147483647 -> no-fold
    %result_addi_si32n2147483647_si32n2147483647 = arc.addi %cst_si32n2147483647, %cst_si32n2147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483647_si32n2147483647:%[^ ]+]] = arc.addi [[CSTsi32n2147483647]], [[CSTsi32n2147483647]] : si32
    "arc.keep"(%result_addi_si32n2147483647_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483647_si32n2147483647]]) : (si32) -> ()

    // addi -2147483647, -1713183800 -> no-fold
    %result_addi_si32n2147483647_si32n1713183800 = arc.addi %cst_si32n2147483647, %cst_si32n1713183800 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483647_si32n1713183800:%[^ ]+]] = arc.addi [[CSTsi32n2147483647]], [[CSTsi32n1713183800]] : si32
    "arc.keep"(%result_addi_si32n2147483647_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483647_si32n1713183800]]) : (si32) -> ()

    // addi -2147483647, -1252582164 -> no-fold
    %result_addi_si32n2147483647_si32n1252582164 = arc.addi %cst_si32n2147483647, %cst_si32n1252582164 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483647_si32n1252582164:%[^ ]+]] = arc.addi [[CSTsi32n2147483647]], [[CSTsi32n1252582164]] : si32
    "arc.keep"(%result_addi_si32n2147483647_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483647_si32n1252582164]]) : (si32) -> ()

    // addi -2147483647, -1035405763 -> no-fold
    %result_addi_si32n2147483647_si32n1035405763 = arc.addi %cst_si32n2147483647, %cst_si32n1035405763 : si32
    // CHECK-DAG: [[RESULT_addi_si32n2147483647_si32n1035405763:%[^ ]+]] = arc.addi [[CSTsi32n2147483647]], [[CSTsi32n1035405763]] : si32
    "arc.keep"(%result_addi_si32n2147483647_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n2147483647_si32n1035405763]]) : (si32) -> ()

    // addi -1713183800, -2147483648 -> no-fold
    %result_addi_si32n1713183800_si32n2147483648 = arc.addi %cst_si32n1713183800, %cst_si32n2147483648 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1713183800_si32n2147483648:%[^ ]+]] = arc.addi [[CSTsi32n1713183800]], [[CSTsi32n2147483648]] : si32
    "arc.keep"(%result_addi_si32n1713183800_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1713183800_si32n2147483648]]) : (si32) -> ()

    // addi -1713183800, -2147483647 -> no-fold
    %result_addi_si32n1713183800_si32n2147483647 = arc.addi %cst_si32n1713183800, %cst_si32n2147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1713183800_si32n2147483647:%[^ ]+]] = arc.addi [[CSTsi32n1713183800]], [[CSTsi32n2147483647]] : si32
    "arc.keep"(%result_addi_si32n1713183800_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1713183800_si32n2147483647]]) : (si32) -> ()

    // addi -1713183800, -1713183800 -> no-fold
    %result_addi_si32n1713183800_si32n1713183800 = arc.addi %cst_si32n1713183800, %cst_si32n1713183800 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1713183800_si32n1713183800:%[^ ]+]] = arc.addi [[CSTsi32n1713183800]], [[CSTsi32n1713183800]] : si32
    "arc.keep"(%result_addi_si32n1713183800_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1713183800_si32n1713183800]]) : (si32) -> ()

    // addi -1713183800, -1252582164 -> no-fold
    %result_addi_si32n1713183800_si32n1252582164 = arc.addi %cst_si32n1713183800, %cst_si32n1252582164 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1713183800_si32n1252582164:%[^ ]+]] = arc.addi [[CSTsi32n1713183800]], [[CSTsi32n1252582164]] : si32
    "arc.keep"(%result_addi_si32n1713183800_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1713183800_si32n1252582164]]) : (si32) -> ()

    // addi -1713183800, -1035405763 -> no-fold
    %result_addi_si32n1713183800_si32n1035405763 = arc.addi %cst_si32n1713183800, %cst_si32n1035405763 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1713183800_si32n1035405763:%[^ ]+]] = arc.addi [[CSTsi32n1713183800]], [[CSTsi32n1035405763]] : si32
    "arc.keep"(%result_addi_si32n1713183800_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1713183800_si32n1035405763]]) : (si32) -> ()

    // addi -1252582164, -2147483648 -> no-fold
    %result_addi_si32n1252582164_si32n2147483648 = arc.addi %cst_si32n1252582164, %cst_si32n2147483648 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1252582164_si32n2147483648:%[^ ]+]] = arc.addi [[CSTsi32n1252582164]], [[CSTsi32n2147483648]] : si32
    "arc.keep"(%result_addi_si32n1252582164_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1252582164_si32n2147483648]]) : (si32) -> ()

    // addi -1252582164, -2147483647 -> no-fold
    %result_addi_si32n1252582164_si32n2147483647 = arc.addi %cst_si32n1252582164, %cst_si32n2147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1252582164_si32n2147483647:%[^ ]+]] = arc.addi [[CSTsi32n1252582164]], [[CSTsi32n2147483647]] : si32
    "arc.keep"(%result_addi_si32n1252582164_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1252582164_si32n2147483647]]) : (si32) -> ()

    // addi -1252582164, -1713183800 -> no-fold
    %result_addi_si32n1252582164_si32n1713183800 = arc.addi %cst_si32n1252582164, %cst_si32n1713183800 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1252582164_si32n1713183800:%[^ ]+]] = arc.addi [[CSTsi32n1252582164]], [[CSTsi32n1713183800]] : si32
    "arc.keep"(%result_addi_si32n1252582164_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1252582164_si32n1713183800]]) : (si32) -> ()

    // addi -1252582164, -1252582164 -> no-fold
    %result_addi_si32n1252582164_si32n1252582164 = arc.addi %cst_si32n1252582164, %cst_si32n1252582164 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1252582164_si32n1252582164:%[^ ]+]] = arc.addi [[CSTsi32n1252582164]], [[CSTsi32n1252582164]] : si32
    "arc.keep"(%result_addi_si32n1252582164_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1252582164_si32n1252582164]]) : (si32) -> ()

    // addi -1252582164, -1035405763 -> no-fold
    %result_addi_si32n1252582164_si32n1035405763 = arc.addi %cst_si32n1252582164, %cst_si32n1035405763 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1252582164_si32n1035405763:%[^ ]+]] = arc.addi [[CSTsi32n1252582164]], [[CSTsi32n1035405763]] : si32
    "arc.keep"(%result_addi_si32n1252582164_si32n1035405763) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1252582164_si32n1035405763]]) : (si32) -> ()

    // addi -1035405763, -2147483648 -> no-fold
    %result_addi_si32n1035405763_si32n2147483648 = arc.addi %cst_si32n1035405763, %cst_si32n2147483648 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1035405763_si32n2147483648:%[^ ]+]] = arc.addi [[CSTsi32n1035405763]], [[CSTsi32n2147483648]] : si32
    "arc.keep"(%result_addi_si32n1035405763_si32n2147483648) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1035405763_si32n2147483648]]) : (si32) -> ()

    // addi -1035405763, -2147483647 -> no-fold
    %result_addi_si32n1035405763_si32n2147483647 = arc.addi %cst_si32n1035405763, %cst_si32n2147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1035405763_si32n2147483647:%[^ ]+]] = arc.addi [[CSTsi32n1035405763]], [[CSTsi32n2147483647]] : si32
    "arc.keep"(%result_addi_si32n1035405763_si32n2147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1035405763_si32n2147483647]]) : (si32) -> ()

    // addi -1035405763, -1713183800 -> no-fold
    %result_addi_si32n1035405763_si32n1713183800 = arc.addi %cst_si32n1035405763, %cst_si32n1713183800 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1035405763_si32n1713183800:%[^ ]+]] = arc.addi [[CSTsi32n1035405763]], [[CSTsi32n1713183800]] : si32
    "arc.keep"(%result_addi_si32n1035405763_si32n1713183800) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1035405763_si32n1713183800]]) : (si32) -> ()

    // addi -1035405763, -1252582164 -> no-fold
    %result_addi_si32n1035405763_si32n1252582164 = arc.addi %cst_si32n1035405763, %cst_si32n1252582164 : si32
    // CHECK-DAG: [[RESULT_addi_si32n1035405763_si32n1252582164:%[^ ]+]] = arc.addi [[CSTsi32n1035405763]], [[CSTsi32n1252582164]] : si32
    "arc.keep"(%result_addi_si32n1035405763_si32n1252582164) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si32n1035405763_si32n1252582164]]) : (si32) -> ()

    // addi 2147483646, 2147483646 -> no-fold
    %result_addi_si322147483646_si322147483646 = arc.addi %cst_si322147483646, %cst_si322147483646 : si32
    // CHECK-DAG: [[RESULT_addi_si322147483646_si322147483646:%[^ ]+]] = arc.addi [[CSTsi322147483646]], [[CSTsi322147483646]] : si32
    "arc.keep"(%result_addi_si322147483646_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si322147483646_si322147483646]]) : (si32) -> ()

    // addi 2147483646, 2147483647 -> no-fold
    %result_addi_si322147483646_si322147483647 = arc.addi %cst_si322147483646, %cst_si322147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si322147483646_si322147483647:%[^ ]+]] = arc.addi [[CSTsi322147483646]], [[CSTsi322147483647]] : si32
    "arc.keep"(%result_addi_si322147483646_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si322147483646_si322147483647]]) : (si32) -> ()

    // addi 2147483647, 2147483646 -> no-fold
    %result_addi_si322147483647_si322147483646 = arc.addi %cst_si322147483647, %cst_si322147483646 : si32
    // CHECK-DAG: [[RESULT_addi_si322147483647_si322147483646:%[^ ]+]] = arc.addi [[CSTsi322147483647]], [[CSTsi322147483646]] : si32
    "arc.keep"(%result_addi_si322147483647_si322147483646) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si322147483647_si322147483646]]) : (si32) -> ()

    // addi 2147483647, 2147483647 -> no-fold
    %result_addi_si322147483647_si322147483647 = arc.addi %cst_si322147483647, %cst_si322147483647 : si32
    // CHECK-DAG: [[RESULT_addi_si322147483647_si322147483647:%[^ ]+]] = arc.addi [[CSTsi322147483647]], [[CSTsi322147483647]] : si32
    "arc.keep"(%result_addi_si322147483647_si322147483647) : (si32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si322147483647_si322147483647]]) : (si32) -> ()

    // addi -9223372036854775808, 0 -> -9223372036854775808
    %result_addi_si64n9223372036854775808_si640 = arc.addi %cst_si64n9223372036854775808, %cst_si640 : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n9223372036854775808]]) : (si64) -> ()

    // addi 0, -9223372036854775808 -> -9223372036854775808
    %result_addi_si640_si64n9223372036854775808 = arc.addi %cst_si640, %cst_si64n9223372036854775808 : si64
    "arc.keep"(%result_addi_si640_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n9223372036854775808]]) : (si64) -> ()

    // addi -9223372036854775807, 0 -> -9223372036854775807
    %result_addi_si64n9223372036854775807_si640 = arc.addi %cst_si64n9223372036854775807, %cst_si640 : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n9223372036854775807]]) : (si64) -> ()

    // addi 0, -9223372036854775807 -> -9223372036854775807
    %result_addi_si640_si64n9223372036854775807 = arc.addi %cst_si640, %cst_si64n9223372036854775807 : si64
    "arc.keep"(%result_addi_si640_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n9223372036854775807]]) : (si64) -> ()

    // addi -9223372036854775808, 5577148965131116544 -> -3646223071723659264
    %result_addi_si64n9223372036854775808_si645577148965131116544 = arc.addi %cst_si64n9223372036854775808, %cst_si645577148965131116544 : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3646223071723659264]]) : (si64) -> ()

    // addi 5577148965131116544, -9223372036854775808 -> -3646223071723659264
    %result_addi_si645577148965131116544_si64n9223372036854775808 = arc.addi %cst_si645577148965131116544, %cst_si64n9223372036854775808 : si64
    "arc.keep"(%result_addi_si645577148965131116544_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3646223071723659264]]) : (si64) -> ()

    // addi -9223372036854775807, 5577148965131116544 -> -3646223071723659263
    %result_addi_si64n9223372036854775807_si645577148965131116544 = arc.addi %cst_si64n9223372036854775807, %cst_si645577148965131116544 : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3646223071723659263]]) : (si64) -> ()

    // addi 5577148965131116544, -9223372036854775807 -> -3646223071723659263
    %result_addi_si645577148965131116544_si64n9223372036854775807 = arc.addi %cst_si645577148965131116544, %cst_si64n9223372036854775807 : si64
    "arc.keep"(%result_addi_si645577148965131116544_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3646223071723659263]]) : (si64) -> ()

    // addi -1741927215160008704, -1741927215160008704 -> -3483854430320017408
    %result_addi_si64n1741927215160008704_si64n1741927215160008704 = arc.addi %cst_si64n1741927215160008704, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3483854430320017408]]) : (si64) -> ()

    // addi -1741927215160008704, -1328271339354574848 -> -3070198554514583552
    %result_addi_si64n1741927215160008704_si64n1328271339354574848 = arc.addi %cst_si64n1741927215160008704, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3070198554514583552]]) : (si64) -> ()

    // addi -1328271339354574848, -1741927215160008704 -> -3070198554514583552
    %result_addi_si64n1328271339354574848_si64n1741927215160008704 = arc.addi %cst_si64n1328271339354574848, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n3070198554514583552]]) : (si64) -> ()

    // addi -1328271339354574848, -1328271339354574848 -> -2656542678709149696
    %result_addi_si64n1328271339354574848_si64n1328271339354574848 = arc.addi %cst_si64n1328271339354574848, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n2656542678709149696]]) : (si64) -> ()

    // addi -1741927215160008704, 0 -> -1741927215160008704
    %result_addi_si64n1741927215160008704_si640 = arc.addi %cst_si64n1741927215160008704, %cst_si640 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1741927215160008704]]) : (si64) -> ()

    // addi 0, -1741927215160008704 -> -1741927215160008704
    %result_addi_si640_si64n1741927215160008704 = arc.addi %cst_si640, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si640_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1741927215160008704]]) : (si64) -> ()

    // addi -1328271339354574848, 0 -> -1328271339354574848
    %result_addi_si64n1328271339354574848_si640 = arc.addi %cst_si64n1328271339354574848, %cst_si640 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1328271339354574848]]) : (si64) -> ()

    // addi 0, -1328271339354574848 -> -1328271339354574848
    %result_addi_si640_si64n1328271339354574848 = arc.addi %cst_si640, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si640_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1328271339354574848]]) : (si64) -> ()

    // addi -9223372036854775808, 9223372036854775806 -> -2
    %result_addi_si64n9223372036854775808_si649223372036854775806 = arc.addi %cst_si64n9223372036854775808, %cst_si649223372036854775806 : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n2]]) : (si64) -> ()

    // addi 9223372036854775806, -9223372036854775808 -> -2
    %result_addi_si649223372036854775806_si64n9223372036854775808 = arc.addi %cst_si649223372036854775806, %cst_si64n9223372036854775808 : si64
    "arc.keep"(%result_addi_si649223372036854775806_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n2]]) : (si64) -> ()

    // addi -9223372036854775808, 9223372036854775807 -> -1
    %result_addi_si64n9223372036854775808_si649223372036854775807 = arc.addi %cst_si64n9223372036854775808, %cst_si649223372036854775807 : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1]]) : (si64) -> ()

    // addi -9223372036854775807, 9223372036854775806 -> -1
    %result_addi_si64n9223372036854775807_si649223372036854775806 = arc.addi %cst_si64n9223372036854775807, %cst_si649223372036854775806 : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1]]) : (si64) -> ()

    // addi 9223372036854775806, -9223372036854775807 -> -1
    %result_addi_si649223372036854775806_si64n9223372036854775807 = arc.addi %cst_si649223372036854775806, %cst_si64n9223372036854775807 : si64
    "arc.keep"(%result_addi_si649223372036854775806_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1]]) : (si64) -> ()

    // addi 9223372036854775807, -9223372036854775808 -> -1
    %result_addi_si649223372036854775807_si64n9223372036854775808 = arc.addi %cst_si649223372036854775807, %cst_si64n9223372036854775808 : si64
    "arc.keep"(%result_addi_si649223372036854775807_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi64n1]]) : (si64) -> ()

    // addi -9223372036854775807, 9223372036854775807 -> 0
    %result_addi_si64n9223372036854775807_si649223372036854775807 = arc.addi %cst_si64n9223372036854775807, %cst_si649223372036854775807 : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi640]]) : (si64) -> ()

    // addi 0, 0 -> 0
    %result_addi_si640_si640 = arc.addi %cst_si640, %cst_si640 : si64
    "arc.keep"(%result_addi_si640_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi640]]) : (si64) -> ()

    // addi 9223372036854775807, -9223372036854775807 -> 0
    %result_addi_si649223372036854775807_si64n9223372036854775807 = arc.addi %cst_si649223372036854775807, %cst_si64n9223372036854775807 : si64
    "arc.keep"(%result_addi_si649223372036854775807_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi640]]) : (si64) -> ()

    // addi -1741927215160008704, 5577148965131116544 -> 3835221749971107840
    %result_addi_si64n1741927215160008704_si645577148965131116544 = arc.addi %cst_si64n1741927215160008704, %cst_si645577148965131116544 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi643835221749971107840]]) : (si64) -> ()

    // addi 5577148965131116544, -1741927215160008704 -> 3835221749971107840
    %result_addi_si645577148965131116544_si64n1741927215160008704 = arc.addi %cst_si645577148965131116544, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si645577148965131116544_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi643835221749971107840]]) : (si64) -> ()

    // addi -1328271339354574848, 5577148965131116544 -> 4248877625776541696
    %result_addi_si64n1328271339354574848_si645577148965131116544 = arc.addi %cst_si64n1328271339354574848, %cst_si645577148965131116544 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi644248877625776541696]]) : (si64) -> ()

    // addi 5577148965131116544, -1328271339354574848 -> 4248877625776541696
    %result_addi_si645577148965131116544_si64n1328271339354574848 = arc.addi %cst_si645577148965131116544, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si645577148965131116544_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi644248877625776541696]]) : (si64) -> ()

    // addi 0, 5577148965131116544 -> 5577148965131116544
    %result_addi_si640_si645577148965131116544 = arc.addi %cst_si640, %cst_si645577148965131116544 : si64
    "arc.keep"(%result_addi_si640_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi645577148965131116544]]) : (si64) -> ()

    // addi 5577148965131116544, 0 -> 5577148965131116544
    %result_addi_si645577148965131116544_si640 = arc.addi %cst_si645577148965131116544, %cst_si640 : si64
    "arc.keep"(%result_addi_si645577148965131116544_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi645577148965131116544]]) : (si64) -> ()

    // addi -1741927215160008704, 9223372036854775806 -> 7481444821694767102
    %result_addi_si64n1741927215160008704_si649223372036854775806 = arc.addi %cst_si64n1741927215160008704, %cst_si649223372036854775806 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647481444821694767102]]) : (si64) -> ()

    // addi 9223372036854775806, -1741927215160008704 -> 7481444821694767102
    %result_addi_si649223372036854775806_si64n1741927215160008704 = arc.addi %cst_si649223372036854775806, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si649223372036854775806_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647481444821694767102]]) : (si64) -> ()

    // addi -1741927215160008704, 9223372036854775807 -> 7481444821694767103
    %result_addi_si64n1741927215160008704_si649223372036854775807 = arc.addi %cst_si64n1741927215160008704, %cst_si649223372036854775807 : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647481444821694767103]]) : (si64) -> ()

    // addi 9223372036854775807, -1741927215160008704 -> 7481444821694767103
    %result_addi_si649223372036854775807_si64n1741927215160008704 = arc.addi %cst_si649223372036854775807, %cst_si64n1741927215160008704 : si64
    "arc.keep"(%result_addi_si649223372036854775807_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647481444821694767103]]) : (si64) -> ()

    // addi -1328271339354574848, 9223372036854775806 -> 7895100697500200958
    %result_addi_si64n1328271339354574848_si649223372036854775806 = arc.addi %cst_si64n1328271339354574848, %cst_si649223372036854775806 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647895100697500200958]]) : (si64) -> ()

    // addi 9223372036854775806, -1328271339354574848 -> 7895100697500200958
    %result_addi_si649223372036854775806_si64n1328271339354574848 = arc.addi %cst_si649223372036854775806, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si649223372036854775806_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647895100697500200958]]) : (si64) -> ()

    // addi -1328271339354574848, 9223372036854775807 -> 7895100697500200959
    %result_addi_si64n1328271339354574848_si649223372036854775807 = arc.addi %cst_si64n1328271339354574848, %cst_si649223372036854775807 : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647895100697500200959]]) : (si64) -> ()

    // addi 9223372036854775807, -1328271339354574848 -> 7895100697500200959
    %result_addi_si649223372036854775807_si64n1328271339354574848 = arc.addi %cst_si649223372036854775807, %cst_si64n1328271339354574848 : si64
    "arc.keep"(%result_addi_si649223372036854775807_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi647895100697500200959]]) : (si64) -> ()

    // addi 0, 9223372036854775806 -> 9223372036854775806
    %result_addi_si640_si649223372036854775806 = arc.addi %cst_si640, %cst_si649223372036854775806 : si64
    "arc.keep"(%result_addi_si640_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi649223372036854775806]]) : (si64) -> ()

    // addi 9223372036854775806, 0 -> 9223372036854775806
    %result_addi_si649223372036854775806_si640 = arc.addi %cst_si649223372036854775806, %cst_si640 : si64
    "arc.keep"(%result_addi_si649223372036854775806_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi649223372036854775806]]) : (si64) -> ()

    // addi 0, 9223372036854775807 -> 9223372036854775807
    %result_addi_si640_si649223372036854775807 = arc.addi %cst_si640, %cst_si649223372036854775807 : si64
    "arc.keep"(%result_addi_si640_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi649223372036854775807]]) : (si64) -> ()

    // addi 9223372036854775807, 0 -> 9223372036854775807
    %result_addi_si649223372036854775807_si640 = arc.addi %cst_si649223372036854775807, %cst_si640 : si64
    "arc.keep"(%result_addi_si649223372036854775807_si640) : (si64) -> ()
    // CHECK: "arc.keep"([[CSTsi649223372036854775807]]) : (si64) -> ()

    // addi -9223372036854775808, -9223372036854775808 -> no-fold
    %result_addi_si64n9223372036854775808_si64n9223372036854775808 = arc.addi %cst_si64n9223372036854775808, %cst_si64n9223372036854775808 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775808_si64n9223372036854775808:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775808]], [[CSTsi64n9223372036854775808]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775808_si64n9223372036854775808]]) : (si64) -> ()

    // addi -9223372036854775808, -9223372036854775807 -> no-fold
    %result_addi_si64n9223372036854775808_si64n9223372036854775807 = arc.addi %cst_si64n9223372036854775808, %cst_si64n9223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775808_si64n9223372036854775807:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775808]], [[CSTsi64n9223372036854775807]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775808_si64n9223372036854775807]]) : (si64) -> ()

    // addi -9223372036854775808, -1741927215160008704 -> no-fold
    %result_addi_si64n9223372036854775808_si64n1741927215160008704 = arc.addi %cst_si64n9223372036854775808, %cst_si64n1741927215160008704 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775808_si64n1741927215160008704:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775808]], [[CSTsi64n1741927215160008704]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775808_si64n1741927215160008704]]) : (si64) -> ()

    // addi -9223372036854775808, -1328271339354574848 -> no-fold
    %result_addi_si64n9223372036854775808_si64n1328271339354574848 = arc.addi %cst_si64n9223372036854775808, %cst_si64n1328271339354574848 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775808_si64n1328271339354574848:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775808]], [[CSTsi64n1328271339354574848]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775808_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775808_si64n1328271339354574848]]) : (si64) -> ()

    // addi -9223372036854775807, -9223372036854775808 -> no-fold
    %result_addi_si64n9223372036854775807_si64n9223372036854775808 = arc.addi %cst_si64n9223372036854775807, %cst_si64n9223372036854775808 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775807_si64n9223372036854775808:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775807]], [[CSTsi64n9223372036854775808]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775807_si64n9223372036854775808]]) : (si64) -> ()

    // addi -9223372036854775807, -9223372036854775807 -> no-fold
    %result_addi_si64n9223372036854775807_si64n9223372036854775807 = arc.addi %cst_si64n9223372036854775807, %cst_si64n9223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775807_si64n9223372036854775807:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775807]], [[CSTsi64n9223372036854775807]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775807_si64n9223372036854775807]]) : (si64) -> ()

    // addi -9223372036854775807, -1741927215160008704 -> no-fold
    %result_addi_si64n9223372036854775807_si64n1741927215160008704 = arc.addi %cst_si64n9223372036854775807, %cst_si64n1741927215160008704 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775807_si64n1741927215160008704:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775807]], [[CSTsi64n1741927215160008704]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si64n1741927215160008704) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775807_si64n1741927215160008704]]) : (si64) -> ()

    // addi -9223372036854775807, -1328271339354574848 -> no-fold
    %result_addi_si64n9223372036854775807_si64n1328271339354574848 = arc.addi %cst_si64n9223372036854775807, %cst_si64n1328271339354574848 : si64
    // CHECK-DAG: [[RESULT_addi_si64n9223372036854775807_si64n1328271339354574848:%[^ ]+]] = arc.addi [[CSTsi64n9223372036854775807]], [[CSTsi64n1328271339354574848]] : si64
    "arc.keep"(%result_addi_si64n9223372036854775807_si64n1328271339354574848) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n9223372036854775807_si64n1328271339354574848]]) : (si64) -> ()

    // addi -1741927215160008704, -9223372036854775808 -> no-fold
    %result_addi_si64n1741927215160008704_si64n9223372036854775808 = arc.addi %cst_si64n1741927215160008704, %cst_si64n9223372036854775808 : si64
    // CHECK-DAG: [[RESULT_addi_si64n1741927215160008704_si64n9223372036854775808:%[^ ]+]] = arc.addi [[CSTsi64n1741927215160008704]], [[CSTsi64n9223372036854775808]] : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n1741927215160008704_si64n9223372036854775808]]) : (si64) -> ()

    // addi -1741927215160008704, -9223372036854775807 -> no-fold
    %result_addi_si64n1741927215160008704_si64n9223372036854775807 = arc.addi %cst_si64n1741927215160008704, %cst_si64n9223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si64n1741927215160008704_si64n9223372036854775807:%[^ ]+]] = arc.addi [[CSTsi64n1741927215160008704]], [[CSTsi64n9223372036854775807]] : si64
    "arc.keep"(%result_addi_si64n1741927215160008704_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n1741927215160008704_si64n9223372036854775807]]) : (si64) -> ()

    // addi -1328271339354574848, -9223372036854775808 -> no-fold
    %result_addi_si64n1328271339354574848_si64n9223372036854775808 = arc.addi %cst_si64n1328271339354574848, %cst_si64n9223372036854775808 : si64
    // CHECK-DAG: [[RESULT_addi_si64n1328271339354574848_si64n9223372036854775808:%[^ ]+]] = arc.addi [[CSTsi64n1328271339354574848]], [[CSTsi64n9223372036854775808]] : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si64n9223372036854775808) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n1328271339354574848_si64n9223372036854775808]]) : (si64) -> ()

    // addi -1328271339354574848, -9223372036854775807 -> no-fold
    %result_addi_si64n1328271339354574848_si64n9223372036854775807 = arc.addi %cst_si64n1328271339354574848, %cst_si64n9223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si64n1328271339354574848_si64n9223372036854775807:%[^ ]+]] = arc.addi [[CSTsi64n1328271339354574848]], [[CSTsi64n9223372036854775807]] : si64
    "arc.keep"(%result_addi_si64n1328271339354574848_si64n9223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si64n1328271339354574848_si64n9223372036854775807]]) : (si64) -> ()

    // addi 5577148965131116544, 5577148965131116544 -> no-fold
    %result_addi_si645577148965131116544_si645577148965131116544 = arc.addi %cst_si645577148965131116544, %cst_si645577148965131116544 : si64
    // CHECK-DAG: [[RESULT_addi_si645577148965131116544_si645577148965131116544:%[^ ]+]] = arc.addi [[CSTsi645577148965131116544]], [[CSTsi645577148965131116544]] : si64
    "arc.keep"(%result_addi_si645577148965131116544_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si645577148965131116544_si645577148965131116544]]) : (si64) -> ()

    // addi 5577148965131116544, 9223372036854775806 -> no-fold
    %result_addi_si645577148965131116544_si649223372036854775806 = arc.addi %cst_si645577148965131116544, %cst_si649223372036854775806 : si64
    // CHECK-DAG: [[RESULT_addi_si645577148965131116544_si649223372036854775806:%[^ ]+]] = arc.addi [[CSTsi645577148965131116544]], [[CSTsi649223372036854775806]] : si64
    "arc.keep"(%result_addi_si645577148965131116544_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si645577148965131116544_si649223372036854775806]]) : (si64) -> ()

    // addi 5577148965131116544, 9223372036854775807 -> no-fold
    %result_addi_si645577148965131116544_si649223372036854775807 = arc.addi %cst_si645577148965131116544, %cst_si649223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si645577148965131116544_si649223372036854775807:%[^ ]+]] = arc.addi [[CSTsi645577148965131116544]], [[CSTsi649223372036854775807]] : si64
    "arc.keep"(%result_addi_si645577148965131116544_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si645577148965131116544_si649223372036854775807]]) : (si64) -> ()

    // addi 9223372036854775806, 5577148965131116544 -> no-fold
    %result_addi_si649223372036854775806_si645577148965131116544 = arc.addi %cst_si649223372036854775806, %cst_si645577148965131116544 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775806_si645577148965131116544:%[^ ]+]] = arc.addi [[CSTsi649223372036854775806]], [[CSTsi645577148965131116544]] : si64
    "arc.keep"(%result_addi_si649223372036854775806_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775806_si645577148965131116544]]) : (si64) -> ()

    // addi 9223372036854775806, 9223372036854775806 -> no-fold
    %result_addi_si649223372036854775806_si649223372036854775806 = arc.addi %cst_si649223372036854775806, %cst_si649223372036854775806 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775806_si649223372036854775806:%[^ ]+]] = arc.addi [[CSTsi649223372036854775806]], [[CSTsi649223372036854775806]] : si64
    "arc.keep"(%result_addi_si649223372036854775806_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775806_si649223372036854775806]]) : (si64) -> ()

    // addi 9223372036854775806, 9223372036854775807 -> no-fold
    %result_addi_si649223372036854775806_si649223372036854775807 = arc.addi %cst_si649223372036854775806, %cst_si649223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775806_si649223372036854775807:%[^ ]+]] = arc.addi [[CSTsi649223372036854775806]], [[CSTsi649223372036854775807]] : si64
    "arc.keep"(%result_addi_si649223372036854775806_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775806_si649223372036854775807]]) : (si64) -> ()

    // addi 9223372036854775807, 5577148965131116544 -> no-fold
    %result_addi_si649223372036854775807_si645577148965131116544 = arc.addi %cst_si649223372036854775807, %cst_si645577148965131116544 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775807_si645577148965131116544:%[^ ]+]] = arc.addi [[CSTsi649223372036854775807]], [[CSTsi645577148965131116544]] : si64
    "arc.keep"(%result_addi_si649223372036854775807_si645577148965131116544) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775807_si645577148965131116544]]) : (si64) -> ()

    // addi 9223372036854775807, 9223372036854775806 -> no-fold
    %result_addi_si649223372036854775807_si649223372036854775806 = arc.addi %cst_si649223372036854775807, %cst_si649223372036854775806 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775807_si649223372036854775806:%[^ ]+]] = arc.addi [[CSTsi649223372036854775807]], [[CSTsi649223372036854775806]] : si64
    "arc.keep"(%result_addi_si649223372036854775807_si649223372036854775806) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775807_si649223372036854775806]]) : (si64) -> ()

    // addi 9223372036854775807, 9223372036854775807 -> no-fold
    %result_addi_si649223372036854775807_si649223372036854775807 = arc.addi %cst_si649223372036854775807, %cst_si649223372036854775807 : si64
    // CHECK-DAG: [[RESULT_addi_si649223372036854775807_si649223372036854775807:%[^ ]+]] = arc.addi [[CSTsi649223372036854775807]], [[CSTsi649223372036854775807]] : si64
    "arc.keep"(%result_addi_si649223372036854775807_si649223372036854775807) : (si64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si649223372036854775807_si649223372036854775807]]) : (si64) -> ()

    // addi -128, 0 -> -128
    %result_addi_si8n128_si80 = arc.addi %cst_si8n128, %cst_si80 : si8
    "arc.keep"(%result_addi_si8n128_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n128]]) : (si8) -> ()

    // addi 0, -128 -> -128
    %result_addi_si80_si8n128 = arc.addi %cst_si80, %cst_si8n128 : si8
    "arc.keep"(%result_addi_si80_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n128]]) : (si8) -> ()

    // addi -128, 1 -> -127
    %result_addi_si8n128_si81 = arc.addi %cst_si8n128, %cst_si81 : si8
    "arc.keep"(%result_addi_si8n128_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n127]]) : (si8) -> ()

    // addi -127, 0 -> -127
    %result_addi_si8n127_si80 = arc.addi %cst_si8n127, %cst_si80 : si8
    "arc.keep"(%result_addi_si8n127_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n127]]) : (si8) -> ()

    // addi 0, -127 -> -127
    %result_addi_si80_si8n127 = arc.addi %cst_si80, %cst_si8n127 : si8
    "arc.keep"(%result_addi_si80_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n127]]) : (si8) -> ()

    // addi 1, -128 -> -127
    %result_addi_si81_si8n128 = arc.addi %cst_si81, %cst_si8n128 : si8
    "arc.keep"(%result_addi_si81_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n127]]) : (si8) -> ()

    // addi -127, 1 -> -126
    %result_addi_si8n127_si81 = arc.addi %cst_si8n127, %cst_si81 : si8
    "arc.keep"(%result_addi_si8n127_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n126]]) : (si8) -> ()

    // addi 1, -127 -> -126
    %result_addi_si81_si8n127 = arc.addi %cst_si81, %cst_si8n127 : si8
    "arc.keep"(%result_addi_si81_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n126]]) : (si8) -> ()

    // addi -128, 16 -> -112
    %result_addi_si8n128_si816 = arc.addi %cst_si8n128, %cst_si816 : si8
    "arc.keep"(%result_addi_si8n128_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n112]]) : (si8) -> ()

    // addi 16, -128 -> -112
    %result_addi_si816_si8n128 = arc.addi %cst_si816, %cst_si8n128 : si8
    "arc.keep"(%result_addi_si816_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n112]]) : (si8) -> ()

    // addi -127, 16 -> -111
    %result_addi_si8n127_si816 = arc.addi %cst_si8n127, %cst_si816 : si8
    "arc.keep"(%result_addi_si8n127_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n111]]) : (si8) -> ()

    // addi 16, -127 -> -111
    %result_addi_si816_si8n127 = arc.addi %cst_si816, %cst_si8n127 : si8
    "arc.keep"(%result_addi_si816_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n111]]) : (si8) -> ()

    // addi -128, 126 -> -2
    %result_addi_si8n128_si8126 = arc.addi %cst_si8n128, %cst_si8126 : si8
    "arc.keep"(%result_addi_si8n128_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n2]]) : (si8) -> ()

    // addi 126, -128 -> -2
    %result_addi_si8126_si8n128 = arc.addi %cst_si8126, %cst_si8n128 : si8
    "arc.keep"(%result_addi_si8126_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n2]]) : (si8) -> ()

    // addi -128, 127 -> -1
    %result_addi_si8n128_si8127 = arc.addi %cst_si8n128, %cst_si8127 : si8
    "arc.keep"(%result_addi_si8n128_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n1]]) : (si8) -> ()

    // addi -127, 126 -> -1
    %result_addi_si8n127_si8126 = arc.addi %cst_si8n127, %cst_si8126 : si8
    "arc.keep"(%result_addi_si8n127_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n1]]) : (si8) -> ()

    // addi 126, -127 -> -1
    %result_addi_si8126_si8n127 = arc.addi %cst_si8126, %cst_si8n127 : si8
    "arc.keep"(%result_addi_si8126_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n1]]) : (si8) -> ()

    // addi 127, -128 -> -1
    %result_addi_si8127_si8n128 = arc.addi %cst_si8127, %cst_si8n128 : si8
    "arc.keep"(%result_addi_si8127_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8n1]]) : (si8) -> ()

    // addi -127, 127 -> 0
    %result_addi_si8n127_si8127 = arc.addi %cst_si8n127, %cst_si8127 : si8
    "arc.keep"(%result_addi_si8n127_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi80]]) : (si8) -> ()

    // addi 0, 0 -> 0
    %result_addi_si80_si80 = arc.addi %cst_si80, %cst_si80 : si8
    "arc.keep"(%result_addi_si80_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi80]]) : (si8) -> ()

    // addi 127, -127 -> 0
    %result_addi_si8127_si8n127 = arc.addi %cst_si8127, %cst_si8n127 : si8
    "arc.keep"(%result_addi_si8127_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi80]]) : (si8) -> ()

    // addi 0, 1 -> 1
    %result_addi_si80_si81 = arc.addi %cst_si80, %cst_si81 : si8
    "arc.keep"(%result_addi_si80_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi81]]) : (si8) -> ()

    // addi 1, 0 -> 1
    %result_addi_si81_si80 = arc.addi %cst_si81, %cst_si80 : si8
    "arc.keep"(%result_addi_si81_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi81]]) : (si8) -> ()

    // addi 1, 1 -> 2
    %result_addi_si81_si81 = arc.addi %cst_si81, %cst_si81 : si8
    "arc.keep"(%result_addi_si81_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi82]]) : (si8) -> ()

    // addi 0, 16 -> 16
    %result_addi_si80_si816 = arc.addi %cst_si80, %cst_si816 : si8
    "arc.keep"(%result_addi_si80_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi816]]) : (si8) -> ()

    // addi 16, 0 -> 16
    %result_addi_si816_si80 = arc.addi %cst_si816, %cst_si80 : si8
    "arc.keep"(%result_addi_si816_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi816]]) : (si8) -> ()

    // addi 1, 16 -> 17
    %result_addi_si81_si816 = arc.addi %cst_si81, %cst_si816 : si8
    "arc.keep"(%result_addi_si81_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi817]]) : (si8) -> ()

    // addi 16, 1 -> 17
    %result_addi_si816_si81 = arc.addi %cst_si816, %cst_si81 : si8
    "arc.keep"(%result_addi_si816_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi817]]) : (si8) -> ()

    // addi 16, 16 -> 32
    %result_addi_si816_si816 = arc.addi %cst_si816, %cst_si816 : si8
    "arc.keep"(%result_addi_si816_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi832]]) : (si8) -> ()

    // addi 0, 126 -> 126
    %result_addi_si80_si8126 = arc.addi %cst_si80, %cst_si8126 : si8
    "arc.keep"(%result_addi_si80_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8126]]) : (si8) -> ()

    // addi 126, 0 -> 126
    %result_addi_si8126_si80 = arc.addi %cst_si8126, %cst_si80 : si8
    "arc.keep"(%result_addi_si8126_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8126]]) : (si8) -> ()

    // addi 0, 127 -> 127
    %result_addi_si80_si8127 = arc.addi %cst_si80, %cst_si8127 : si8
    "arc.keep"(%result_addi_si80_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8127]]) : (si8) -> ()

    // addi 1, 126 -> 127
    %result_addi_si81_si8126 = arc.addi %cst_si81, %cst_si8126 : si8
    "arc.keep"(%result_addi_si81_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8127]]) : (si8) -> ()

    // addi 126, 1 -> 127
    %result_addi_si8126_si81 = arc.addi %cst_si8126, %cst_si81 : si8
    "arc.keep"(%result_addi_si8126_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8127]]) : (si8) -> ()

    // addi 127, 0 -> 127
    %result_addi_si8127_si80 = arc.addi %cst_si8127, %cst_si80 : si8
    "arc.keep"(%result_addi_si8127_si80) : (si8) -> ()
    // CHECK: "arc.keep"([[CSTsi8127]]) : (si8) -> ()

    // addi -128, -128 -> no-fold
    %result_addi_si8n128_si8n128 = arc.addi %cst_si8n128, %cst_si8n128 : si8
    // CHECK-DAG: [[RESULT_addi_si8n128_si8n128:%[^ ]+]] = arc.addi [[CSTsi8n128]], [[CSTsi8n128]] : si8
    "arc.keep"(%result_addi_si8n128_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8n128_si8n128]]) : (si8) -> ()

    // addi -128, -127 -> no-fold
    %result_addi_si8n128_si8n127 = arc.addi %cst_si8n128, %cst_si8n127 : si8
    // CHECK-DAG: [[RESULT_addi_si8n128_si8n127:%[^ ]+]] = arc.addi [[CSTsi8n128]], [[CSTsi8n127]] : si8
    "arc.keep"(%result_addi_si8n128_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8n128_si8n127]]) : (si8) -> ()

    // addi -127, -128 -> no-fold
    %result_addi_si8n127_si8n128 = arc.addi %cst_si8n127, %cst_si8n128 : si8
    // CHECK-DAG: [[RESULT_addi_si8n127_si8n128:%[^ ]+]] = arc.addi [[CSTsi8n127]], [[CSTsi8n128]] : si8
    "arc.keep"(%result_addi_si8n127_si8n128) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8n127_si8n128]]) : (si8) -> ()

    // addi -127, -127 -> no-fold
    %result_addi_si8n127_si8n127 = arc.addi %cst_si8n127, %cst_si8n127 : si8
    // CHECK-DAG: [[RESULT_addi_si8n127_si8n127:%[^ ]+]] = arc.addi [[CSTsi8n127]], [[CSTsi8n127]] : si8
    "arc.keep"(%result_addi_si8n127_si8n127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8n127_si8n127]]) : (si8) -> ()

    // addi 1, 127 -> no-fold
    %result_addi_si81_si8127 = arc.addi %cst_si81, %cst_si8127 : si8
    // CHECK-DAG: [[RESULT_addi_si81_si8127:%[^ ]+]] = arc.addi [[CSTsi81]], [[CSTsi8127]] : si8
    "arc.keep"(%result_addi_si81_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si81_si8127]]) : (si8) -> ()

    // addi 16, 126 -> no-fold
    %result_addi_si816_si8126 = arc.addi %cst_si816, %cst_si8126 : si8
    // CHECK-DAG: [[RESULT_addi_si816_si8126:%[^ ]+]] = arc.addi [[CSTsi816]], [[CSTsi8126]] : si8
    "arc.keep"(%result_addi_si816_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si816_si8126]]) : (si8) -> ()

    // addi 16, 127 -> no-fold
    %result_addi_si816_si8127 = arc.addi %cst_si816, %cst_si8127 : si8
    // CHECK-DAG: [[RESULT_addi_si816_si8127:%[^ ]+]] = arc.addi [[CSTsi816]], [[CSTsi8127]] : si8
    "arc.keep"(%result_addi_si816_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si816_si8127]]) : (si8) -> ()

    // addi 126, 16 -> no-fold
    %result_addi_si8126_si816 = arc.addi %cst_si8126, %cst_si816 : si8
    // CHECK-DAG: [[RESULT_addi_si8126_si816:%[^ ]+]] = arc.addi [[CSTsi8126]], [[CSTsi816]] : si8
    "arc.keep"(%result_addi_si8126_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8126_si816]]) : (si8) -> ()

    // addi 126, 126 -> no-fold
    %result_addi_si8126_si8126 = arc.addi %cst_si8126, %cst_si8126 : si8
    // CHECK-DAG: [[RESULT_addi_si8126_si8126:%[^ ]+]] = arc.addi [[CSTsi8126]], [[CSTsi8126]] : si8
    "arc.keep"(%result_addi_si8126_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8126_si8126]]) : (si8) -> ()

    // addi 126, 127 -> no-fold
    %result_addi_si8126_si8127 = arc.addi %cst_si8126, %cst_si8127 : si8
    // CHECK-DAG: [[RESULT_addi_si8126_si8127:%[^ ]+]] = arc.addi [[CSTsi8126]], [[CSTsi8127]] : si8
    "arc.keep"(%result_addi_si8126_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8126_si8127]]) : (si8) -> ()

    // addi 127, 1 -> no-fold
    %result_addi_si8127_si81 = arc.addi %cst_si8127, %cst_si81 : si8
    // CHECK-DAG: [[RESULT_addi_si8127_si81:%[^ ]+]] = arc.addi [[CSTsi8127]], [[CSTsi81]] : si8
    "arc.keep"(%result_addi_si8127_si81) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8127_si81]]) : (si8) -> ()

    // addi 127, 16 -> no-fold
    %result_addi_si8127_si816 = arc.addi %cst_si8127, %cst_si816 : si8
    // CHECK-DAG: [[RESULT_addi_si8127_si816:%[^ ]+]] = arc.addi [[CSTsi8127]], [[CSTsi816]] : si8
    "arc.keep"(%result_addi_si8127_si816) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8127_si816]]) : (si8) -> ()

    // addi 127, 126 -> no-fold
    %result_addi_si8127_si8126 = arc.addi %cst_si8127, %cst_si8126 : si8
    // CHECK-DAG: [[RESULT_addi_si8127_si8126:%[^ ]+]] = arc.addi [[CSTsi8127]], [[CSTsi8126]] : si8
    "arc.keep"(%result_addi_si8127_si8126) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8127_si8126]]) : (si8) -> ()

    // addi 127, 127 -> no-fold
    %result_addi_si8127_si8127 = arc.addi %cst_si8127, %cst_si8127 : si8
    // CHECK-DAG: [[RESULT_addi_si8127_si8127:%[^ ]+]] = arc.addi [[CSTsi8127]], [[CSTsi8127]] : si8
    "arc.keep"(%result_addi_si8127_si8127) : (si8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_si8127_si8127]]) : (si8) -> ()

    // addi 0, 0 -> 0
    %result_addi_ui160_ui160 = arc.addi %cst_ui160, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui160_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui160]]) : (ui16) -> ()

    // addi 0, 1 -> 1
    %result_addi_ui160_ui161 = arc.addi %cst_ui160, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui160_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161]]) : (ui16) -> ()

    // addi 1, 0 -> 1
    %result_addi_ui161_ui160 = arc.addi %cst_ui161, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui161_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161]]) : (ui16) -> ()

    // addi 1, 1 -> 2
    %result_addi_ui161_ui161 = arc.addi %cst_ui161, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui161_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui162]]) : (ui16) -> ()

    // addi 0, 1717 -> 1717
    %result_addi_ui160_ui161717 = arc.addi %cst_ui160, %cst_ui161717 : ui16
    "arc.keep"(%result_addi_ui160_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161717]]) : (ui16) -> ()

    // addi 1717, 0 -> 1717
    %result_addi_ui161717_ui160 = arc.addi %cst_ui161717, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui161717_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161717]]) : (ui16) -> ()

    // addi 1, 1717 -> 1718
    %result_addi_ui161_ui161717 = arc.addi %cst_ui161, %cst_ui161717 : ui16
    "arc.keep"(%result_addi_ui161_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161718]]) : (ui16) -> ()

    // addi 1717, 1 -> 1718
    %result_addi_ui161717_ui161 = arc.addi %cst_ui161717, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui161717_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui161718]]) : (ui16) -> ()

    // addi 1717, 1717 -> 3434
    %result_addi_ui161717_ui161717 = arc.addi %cst_ui161717, %cst_ui161717 : ui16
    "arc.keep"(%result_addi_ui161717_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui163434]]) : (ui16) -> ()

    // addi 0, 17988 -> 17988
    %result_addi_ui160_ui1617988 = arc.addi %cst_ui160, %cst_ui1617988 : ui16
    "arc.keep"(%result_addi_ui160_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1617988]]) : (ui16) -> ()

    // addi 17988, 0 -> 17988
    %result_addi_ui1617988_ui160 = arc.addi %cst_ui1617988, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui1617988_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1617988]]) : (ui16) -> ()

    // addi 1, 17988 -> 17989
    %result_addi_ui161_ui1617988 = arc.addi %cst_ui161, %cst_ui1617988 : ui16
    "arc.keep"(%result_addi_ui161_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1617989]]) : (ui16) -> ()

    // addi 17988, 1 -> 17989
    %result_addi_ui1617988_ui161 = arc.addi %cst_ui1617988, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui1617988_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1617989]]) : (ui16) -> ()

    // addi 1717, 17988 -> 19705
    %result_addi_ui161717_ui1617988 = arc.addi %cst_ui161717, %cst_ui1617988 : ui16
    "arc.keep"(%result_addi_ui161717_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1619705]]) : (ui16) -> ()

    // addi 17988, 1717 -> 19705
    %result_addi_ui1617988_ui161717 = arc.addi %cst_ui1617988, %cst_ui161717 : ui16
    "arc.keep"(%result_addi_ui1617988_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1619705]]) : (ui16) -> ()

    // addi 17988, 17988 -> 35976
    %result_addi_ui1617988_ui1617988 = arc.addi %cst_ui1617988, %cst_ui1617988 : ui16
    "arc.keep"(%result_addi_ui1617988_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1635976]]) : (ui16) -> ()

    // addi 0, 65096 -> 65096
    %result_addi_ui160_ui1665096 = arc.addi %cst_ui160, %cst_ui1665096 : ui16
    "arc.keep"(%result_addi_ui160_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665096]]) : (ui16) -> ()

    // addi 65096, 0 -> 65096
    %result_addi_ui1665096_ui160 = arc.addi %cst_ui1665096, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui1665096_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665096]]) : (ui16) -> ()

    // addi 1, 65096 -> 65097
    %result_addi_ui161_ui1665096 = arc.addi %cst_ui161, %cst_ui1665096 : ui16
    "arc.keep"(%result_addi_ui161_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665097]]) : (ui16) -> ()

    // addi 65096, 1 -> 65097
    %result_addi_ui1665096_ui161 = arc.addi %cst_ui1665096, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui1665096_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665097]]) : (ui16) -> ()

    // addi 0, 65534 -> 65534
    %result_addi_ui160_ui1665534 = arc.addi %cst_ui160, %cst_ui1665534 : ui16
    "arc.keep"(%result_addi_ui160_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665534]]) : (ui16) -> ()

    // addi 65534, 0 -> 65534
    %result_addi_ui1665534_ui160 = arc.addi %cst_ui1665534, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui1665534_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665534]]) : (ui16) -> ()

    // addi 0, 65535 -> 65535
    %result_addi_ui160_ui1665535 = arc.addi %cst_ui160, %cst_ui1665535 : ui16
    "arc.keep"(%result_addi_ui160_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665535]]) : (ui16) -> ()

    // addi 1, 65534 -> 65535
    %result_addi_ui161_ui1665534 = arc.addi %cst_ui161, %cst_ui1665534 : ui16
    "arc.keep"(%result_addi_ui161_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665535]]) : (ui16) -> ()

    // addi 65534, 1 -> 65535
    %result_addi_ui1665534_ui161 = arc.addi %cst_ui1665534, %cst_ui161 : ui16
    "arc.keep"(%result_addi_ui1665534_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665535]]) : (ui16) -> ()

    // addi 65535, 0 -> 65535
    %result_addi_ui1665535_ui160 = arc.addi %cst_ui1665535, %cst_ui160 : ui16
    "arc.keep"(%result_addi_ui1665535_ui160) : (ui16) -> ()
    // CHECK: "arc.keep"([[CSTui1665535]]) : (ui16) -> ()

    // addi 1, 65535 -> no-fold
    %result_addi_ui161_ui1665535 = arc.addi %cst_ui161, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui161_ui1665535:%[^ ]+]] = arc.addi [[CSTui161]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui161_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui161_ui1665535]]) : (ui16) -> ()

    // addi 1717, 65096 -> no-fold
    %result_addi_ui161717_ui1665096 = arc.addi %cst_ui161717, %cst_ui1665096 : ui16
    // CHECK-DAG: [[RESULT_addi_ui161717_ui1665096:%[^ ]+]] = arc.addi [[CSTui161717]], [[CSTui1665096]] : ui16
    "arc.keep"(%result_addi_ui161717_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui161717_ui1665096]]) : (ui16) -> ()

    // addi 1717, 65534 -> no-fold
    %result_addi_ui161717_ui1665534 = arc.addi %cst_ui161717, %cst_ui1665534 : ui16
    // CHECK-DAG: [[RESULT_addi_ui161717_ui1665534:%[^ ]+]] = arc.addi [[CSTui161717]], [[CSTui1665534]] : ui16
    "arc.keep"(%result_addi_ui161717_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui161717_ui1665534]]) : (ui16) -> ()

    // addi 1717, 65535 -> no-fold
    %result_addi_ui161717_ui1665535 = arc.addi %cst_ui161717, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui161717_ui1665535:%[^ ]+]] = arc.addi [[CSTui161717]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui161717_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui161717_ui1665535]]) : (ui16) -> ()

    // addi 17988, 65096 -> no-fold
    %result_addi_ui1617988_ui1665096 = arc.addi %cst_ui1617988, %cst_ui1665096 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1617988_ui1665096:%[^ ]+]] = arc.addi [[CSTui1617988]], [[CSTui1665096]] : ui16
    "arc.keep"(%result_addi_ui1617988_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1617988_ui1665096]]) : (ui16) -> ()

    // addi 17988, 65534 -> no-fold
    %result_addi_ui1617988_ui1665534 = arc.addi %cst_ui1617988, %cst_ui1665534 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1617988_ui1665534:%[^ ]+]] = arc.addi [[CSTui1617988]], [[CSTui1665534]] : ui16
    "arc.keep"(%result_addi_ui1617988_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1617988_ui1665534]]) : (ui16) -> ()

    // addi 17988, 65535 -> no-fold
    %result_addi_ui1617988_ui1665535 = arc.addi %cst_ui1617988, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1617988_ui1665535:%[^ ]+]] = arc.addi [[CSTui1617988]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui1617988_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1617988_ui1665535]]) : (ui16) -> ()

    // addi 65096, 1717 -> no-fold
    %result_addi_ui1665096_ui161717 = arc.addi %cst_ui1665096, %cst_ui161717 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665096_ui161717:%[^ ]+]] = arc.addi [[CSTui1665096]], [[CSTui161717]] : ui16
    "arc.keep"(%result_addi_ui1665096_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665096_ui161717]]) : (ui16) -> ()

    // addi 65096, 17988 -> no-fold
    %result_addi_ui1665096_ui1617988 = arc.addi %cst_ui1665096, %cst_ui1617988 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665096_ui1617988:%[^ ]+]] = arc.addi [[CSTui1665096]], [[CSTui1617988]] : ui16
    "arc.keep"(%result_addi_ui1665096_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665096_ui1617988]]) : (ui16) -> ()

    // addi 65096, 65096 -> no-fold
    %result_addi_ui1665096_ui1665096 = arc.addi %cst_ui1665096, %cst_ui1665096 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665096_ui1665096:%[^ ]+]] = arc.addi [[CSTui1665096]], [[CSTui1665096]] : ui16
    "arc.keep"(%result_addi_ui1665096_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665096_ui1665096]]) : (ui16) -> ()

    // addi 65096, 65534 -> no-fold
    %result_addi_ui1665096_ui1665534 = arc.addi %cst_ui1665096, %cst_ui1665534 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665096_ui1665534:%[^ ]+]] = arc.addi [[CSTui1665096]], [[CSTui1665534]] : ui16
    "arc.keep"(%result_addi_ui1665096_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665096_ui1665534]]) : (ui16) -> ()

    // addi 65096, 65535 -> no-fold
    %result_addi_ui1665096_ui1665535 = arc.addi %cst_ui1665096, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665096_ui1665535:%[^ ]+]] = arc.addi [[CSTui1665096]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui1665096_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665096_ui1665535]]) : (ui16) -> ()

    // addi 65534, 1717 -> no-fold
    %result_addi_ui1665534_ui161717 = arc.addi %cst_ui1665534, %cst_ui161717 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665534_ui161717:%[^ ]+]] = arc.addi [[CSTui1665534]], [[CSTui161717]] : ui16
    "arc.keep"(%result_addi_ui1665534_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665534_ui161717]]) : (ui16) -> ()

    // addi 65534, 17988 -> no-fold
    %result_addi_ui1665534_ui1617988 = arc.addi %cst_ui1665534, %cst_ui1617988 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665534_ui1617988:%[^ ]+]] = arc.addi [[CSTui1665534]], [[CSTui1617988]] : ui16
    "arc.keep"(%result_addi_ui1665534_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665534_ui1617988]]) : (ui16) -> ()

    // addi 65534, 65096 -> no-fold
    %result_addi_ui1665534_ui1665096 = arc.addi %cst_ui1665534, %cst_ui1665096 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665534_ui1665096:%[^ ]+]] = arc.addi [[CSTui1665534]], [[CSTui1665096]] : ui16
    "arc.keep"(%result_addi_ui1665534_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665534_ui1665096]]) : (ui16) -> ()

    // addi 65534, 65534 -> no-fold
    %result_addi_ui1665534_ui1665534 = arc.addi %cst_ui1665534, %cst_ui1665534 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665534_ui1665534:%[^ ]+]] = arc.addi [[CSTui1665534]], [[CSTui1665534]] : ui16
    "arc.keep"(%result_addi_ui1665534_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665534_ui1665534]]) : (ui16) -> ()

    // addi 65534, 65535 -> no-fold
    %result_addi_ui1665534_ui1665535 = arc.addi %cst_ui1665534, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665534_ui1665535:%[^ ]+]] = arc.addi [[CSTui1665534]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui1665534_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665534_ui1665535]]) : (ui16) -> ()

    // addi 65535, 1 -> no-fold
    %result_addi_ui1665535_ui161 = arc.addi %cst_ui1665535, %cst_ui161 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui161:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui161]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui161) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui161]]) : (ui16) -> ()

    // addi 65535, 1717 -> no-fold
    %result_addi_ui1665535_ui161717 = arc.addi %cst_ui1665535, %cst_ui161717 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui161717:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui161717]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui161717) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui161717]]) : (ui16) -> ()

    // addi 65535, 17988 -> no-fold
    %result_addi_ui1665535_ui1617988 = arc.addi %cst_ui1665535, %cst_ui1617988 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui1617988:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui1617988]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui1617988) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui1617988]]) : (ui16) -> ()

    // addi 65535, 65096 -> no-fold
    %result_addi_ui1665535_ui1665096 = arc.addi %cst_ui1665535, %cst_ui1665096 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui1665096:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui1665096]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui1665096) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui1665096]]) : (ui16) -> ()

    // addi 65535, 65534 -> no-fold
    %result_addi_ui1665535_ui1665534 = arc.addi %cst_ui1665535, %cst_ui1665534 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui1665534:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui1665534]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui1665534) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui1665534]]) : (ui16) -> ()

    // addi 65535, 65535 -> no-fold
    %result_addi_ui1665535_ui1665535 = arc.addi %cst_ui1665535, %cst_ui1665535 : ui16
    // CHECK-DAG: [[RESULT_addi_ui1665535_ui1665535:%[^ ]+]] = arc.addi [[CSTui1665535]], [[CSTui1665535]] : ui16
    "arc.keep"(%result_addi_ui1665535_ui1665535) : (ui16) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui1665535_ui1665535]]) : (ui16) -> ()

    // addi 0, 0 -> 0
    %result_addi_ui320_ui320 = arc.addi %cst_ui320, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui320_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui320]]) : (ui32) -> ()

    // addi 0, 1 -> 1
    %result_addi_ui320_ui321 = arc.addi %cst_ui320, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui320_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui321]]) : (ui32) -> ()

    // addi 1, 0 -> 1
    %result_addi_ui321_ui320 = arc.addi %cst_ui321, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui321_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui321]]) : (ui32) -> ()

    // addi 1, 1 -> 2
    %result_addi_ui321_ui321 = arc.addi %cst_ui321, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui321_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui322]]) : (ui32) -> ()

    // addi 0, 2119154652 -> 2119154652
    %result_addi_ui320_ui322119154652 = arc.addi %cst_ui320, %cst_ui322119154652 : ui32
    "arc.keep"(%result_addi_ui320_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui322119154652]]) : (ui32) -> ()

    // addi 2119154652, 0 -> 2119154652
    %result_addi_ui322119154652_ui320 = arc.addi %cst_ui322119154652, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui322119154652_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui322119154652]]) : (ui32) -> ()

    // addi 1, 2119154652 -> 2119154653
    %result_addi_ui321_ui322119154652 = arc.addi %cst_ui321, %cst_ui322119154652 : ui32
    "arc.keep"(%result_addi_ui321_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui322119154653]]) : (ui32) -> ()

    // addi 2119154652, 1 -> 2119154653
    %result_addi_ui322119154652_ui321 = arc.addi %cst_ui322119154652, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui322119154652_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui322119154653]]) : (ui32) -> ()

    // addi 0, 3002788344 -> 3002788344
    %result_addi_ui320_ui323002788344 = arc.addi %cst_ui320, %cst_ui323002788344 : ui32
    "arc.keep"(%result_addi_ui320_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323002788344]]) : (ui32) -> ()

    // addi 3002788344, 0 -> 3002788344
    %result_addi_ui323002788344_ui320 = arc.addi %cst_ui323002788344, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui323002788344_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323002788344]]) : (ui32) -> ()

    // addi 1, 3002788344 -> 3002788345
    %result_addi_ui321_ui323002788344 = arc.addi %cst_ui321, %cst_ui323002788344 : ui32
    "arc.keep"(%result_addi_ui321_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323002788345]]) : (ui32) -> ()

    // addi 3002788344, 1 -> 3002788345
    %result_addi_ui323002788344_ui321 = arc.addi %cst_ui323002788344, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui323002788344_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323002788345]]) : (ui32) -> ()

    // addi 0, 3482297128 -> 3482297128
    %result_addi_ui320_ui323482297128 = arc.addi %cst_ui320, %cst_ui323482297128 : ui32
    "arc.keep"(%result_addi_ui320_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323482297128]]) : (ui32) -> ()

    // addi 3482297128, 0 -> 3482297128
    %result_addi_ui323482297128_ui320 = arc.addi %cst_ui323482297128, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui323482297128_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323482297128]]) : (ui32) -> ()

    // addi 1, 3482297128 -> 3482297129
    %result_addi_ui321_ui323482297128 = arc.addi %cst_ui321, %cst_ui323482297128 : ui32
    "arc.keep"(%result_addi_ui321_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323482297129]]) : (ui32) -> ()

    // addi 3482297128, 1 -> 3482297129
    %result_addi_ui323482297128_ui321 = arc.addi %cst_ui323482297128, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui323482297128_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui323482297129]]) : (ui32) -> ()

    // addi 2119154652, 2119154652 -> 4238309304
    %result_addi_ui322119154652_ui322119154652 = arc.addi %cst_ui322119154652, %cst_ui322119154652 : ui32
    "arc.keep"(%result_addi_ui322119154652_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324238309304]]) : (ui32) -> ()

    // addi 0, 4294967294 -> 4294967294
    %result_addi_ui320_ui324294967294 = arc.addi %cst_ui320, %cst_ui324294967294 : ui32
    "arc.keep"(%result_addi_ui320_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967294]]) : (ui32) -> ()

    // addi 4294967294, 0 -> 4294967294
    %result_addi_ui324294967294_ui320 = arc.addi %cst_ui324294967294, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui324294967294_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967294]]) : (ui32) -> ()

    // addi 0, 4294967295 -> 4294967295
    %result_addi_ui320_ui324294967295 = arc.addi %cst_ui320, %cst_ui324294967295 : ui32
    "arc.keep"(%result_addi_ui320_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967295]]) : (ui32) -> ()

    // addi 1, 4294967294 -> 4294967295
    %result_addi_ui321_ui324294967294 = arc.addi %cst_ui321, %cst_ui324294967294 : ui32
    "arc.keep"(%result_addi_ui321_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967295]]) : (ui32) -> ()

    // addi 4294967294, 1 -> 4294967295
    %result_addi_ui324294967294_ui321 = arc.addi %cst_ui324294967294, %cst_ui321 : ui32
    "arc.keep"(%result_addi_ui324294967294_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967295]]) : (ui32) -> ()

    // addi 4294967295, 0 -> 4294967295
    %result_addi_ui324294967295_ui320 = arc.addi %cst_ui324294967295, %cst_ui320 : ui32
    "arc.keep"(%result_addi_ui324294967295_ui320) : (ui32) -> ()
    // CHECK: "arc.keep"([[CSTui324294967295]]) : (ui32) -> ()

    // addi 1, 4294967295 -> no-fold
    %result_addi_ui321_ui324294967295 = arc.addi %cst_ui321, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui321_ui324294967295:%[^ ]+]] = arc.addi [[CSTui321]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui321_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui321_ui324294967295]]) : (ui32) -> ()

    // addi 2119154652, 3002788344 -> no-fold
    %result_addi_ui322119154652_ui323002788344 = arc.addi %cst_ui322119154652, %cst_ui323002788344 : ui32
    // CHECK-DAG: [[RESULT_addi_ui322119154652_ui323002788344:%[^ ]+]] = arc.addi [[CSTui322119154652]], [[CSTui323002788344]] : ui32
    "arc.keep"(%result_addi_ui322119154652_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui322119154652_ui323002788344]]) : (ui32) -> ()

    // addi 2119154652, 3482297128 -> no-fold
    %result_addi_ui322119154652_ui323482297128 = arc.addi %cst_ui322119154652, %cst_ui323482297128 : ui32
    // CHECK-DAG: [[RESULT_addi_ui322119154652_ui323482297128:%[^ ]+]] = arc.addi [[CSTui322119154652]], [[CSTui323482297128]] : ui32
    "arc.keep"(%result_addi_ui322119154652_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui322119154652_ui323482297128]]) : (ui32) -> ()

    // addi 2119154652, 4294967294 -> no-fold
    %result_addi_ui322119154652_ui324294967294 = arc.addi %cst_ui322119154652, %cst_ui324294967294 : ui32
    // CHECK-DAG: [[RESULT_addi_ui322119154652_ui324294967294:%[^ ]+]] = arc.addi [[CSTui322119154652]], [[CSTui324294967294]] : ui32
    "arc.keep"(%result_addi_ui322119154652_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui322119154652_ui324294967294]]) : (ui32) -> ()

    // addi 2119154652, 4294967295 -> no-fold
    %result_addi_ui322119154652_ui324294967295 = arc.addi %cst_ui322119154652, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui322119154652_ui324294967295:%[^ ]+]] = arc.addi [[CSTui322119154652]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui322119154652_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui322119154652_ui324294967295]]) : (ui32) -> ()

    // addi 3002788344, 2119154652 -> no-fold
    %result_addi_ui323002788344_ui322119154652 = arc.addi %cst_ui323002788344, %cst_ui322119154652 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323002788344_ui322119154652:%[^ ]+]] = arc.addi [[CSTui323002788344]], [[CSTui322119154652]] : ui32
    "arc.keep"(%result_addi_ui323002788344_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323002788344_ui322119154652]]) : (ui32) -> ()

    // addi 3002788344, 3002788344 -> no-fold
    %result_addi_ui323002788344_ui323002788344 = arc.addi %cst_ui323002788344, %cst_ui323002788344 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323002788344_ui323002788344:%[^ ]+]] = arc.addi [[CSTui323002788344]], [[CSTui323002788344]] : ui32
    "arc.keep"(%result_addi_ui323002788344_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323002788344_ui323002788344]]) : (ui32) -> ()

    // addi 3002788344, 3482297128 -> no-fold
    %result_addi_ui323002788344_ui323482297128 = arc.addi %cst_ui323002788344, %cst_ui323482297128 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323002788344_ui323482297128:%[^ ]+]] = arc.addi [[CSTui323002788344]], [[CSTui323482297128]] : ui32
    "arc.keep"(%result_addi_ui323002788344_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323002788344_ui323482297128]]) : (ui32) -> ()

    // addi 3002788344, 4294967294 -> no-fold
    %result_addi_ui323002788344_ui324294967294 = arc.addi %cst_ui323002788344, %cst_ui324294967294 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323002788344_ui324294967294:%[^ ]+]] = arc.addi [[CSTui323002788344]], [[CSTui324294967294]] : ui32
    "arc.keep"(%result_addi_ui323002788344_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323002788344_ui324294967294]]) : (ui32) -> ()

    // addi 3002788344, 4294967295 -> no-fold
    %result_addi_ui323002788344_ui324294967295 = arc.addi %cst_ui323002788344, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323002788344_ui324294967295:%[^ ]+]] = arc.addi [[CSTui323002788344]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui323002788344_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323002788344_ui324294967295]]) : (ui32) -> ()

    // addi 3482297128, 2119154652 -> no-fold
    %result_addi_ui323482297128_ui322119154652 = arc.addi %cst_ui323482297128, %cst_ui322119154652 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323482297128_ui322119154652:%[^ ]+]] = arc.addi [[CSTui323482297128]], [[CSTui322119154652]] : ui32
    "arc.keep"(%result_addi_ui323482297128_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323482297128_ui322119154652]]) : (ui32) -> ()

    // addi 3482297128, 3002788344 -> no-fold
    %result_addi_ui323482297128_ui323002788344 = arc.addi %cst_ui323482297128, %cst_ui323002788344 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323482297128_ui323002788344:%[^ ]+]] = arc.addi [[CSTui323482297128]], [[CSTui323002788344]] : ui32
    "arc.keep"(%result_addi_ui323482297128_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323482297128_ui323002788344]]) : (ui32) -> ()

    // addi 3482297128, 3482297128 -> no-fold
    %result_addi_ui323482297128_ui323482297128 = arc.addi %cst_ui323482297128, %cst_ui323482297128 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323482297128_ui323482297128:%[^ ]+]] = arc.addi [[CSTui323482297128]], [[CSTui323482297128]] : ui32
    "arc.keep"(%result_addi_ui323482297128_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323482297128_ui323482297128]]) : (ui32) -> ()

    // addi 3482297128, 4294967294 -> no-fold
    %result_addi_ui323482297128_ui324294967294 = arc.addi %cst_ui323482297128, %cst_ui324294967294 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323482297128_ui324294967294:%[^ ]+]] = arc.addi [[CSTui323482297128]], [[CSTui324294967294]] : ui32
    "arc.keep"(%result_addi_ui323482297128_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323482297128_ui324294967294]]) : (ui32) -> ()

    // addi 3482297128, 4294967295 -> no-fold
    %result_addi_ui323482297128_ui324294967295 = arc.addi %cst_ui323482297128, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui323482297128_ui324294967295:%[^ ]+]] = arc.addi [[CSTui323482297128]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui323482297128_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui323482297128_ui324294967295]]) : (ui32) -> ()

    // addi 4294967294, 2119154652 -> no-fold
    %result_addi_ui324294967294_ui322119154652 = arc.addi %cst_ui324294967294, %cst_ui322119154652 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967294_ui322119154652:%[^ ]+]] = arc.addi [[CSTui324294967294]], [[CSTui322119154652]] : ui32
    "arc.keep"(%result_addi_ui324294967294_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967294_ui322119154652]]) : (ui32) -> ()

    // addi 4294967294, 3002788344 -> no-fold
    %result_addi_ui324294967294_ui323002788344 = arc.addi %cst_ui324294967294, %cst_ui323002788344 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967294_ui323002788344:%[^ ]+]] = arc.addi [[CSTui324294967294]], [[CSTui323002788344]] : ui32
    "arc.keep"(%result_addi_ui324294967294_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967294_ui323002788344]]) : (ui32) -> ()

    // addi 4294967294, 3482297128 -> no-fold
    %result_addi_ui324294967294_ui323482297128 = arc.addi %cst_ui324294967294, %cst_ui323482297128 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967294_ui323482297128:%[^ ]+]] = arc.addi [[CSTui324294967294]], [[CSTui323482297128]] : ui32
    "arc.keep"(%result_addi_ui324294967294_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967294_ui323482297128]]) : (ui32) -> ()

    // addi 4294967294, 4294967294 -> no-fold
    %result_addi_ui324294967294_ui324294967294 = arc.addi %cst_ui324294967294, %cst_ui324294967294 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967294_ui324294967294:%[^ ]+]] = arc.addi [[CSTui324294967294]], [[CSTui324294967294]] : ui32
    "arc.keep"(%result_addi_ui324294967294_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967294_ui324294967294]]) : (ui32) -> ()

    // addi 4294967294, 4294967295 -> no-fold
    %result_addi_ui324294967294_ui324294967295 = arc.addi %cst_ui324294967294, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967294_ui324294967295:%[^ ]+]] = arc.addi [[CSTui324294967294]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui324294967294_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967294_ui324294967295]]) : (ui32) -> ()

    // addi 4294967295, 1 -> no-fold
    %result_addi_ui324294967295_ui321 = arc.addi %cst_ui324294967295, %cst_ui321 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui321:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui321]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui321) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui321]]) : (ui32) -> ()

    // addi 4294967295, 2119154652 -> no-fold
    %result_addi_ui324294967295_ui322119154652 = arc.addi %cst_ui324294967295, %cst_ui322119154652 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui322119154652:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui322119154652]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui322119154652) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui322119154652]]) : (ui32) -> ()

    // addi 4294967295, 3002788344 -> no-fold
    %result_addi_ui324294967295_ui323002788344 = arc.addi %cst_ui324294967295, %cst_ui323002788344 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui323002788344:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui323002788344]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui323002788344) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui323002788344]]) : (ui32) -> ()

    // addi 4294967295, 3482297128 -> no-fold
    %result_addi_ui324294967295_ui323482297128 = arc.addi %cst_ui324294967295, %cst_ui323482297128 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui323482297128:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui323482297128]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui323482297128) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui323482297128]]) : (ui32) -> ()

    // addi 4294967295, 4294967294 -> no-fold
    %result_addi_ui324294967295_ui324294967294 = arc.addi %cst_ui324294967295, %cst_ui324294967294 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui324294967294:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui324294967294]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui324294967294) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui324294967294]]) : (ui32) -> ()

    // addi 4294967295, 4294967295 -> no-fold
    %result_addi_ui324294967295_ui324294967295 = arc.addi %cst_ui324294967295, %cst_ui324294967295 : ui32
    // CHECK-DAG: [[RESULT_addi_ui324294967295_ui324294967295:%[^ ]+]] = arc.addi [[CSTui324294967295]], [[CSTui324294967295]] : ui32
    "arc.keep"(%result_addi_ui324294967295_ui324294967295) : (ui32) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui324294967295_ui324294967295]]) : (ui32) -> ()

    // addi 0, 0 -> 0
    %result_addi_ui640_ui640 = arc.addi %cst_ui640, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui640_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui640]]) : (ui64) -> ()

    // addi 0, 1 -> 1
    %result_addi_ui640_ui641 = arc.addi %cst_ui640, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui640_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui641]]) : (ui64) -> ()

    // addi 1, 0 -> 1
    %result_addi_ui641_ui640 = arc.addi %cst_ui641, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui641_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui641]]) : (ui64) -> ()

    // addi 1, 1 -> 2
    %result_addi_ui641_ui641 = arc.addi %cst_ui641, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui641_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui642]]) : (ui64) -> ()

    // addi 0, 191084152064409600 -> 191084152064409600
    %result_addi_ui640_ui64191084152064409600 = arc.addi %cst_ui640, %cst_ui64191084152064409600 : ui64
    "arc.keep"(%result_addi_ui640_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui64191084152064409600]]) : (ui64) -> ()

    // addi 191084152064409600, 0 -> 191084152064409600
    %result_addi_ui64191084152064409600_ui640 = arc.addi %cst_ui64191084152064409600, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui64191084152064409600]]) : (ui64) -> ()

    // addi 1, 191084152064409600 -> 191084152064409601
    %result_addi_ui641_ui64191084152064409600 = arc.addi %cst_ui641, %cst_ui64191084152064409600 : ui64
    "arc.keep"(%result_addi_ui641_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui64191084152064409601]]) : (ui64) -> ()

    // addi 191084152064409600, 1 -> 191084152064409601
    %result_addi_ui64191084152064409600_ui641 = arc.addi %cst_ui64191084152064409600, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui64191084152064409601]]) : (ui64) -> ()

    // addi 191084152064409600, 191084152064409600 -> 382168304128819200
    %result_addi_ui64191084152064409600_ui64191084152064409600 = arc.addi %cst_ui64191084152064409600, %cst_ui64191084152064409600 : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui64382168304128819200]]) : (ui64) -> ()

    // addi 0, 11015955194427482112 -> 11015955194427482112
    %result_addi_ui640_ui6411015955194427482112 = arc.addi %cst_ui640, %cst_ui6411015955194427482112 : ui64
    "arc.keep"(%result_addi_ui640_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411015955194427482112]]) : (ui64) -> ()

    // addi 11015955194427482112, 0 -> 11015955194427482112
    %result_addi_ui6411015955194427482112_ui640 = arc.addi %cst_ui6411015955194427482112, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411015955194427482112]]) : (ui64) -> ()

    // addi 1, 11015955194427482112 -> 11015955194427482113
    %result_addi_ui641_ui6411015955194427482112 = arc.addi %cst_ui641, %cst_ui6411015955194427482112 : ui64
    "arc.keep"(%result_addi_ui641_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411015955194427482113]]) : (ui64) -> ()

    // addi 11015955194427482112, 1 -> 11015955194427482113
    %result_addi_ui6411015955194427482112_ui641 = arc.addi %cst_ui6411015955194427482112, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411015955194427482113]]) : (ui64) -> ()

    // addi 191084152064409600, 11015955194427482112 -> 11207039346491891712
    %result_addi_ui64191084152064409600_ui6411015955194427482112 = arc.addi %cst_ui64191084152064409600, %cst_ui6411015955194427482112 : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411207039346491891712]]) : (ui64) -> ()

    // addi 11015955194427482112, 191084152064409600 -> 11207039346491891712
    %result_addi_ui6411015955194427482112_ui64191084152064409600 = arc.addi %cst_ui6411015955194427482112, %cst_ui64191084152064409600 : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6411207039346491891712]]) : (ui64) -> ()

    // addi 0, 16990600415051759616 -> 16990600415051759616
    %result_addi_ui640_ui6416990600415051759616 = arc.addi %cst_ui640, %cst_ui6416990600415051759616 : ui64
    "arc.keep"(%result_addi_ui640_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6416990600415051759616]]) : (ui64) -> ()

    // addi 16990600415051759616, 0 -> 16990600415051759616
    %result_addi_ui6416990600415051759616_ui640 = arc.addi %cst_ui6416990600415051759616, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6416990600415051759616]]) : (ui64) -> ()

    // addi 1, 16990600415051759616 -> 16990600415051759617
    %result_addi_ui641_ui6416990600415051759616 = arc.addi %cst_ui641, %cst_ui6416990600415051759616 : ui64
    "arc.keep"(%result_addi_ui641_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6416990600415051759617]]) : (ui64) -> ()

    // addi 16990600415051759616, 1 -> 16990600415051759617
    %result_addi_ui6416990600415051759616_ui641 = arc.addi %cst_ui6416990600415051759616, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6416990600415051759617]]) : (ui64) -> ()

    // addi 191084152064409600, 16990600415051759616 -> 17181684567116169216
    %result_addi_ui64191084152064409600_ui6416990600415051759616 = arc.addi %cst_ui64191084152064409600, %cst_ui6416990600415051759616 : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6417181684567116169216]]) : (ui64) -> ()

    // addi 16990600415051759616, 191084152064409600 -> 17181684567116169216
    %result_addi_ui6416990600415051759616_ui64191084152064409600 = arc.addi %cst_ui6416990600415051759616, %cst_ui64191084152064409600 : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6417181684567116169216]]) : (ui64) -> ()

    // addi 0, 18446744073709551614 -> 18446744073709551614
    %result_addi_ui640_ui6418446744073709551614 = arc.addi %cst_ui640, %cst_ui6418446744073709551614 : ui64
    "arc.keep"(%result_addi_ui640_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551614]]) : (ui64) -> ()

    // addi 18446744073709551614, 0 -> 18446744073709551614
    %result_addi_ui6418446744073709551614_ui640 = arc.addi %cst_ui6418446744073709551614, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551614]]) : (ui64) -> ()

    // addi 0, 18446744073709551615 -> 18446744073709551615
    %result_addi_ui640_ui6418446744073709551615 = arc.addi %cst_ui640, %cst_ui6418446744073709551615 : ui64
    "arc.keep"(%result_addi_ui640_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551615]]) : (ui64) -> ()

    // addi 1, 18446744073709551614 -> 18446744073709551615
    %result_addi_ui641_ui6418446744073709551614 = arc.addi %cst_ui641, %cst_ui6418446744073709551614 : ui64
    "arc.keep"(%result_addi_ui641_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551615]]) : (ui64) -> ()

    // addi 18446744073709551614, 1 -> 18446744073709551615
    %result_addi_ui6418446744073709551614_ui641 = arc.addi %cst_ui6418446744073709551614, %cst_ui641 : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551615]]) : (ui64) -> ()

    // addi 18446744073709551615, 0 -> 18446744073709551615
    %result_addi_ui6418446744073709551615_ui640 = arc.addi %cst_ui6418446744073709551615, %cst_ui640 : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui640) : (ui64) -> ()
    // CHECK: "arc.keep"([[CSTui6418446744073709551615]]) : (ui64) -> ()

    // addi 1, 18446744073709551615 -> no-fold
    %result_addi_ui641_ui6418446744073709551615 = arc.addi %cst_ui641, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui641_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui641]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui641_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui641_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 191084152064409600, 18446744073709551614 -> no-fold
    %result_addi_ui64191084152064409600_ui6418446744073709551614 = arc.addi %cst_ui64191084152064409600, %cst_ui6418446744073709551614 : ui64
    // CHECK-DAG: [[RESULT_addi_ui64191084152064409600_ui6418446744073709551614:%[^ ]+]] = arc.addi [[CSTui64191084152064409600]], [[CSTui6418446744073709551614]] : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui64191084152064409600_ui6418446744073709551614]]) : (ui64) -> ()

    // addi 191084152064409600, 18446744073709551615 -> no-fold
    %result_addi_ui64191084152064409600_ui6418446744073709551615 = arc.addi %cst_ui64191084152064409600, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui64191084152064409600_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui64191084152064409600]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui64191084152064409600_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui64191084152064409600_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 11015955194427482112, 11015955194427482112 -> no-fold
    %result_addi_ui6411015955194427482112_ui6411015955194427482112 = arc.addi %cst_ui6411015955194427482112, %cst_ui6411015955194427482112 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6411015955194427482112_ui6411015955194427482112:%[^ ]+]] = arc.addi [[CSTui6411015955194427482112]], [[CSTui6411015955194427482112]] : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6411015955194427482112_ui6411015955194427482112]]) : (ui64) -> ()

    // addi 11015955194427482112, 16990600415051759616 -> no-fold
    %result_addi_ui6411015955194427482112_ui6416990600415051759616 = arc.addi %cst_ui6411015955194427482112, %cst_ui6416990600415051759616 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6411015955194427482112_ui6416990600415051759616:%[^ ]+]] = arc.addi [[CSTui6411015955194427482112]], [[CSTui6416990600415051759616]] : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6411015955194427482112_ui6416990600415051759616]]) : (ui64) -> ()

    // addi 11015955194427482112, 18446744073709551614 -> no-fold
    %result_addi_ui6411015955194427482112_ui6418446744073709551614 = arc.addi %cst_ui6411015955194427482112, %cst_ui6418446744073709551614 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6411015955194427482112_ui6418446744073709551614:%[^ ]+]] = arc.addi [[CSTui6411015955194427482112]], [[CSTui6418446744073709551614]] : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6411015955194427482112_ui6418446744073709551614]]) : (ui64) -> ()

    // addi 11015955194427482112, 18446744073709551615 -> no-fold
    %result_addi_ui6411015955194427482112_ui6418446744073709551615 = arc.addi %cst_ui6411015955194427482112, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6411015955194427482112_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui6411015955194427482112]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui6411015955194427482112_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6411015955194427482112_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 16990600415051759616, 11015955194427482112 -> no-fold
    %result_addi_ui6416990600415051759616_ui6411015955194427482112 = arc.addi %cst_ui6416990600415051759616, %cst_ui6411015955194427482112 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6416990600415051759616_ui6411015955194427482112:%[^ ]+]] = arc.addi [[CSTui6416990600415051759616]], [[CSTui6411015955194427482112]] : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6416990600415051759616_ui6411015955194427482112]]) : (ui64) -> ()

    // addi 16990600415051759616, 16990600415051759616 -> no-fold
    %result_addi_ui6416990600415051759616_ui6416990600415051759616 = arc.addi %cst_ui6416990600415051759616, %cst_ui6416990600415051759616 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6416990600415051759616_ui6416990600415051759616:%[^ ]+]] = arc.addi [[CSTui6416990600415051759616]], [[CSTui6416990600415051759616]] : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6416990600415051759616_ui6416990600415051759616]]) : (ui64) -> ()

    // addi 16990600415051759616, 18446744073709551614 -> no-fold
    %result_addi_ui6416990600415051759616_ui6418446744073709551614 = arc.addi %cst_ui6416990600415051759616, %cst_ui6418446744073709551614 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6416990600415051759616_ui6418446744073709551614:%[^ ]+]] = arc.addi [[CSTui6416990600415051759616]], [[CSTui6418446744073709551614]] : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6416990600415051759616_ui6418446744073709551614]]) : (ui64) -> ()

    // addi 16990600415051759616, 18446744073709551615 -> no-fold
    %result_addi_ui6416990600415051759616_ui6418446744073709551615 = arc.addi %cst_ui6416990600415051759616, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6416990600415051759616_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui6416990600415051759616]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui6416990600415051759616_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6416990600415051759616_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 18446744073709551614, 191084152064409600 -> no-fold
    %result_addi_ui6418446744073709551614_ui64191084152064409600 = arc.addi %cst_ui6418446744073709551614, %cst_ui64191084152064409600 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551614_ui64191084152064409600:%[^ ]+]] = arc.addi [[CSTui6418446744073709551614]], [[CSTui64191084152064409600]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551614_ui64191084152064409600]]) : (ui64) -> ()

    // addi 18446744073709551614, 11015955194427482112 -> no-fold
    %result_addi_ui6418446744073709551614_ui6411015955194427482112 = arc.addi %cst_ui6418446744073709551614, %cst_ui6411015955194427482112 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551614_ui6411015955194427482112:%[^ ]+]] = arc.addi [[CSTui6418446744073709551614]], [[CSTui6411015955194427482112]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551614_ui6411015955194427482112]]) : (ui64) -> ()

    // addi 18446744073709551614, 16990600415051759616 -> no-fold
    %result_addi_ui6418446744073709551614_ui6416990600415051759616 = arc.addi %cst_ui6418446744073709551614, %cst_ui6416990600415051759616 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551614_ui6416990600415051759616:%[^ ]+]] = arc.addi [[CSTui6418446744073709551614]], [[CSTui6416990600415051759616]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551614_ui6416990600415051759616]]) : (ui64) -> ()

    // addi 18446744073709551614, 18446744073709551614 -> no-fold
    %result_addi_ui6418446744073709551614_ui6418446744073709551614 = arc.addi %cst_ui6418446744073709551614, %cst_ui6418446744073709551614 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551614_ui6418446744073709551614:%[^ ]+]] = arc.addi [[CSTui6418446744073709551614]], [[CSTui6418446744073709551614]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551614_ui6418446744073709551614]]) : (ui64) -> ()

    // addi 18446744073709551614, 18446744073709551615 -> no-fold
    %result_addi_ui6418446744073709551614_ui6418446744073709551615 = arc.addi %cst_ui6418446744073709551614, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551614_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui6418446744073709551614]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551614_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551614_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 18446744073709551615, 1 -> no-fold
    %result_addi_ui6418446744073709551615_ui641 = arc.addi %cst_ui6418446744073709551615, %cst_ui641 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui641:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui641]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui641) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui641]]) : (ui64) -> ()

    // addi 18446744073709551615, 191084152064409600 -> no-fold
    %result_addi_ui6418446744073709551615_ui64191084152064409600 = arc.addi %cst_ui6418446744073709551615, %cst_ui64191084152064409600 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui64191084152064409600:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui64191084152064409600]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui64191084152064409600) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui64191084152064409600]]) : (ui64) -> ()

    // addi 18446744073709551615, 11015955194427482112 -> no-fold
    %result_addi_ui6418446744073709551615_ui6411015955194427482112 = arc.addi %cst_ui6418446744073709551615, %cst_ui6411015955194427482112 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui6411015955194427482112:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui6411015955194427482112]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui6411015955194427482112) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui6411015955194427482112]]) : (ui64) -> ()

    // addi 18446744073709551615, 16990600415051759616 -> no-fold
    %result_addi_ui6418446744073709551615_ui6416990600415051759616 = arc.addi %cst_ui6418446744073709551615, %cst_ui6416990600415051759616 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui6416990600415051759616:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui6416990600415051759616]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui6416990600415051759616) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui6416990600415051759616]]) : (ui64) -> ()

    // addi 18446744073709551615, 18446744073709551614 -> no-fold
    %result_addi_ui6418446744073709551615_ui6418446744073709551614 = arc.addi %cst_ui6418446744073709551615, %cst_ui6418446744073709551614 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui6418446744073709551614:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui6418446744073709551614]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui6418446744073709551614) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui6418446744073709551614]]) : (ui64) -> ()

    // addi 18446744073709551615, 18446744073709551615 -> no-fold
    %result_addi_ui6418446744073709551615_ui6418446744073709551615 = arc.addi %cst_ui6418446744073709551615, %cst_ui6418446744073709551615 : ui64
    // CHECK-DAG: [[RESULT_addi_ui6418446744073709551615_ui6418446744073709551615:%[^ ]+]] = arc.addi [[CSTui6418446744073709551615]], [[CSTui6418446744073709551615]] : ui64
    "arc.keep"(%result_addi_ui6418446744073709551615_ui6418446744073709551615) : (ui64) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui6418446744073709551615_ui6418446744073709551615]]) : (ui64) -> ()

    // addi 0, 0 -> 0
    %result_addi_ui80_ui80 = arc.addi %cst_ui80, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui80_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui80]]) : (ui8) -> ()

    // addi 0, 1 -> 1
    %result_addi_ui80_ui81 = arc.addi %cst_ui80, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui80_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui81]]) : (ui8) -> ()

    // addi 1, 0 -> 1
    %result_addi_ui81_ui80 = arc.addi %cst_ui81, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui81_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui81]]) : (ui8) -> ()

    // addi 1, 1 -> 2
    %result_addi_ui81_ui81 = arc.addi %cst_ui81, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui81_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui82]]) : (ui8) -> ()

    // addi 0, 72 -> 72
    %result_addi_ui80_ui872 = arc.addi %cst_ui80, %cst_ui872 : ui8
    "arc.keep"(%result_addi_ui80_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui872]]) : (ui8) -> ()

    // addi 72, 0 -> 72
    %result_addi_ui872_ui80 = arc.addi %cst_ui872, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui872_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui872]]) : (ui8) -> ()

    // addi 1, 72 -> 73
    %result_addi_ui81_ui872 = arc.addi %cst_ui81, %cst_ui872 : ui8
    "arc.keep"(%result_addi_ui81_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui873]]) : (ui8) -> ()

    // addi 72, 1 -> 73
    %result_addi_ui872_ui81 = arc.addi %cst_ui872, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui872_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui873]]) : (ui8) -> ()

    // addi 0, 100 -> 100
    %result_addi_ui80_ui8100 = arc.addi %cst_ui80, %cst_ui8100 : ui8
    "arc.keep"(%result_addi_ui80_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8100]]) : (ui8) -> ()

    // addi 100, 0 -> 100
    %result_addi_ui8100_ui80 = arc.addi %cst_ui8100, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui8100_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8100]]) : (ui8) -> ()

    // addi 1, 100 -> 101
    %result_addi_ui81_ui8100 = arc.addi %cst_ui81, %cst_ui8100 : ui8
    "arc.keep"(%result_addi_ui81_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8101]]) : (ui8) -> ()

    // addi 100, 1 -> 101
    %result_addi_ui8100_ui81 = arc.addi %cst_ui8100, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui8100_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8101]]) : (ui8) -> ()

    // addi 72, 72 -> 144
    %result_addi_ui872_ui872 = arc.addi %cst_ui872, %cst_ui872 : ui8
    "arc.keep"(%result_addi_ui872_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8144]]) : (ui8) -> ()

    // addi 0, 162 -> 162
    %result_addi_ui80_ui8162 = arc.addi %cst_ui80, %cst_ui8162 : ui8
    "arc.keep"(%result_addi_ui80_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8162]]) : (ui8) -> ()

    // addi 162, 0 -> 162
    %result_addi_ui8162_ui80 = arc.addi %cst_ui8162, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui8162_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8162]]) : (ui8) -> ()

    // addi 1, 162 -> 163
    %result_addi_ui81_ui8162 = arc.addi %cst_ui81, %cst_ui8162 : ui8
    "arc.keep"(%result_addi_ui81_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8163]]) : (ui8) -> ()

    // addi 162, 1 -> 163
    %result_addi_ui8162_ui81 = arc.addi %cst_ui8162, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui8162_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8163]]) : (ui8) -> ()

    // addi 72, 100 -> 172
    %result_addi_ui872_ui8100 = arc.addi %cst_ui872, %cst_ui8100 : ui8
    "arc.keep"(%result_addi_ui872_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8172]]) : (ui8) -> ()

    // addi 100, 72 -> 172
    %result_addi_ui8100_ui872 = arc.addi %cst_ui8100, %cst_ui872 : ui8
    "arc.keep"(%result_addi_ui8100_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8172]]) : (ui8) -> ()

    // addi 100, 100 -> 200
    %result_addi_ui8100_ui8100 = arc.addi %cst_ui8100, %cst_ui8100 : ui8
    "arc.keep"(%result_addi_ui8100_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8200]]) : (ui8) -> ()

    // addi 72, 162 -> 234
    %result_addi_ui872_ui8162 = arc.addi %cst_ui872, %cst_ui8162 : ui8
    "arc.keep"(%result_addi_ui872_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8234]]) : (ui8) -> ()

    // addi 162, 72 -> 234
    %result_addi_ui8162_ui872 = arc.addi %cst_ui8162, %cst_ui872 : ui8
    "arc.keep"(%result_addi_ui8162_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8234]]) : (ui8) -> ()

    // addi 0, 254 -> 254
    %result_addi_ui80_ui8254 = arc.addi %cst_ui80, %cst_ui8254 : ui8
    "arc.keep"(%result_addi_ui80_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8254]]) : (ui8) -> ()

    // addi 254, 0 -> 254
    %result_addi_ui8254_ui80 = arc.addi %cst_ui8254, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui8254_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8254]]) : (ui8) -> ()

    // addi 0, 255 -> 255
    %result_addi_ui80_ui8255 = arc.addi %cst_ui80, %cst_ui8255 : ui8
    "arc.keep"(%result_addi_ui80_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8255]]) : (ui8) -> ()

    // addi 1, 254 -> 255
    %result_addi_ui81_ui8254 = arc.addi %cst_ui81, %cst_ui8254 : ui8
    "arc.keep"(%result_addi_ui81_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8255]]) : (ui8) -> ()

    // addi 254, 1 -> 255
    %result_addi_ui8254_ui81 = arc.addi %cst_ui8254, %cst_ui81 : ui8
    "arc.keep"(%result_addi_ui8254_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8255]]) : (ui8) -> ()

    // addi 255, 0 -> 255
    %result_addi_ui8255_ui80 = arc.addi %cst_ui8255, %cst_ui80 : ui8
    "arc.keep"(%result_addi_ui8255_ui80) : (ui8) -> ()
    // CHECK: "arc.keep"([[CSTui8255]]) : (ui8) -> ()

    // addi 1, 255 -> no-fold
    %result_addi_ui81_ui8255 = arc.addi %cst_ui81, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui81_ui8255:%[^ ]+]] = arc.addi [[CSTui81]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui81_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui81_ui8255]]) : (ui8) -> ()

    // addi 72, 254 -> no-fold
    %result_addi_ui872_ui8254 = arc.addi %cst_ui872, %cst_ui8254 : ui8
    // CHECK-DAG: [[RESULT_addi_ui872_ui8254:%[^ ]+]] = arc.addi [[CSTui872]], [[CSTui8254]] : ui8
    "arc.keep"(%result_addi_ui872_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui872_ui8254]]) : (ui8) -> ()

    // addi 72, 255 -> no-fold
    %result_addi_ui872_ui8255 = arc.addi %cst_ui872, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui872_ui8255:%[^ ]+]] = arc.addi [[CSTui872]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui872_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui872_ui8255]]) : (ui8) -> ()

    // addi 100, 162 -> no-fold
    %result_addi_ui8100_ui8162 = arc.addi %cst_ui8100, %cst_ui8162 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8100_ui8162:%[^ ]+]] = arc.addi [[CSTui8100]], [[CSTui8162]] : ui8
    "arc.keep"(%result_addi_ui8100_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8100_ui8162]]) : (ui8) -> ()

    // addi 100, 254 -> no-fold
    %result_addi_ui8100_ui8254 = arc.addi %cst_ui8100, %cst_ui8254 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8100_ui8254:%[^ ]+]] = arc.addi [[CSTui8100]], [[CSTui8254]] : ui8
    "arc.keep"(%result_addi_ui8100_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8100_ui8254]]) : (ui8) -> ()

    // addi 100, 255 -> no-fold
    %result_addi_ui8100_ui8255 = arc.addi %cst_ui8100, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8100_ui8255:%[^ ]+]] = arc.addi [[CSTui8100]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui8100_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8100_ui8255]]) : (ui8) -> ()

    // addi 162, 100 -> no-fold
    %result_addi_ui8162_ui8100 = arc.addi %cst_ui8162, %cst_ui8100 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8162_ui8100:%[^ ]+]] = arc.addi [[CSTui8162]], [[CSTui8100]] : ui8
    "arc.keep"(%result_addi_ui8162_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8162_ui8100]]) : (ui8) -> ()

    // addi 162, 162 -> no-fold
    %result_addi_ui8162_ui8162 = arc.addi %cst_ui8162, %cst_ui8162 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8162_ui8162:%[^ ]+]] = arc.addi [[CSTui8162]], [[CSTui8162]] : ui8
    "arc.keep"(%result_addi_ui8162_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8162_ui8162]]) : (ui8) -> ()

    // addi 162, 254 -> no-fold
    %result_addi_ui8162_ui8254 = arc.addi %cst_ui8162, %cst_ui8254 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8162_ui8254:%[^ ]+]] = arc.addi [[CSTui8162]], [[CSTui8254]] : ui8
    "arc.keep"(%result_addi_ui8162_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8162_ui8254]]) : (ui8) -> ()

    // addi 162, 255 -> no-fold
    %result_addi_ui8162_ui8255 = arc.addi %cst_ui8162, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8162_ui8255:%[^ ]+]] = arc.addi [[CSTui8162]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui8162_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8162_ui8255]]) : (ui8) -> ()

    // addi 254, 72 -> no-fold
    %result_addi_ui8254_ui872 = arc.addi %cst_ui8254, %cst_ui872 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8254_ui872:%[^ ]+]] = arc.addi [[CSTui8254]], [[CSTui872]] : ui8
    "arc.keep"(%result_addi_ui8254_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8254_ui872]]) : (ui8) -> ()

    // addi 254, 100 -> no-fold
    %result_addi_ui8254_ui8100 = arc.addi %cst_ui8254, %cst_ui8100 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8254_ui8100:%[^ ]+]] = arc.addi [[CSTui8254]], [[CSTui8100]] : ui8
    "arc.keep"(%result_addi_ui8254_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8254_ui8100]]) : (ui8) -> ()

    // addi 254, 162 -> no-fold
    %result_addi_ui8254_ui8162 = arc.addi %cst_ui8254, %cst_ui8162 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8254_ui8162:%[^ ]+]] = arc.addi [[CSTui8254]], [[CSTui8162]] : ui8
    "arc.keep"(%result_addi_ui8254_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8254_ui8162]]) : (ui8) -> ()

    // addi 254, 254 -> no-fold
    %result_addi_ui8254_ui8254 = arc.addi %cst_ui8254, %cst_ui8254 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8254_ui8254:%[^ ]+]] = arc.addi [[CSTui8254]], [[CSTui8254]] : ui8
    "arc.keep"(%result_addi_ui8254_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8254_ui8254]]) : (ui8) -> ()

    // addi 254, 255 -> no-fold
    %result_addi_ui8254_ui8255 = arc.addi %cst_ui8254, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8254_ui8255:%[^ ]+]] = arc.addi [[CSTui8254]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui8254_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8254_ui8255]]) : (ui8) -> ()

    // addi 255, 1 -> no-fold
    %result_addi_ui8255_ui81 = arc.addi %cst_ui8255, %cst_ui81 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui81:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui81]] : ui8
    "arc.keep"(%result_addi_ui8255_ui81) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui81]]) : (ui8) -> ()

    // addi 255, 72 -> no-fold
    %result_addi_ui8255_ui872 = arc.addi %cst_ui8255, %cst_ui872 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui872:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui872]] : ui8
    "arc.keep"(%result_addi_ui8255_ui872) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui872]]) : (ui8) -> ()

    // addi 255, 100 -> no-fold
    %result_addi_ui8255_ui8100 = arc.addi %cst_ui8255, %cst_ui8100 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui8100:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui8100]] : ui8
    "arc.keep"(%result_addi_ui8255_ui8100) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui8100]]) : (ui8) -> ()

    // addi 255, 162 -> no-fold
    %result_addi_ui8255_ui8162 = arc.addi %cst_ui8255, %cst_ui8162 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui8162:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui8162]] : ui8
    "arc.keep"(%result_addi_ui8255_ui8162) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui8162]]) : (ui8) -> ()

    // addi 255, 254 -> no-fold
    %result_addi_ui8255_ui8254 = arc.addi %cst_ui8255, %cst_ui8254 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui8254:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui8254]] : ui8
    "arc.keep"(%result_addi_ui8255_ui8254) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui8254]]) : (ui8) -> ()

    // addi 255, 255 -> no-fold
    %result_addi_ui8255_ui8255 = arc.addi %cst_ui8255, %cst_ui8255 : ui8
    // CHECK-DAG: [[RESULT_addi_ui8255_ui8255:%[^ ]+]] = arc.addi [[CSTui8255]], [[CSTui8255]] : ui8
    "arc.keep"(%result_addi_ui8255_ui8255) : (ui8) -> ()
    // CHECK: "arc.keep"([[RESULT_addi_ui8255_ui8255]]) : (ui8) -> ()

    return
  }
}
