//
// Created by 杜建璋 on 2024/10/25.
//

#ifndef MPC_PACKAGE_FIXED_AB_PAIRS_0_H
#define MPC_PACKAGE_FIXED_AB_PAIRS_0_H

#include <array>

// 8 bit
std::array<int8_t, 100> R_BOOL_8 = {-5, 99, 14, -106, 1, -73, 67, -63, -11, -79, 32, -66, -72, -21, 38, 44, 120, -62,
                                    111, -43, -82, -10, -52, -81, -32, -55, 47, -103, -125, 80, 35, 50, 50, -127, -128,
                                    -16, -19, 10, -17, 21, -31, -89, 110, 80, 88, 4, 49, -115, -5, 81, 119, -51, 19, 25,
                                    -16, 33, 19, -47, 116, 122, 113, -67, -8, -47, -31, -21, -46, 120, -73, 123, -106,
                                    61, 85, 118, 72, 15, -53, 63, -79, 49, -71, -7, 16, 51, 40, 41, -24, -122, -105,
                                    -101, 0, -111, -101, 35, 108, -66, 49, -104, -70, 43};
std::array<int8_t, 100> R_ARITH_8 = {41, 25, -69, 94, 2, 124, -96, -5, 39, 15, -31, 7, 97, -83, -43, -18, 91, 29, -86,
                                     33, -94, 59, -5, 63, 92, -104, 127, -91, -108, -75, 53, 73, -92, 112, -120, 21, 0,
                                     -67, -91, 12, 6, -43, 107, -119, 3, 99, 56, 83, -108, -115, -95, 15, 93, -16, -39,
                                     -20, 76, -128, -40, 78, 93, 56, -21, 125, 49, 93, 92, 99, 33, 50, -119, 25, -110,
                                     -35, -65, -121, -74, 34, -20, 67, -71, -102, -104, 64, 30, 3, 11, 20, -122, -92,
                                     -70, -66, 97, 110, 13, -122, 73, 38, -70, 104};

// 16 bit
std::array<int16_t, 100> R_BOOL_16 = {31565, -19779, 15428, -9397, -13939, -10823, -10965, -32314, -3860, 27840, -13793,
                                      23175, -30354, -21339, -2112, -6503, 23538, -28612, 2926, 22155, -26866, 2790,
                                      -4068, 30270, -14936, 13097, 7470, -24841, -17014, 4849, 17953, 27322, 1024,
                                      -1473, 7322, -7656, -13539, 32258, -22013, -14536, -1724, 9193, -31639, 647, 4388,
                                      21010, -31548, -28239, 23544, 13412, 17821, 31926, -14354, -24481, -11695, -7549,
                                      30604, -2872, -5510, 26489, -12854, -19618, 23406, -31279, 4753, 22155, 20692,
                                      19669, -32552, -17438, -27413, -8602, 11834, -26605, 13720, -21859, -31647,
                                      -29209, 4571, -8718, -30397, -15918, -31565, -1946, 28087, -15224, -2339, -9625,
                                      -22744, -13352, -16751, -2357, -3895, 26244, -25925, -16876, 5298, 15487, -29744,
                                      -17970};
std::array<int16_t, 100> R_ARITH_16 = {-25931, 15618, -2977, 17652, -8261, 9852, -8366, -20299, -4999, 7734, 13892,
                                       7739, -32672, -2367, 30777, -18564, 7886, -246, -29988, 10002, 28576, 17160,
                                       18028, -2783, 5529, -788, 5387, -29273, 22748, 9461, 11519, 409, 29057, -346,
                                       23858, 16801, 22746, 6166, 15945, 3747, -14350, -29055, 1641, -1473, -31842,
                                       -7956, -22484, -10265, -4270, -9830, 5192, 20429, 873, 14461, -17670, -32030,
                                       -9982, -4321, 14667, 31910, -16892, 11027, -21098, -12988, 27053, -31322, 29580,
                                       23807, -29331, -18622, -28180, 6439, -17847, -10188, -16710, 235, 32766, -24975,
                                       28215, -26694, -639, -12553, -21442, -17961, -5784, 4289, -25567, 7056, 20848,
                                       28307, -13422, 26739, 4592, -31828, -18858, 17411, -29172, 16349, 25575, 25848};

// 32 bit
std::array<int32_t, 100> R_BOOL_32 = {2033968399, 555547143, -1830463142, 315727744, -1212061995, 1243407456,
                                      -378817231, 2031527798, 2045935831, 714462877, -986602900, -1902598666,
                                      2070363506, -481311694, 1986794180, 1655467081, -156757878, 27527158, -1656784281,
                                      2088366676, 1257715586, -660828438, -147324092, -1587845751, 1906715956,
                                      1978563997, -993639752, 86328683, -213203595, 1919879735, -117962366, 1700186378,
                                      425826486, -1172115038, 764163828, 150098016, 293192870, -850165089, -457016437,
                                      -612960579, -755435690, 667839320, 461692286, 531232565, 556764799, 1879663485,
                                      -1424045242, -1795948559, -428714136, -214676989, 595930965, -1365187223,
                                      1166241409, -378863884, -1433792351, -468777869, -1994683444, 1477121899,
                                      2036580500, 977205056, -2121867376, -1851365309, 199820439, -149359359, 39140799,
                                      -176586714, -672380312, 1979703768, -1713553790, 536903023, 125497575, 687902628,
                                      736669371, 261482752, -1216796053, 2114819209, -601467257, 72734485, -541978527,
                                      1770772158, 79553028, -1068435729, -1553235617, 365818899, 1404031663, 1595973915,
                                      -573469217, -2018431994, 286403549, 637728401, -881889096, -390488565,
                                      -1935879731, -310813969, 2093351483, -1086700989, -445721591, 665793679,
                                      137953162, 1143266861};
std::array<int32_t, 100> R_ARITH_32 = {-1383092341, -1312328583, 1134024120, -645720625, -1303438557, 1206421478,
                                       236434202, -566162866, -1524751100, 1856506052, -1351541945, -196838933,
                                       108986218, -2043787670, -1532261991, -1195082688, 663584930, 1292316381,
                                       -1017401523, 897525956, -1588658085, -1457508897, 370539654, 1981789911,
                                       -2035498263, 1649332728, -606920194, 271791889, 676804945, 1707893797, 380283286,
                                       -311400635, -1912052907, 488378982, -1369483950, -628728326, 721600740,
                                       2017072842, 880835063, 1347194093, 363648889, 6067817, 21861859, -2050654058,
                                       -1366425983, 1926297764, 879200170, 298881130, 481357391, 683740387, -1299857711,
                                       -918366807, 1930126407, 1700261880, -604680923, -1400240657, 1649347667,
                                       1951929794, -1989349997, 662614355, -1120396214, 669207410, -396760000,
                                       1423901668, -720470065, -506140747, 1962099692, -530874921, 722477714,
                                       1194952824, -11711019, -765751501, 1031830182, 116839657, 913328768, 1883267150,
                                       549030968, 273398190, -724169328, 709115760, 258710416, 1406310875, -1359977531,
                                       376278177, 2121700072, 1853975637, 90681061, 577147533, -220091795, 1407426351,
                                       -795039599, 1489822736, -502596262, 1279241527, -1659473562, -1096751734,
                                       844534978, 2111391929, -2091060247, -1605029651};

// 64 bit
std::array<int64_t, 100> R_BOOL_64 = {-8882390002803861750, -5119629161818581848, 4231078480236613361,
                                      -3608666111942787326, 4299707662842254264, 1831338141022482546,
                                      7007024492093061709, -3910105658616167593, 528859210176284054,
                                      5474150757448805648, -807138018556356863, 2264899991818726155,
                                      -1272330632221294365, -8562892025055529037, 2722378043046449797,
                                      -2240812580173226717, -7753840151868270117, 1171946988568031147,
                                      -7824276460951236769, -5982621610592345842, 7656525635539639640,
                                      6535528118720212172, 391130701110827306, -4969885967513519666,
                                      -7742680343638584657, 1876062806281652494, -7098857384508310339,
                                      -3972350935889358632, -947546605523205315, 1170741623087453710,
                                      -6619890767074607960, -8032687578194712476, 5217104550654404088,
                                      -7493091990296618616, 1540742214791449524, -7683149491655886310,
                                      3674296228999676755, -6636738731886751380, -7892921501140064395,
                                      3628361537363372235, -8196059555216921882, -8296783386876738458,
                                      1654313698794728223, -639534761130998237, 3067590061241211957,
                                      -3042249962329828556, 5780145221035318313, 54566264852628010, 304903654143280606,
                                      -948277016727310307, 6294437682920087970, 2286420292361401279,
                                      -1082272023244184450, -7217197795256075943, -7904000189528575828,
                                      7500926991700764836, -6765319085738154635, -4734594472139596451,
                                      7087173537845071955, 5358555085286985488, 3386724860630846218,
                                      5578993599241603064, -5438325139551656164, 8795601246918852137,
                                      6212962297569215286, -8379193254014943259, -6441123296767040553,
                                      -438332770695536776, -2172901339979065833, 94295156103019449,
                                      -3449301462920958964, 79719050432341873, 736152833550466150, -3884307490202732350,
                                      5014883121988201436, -6672541048301773034, -4058853609622659892,
                                      -742334337744263710, 3710069102605399848, 8755089198671657635,
                                      -4872071830147973912, 6764710411648356384, 5593433271881571391,
                                      6763423986840897868, -8342287873889324464, 8836974150905563258,
                                      -4402819586109675570, 3804291883073065571, 948026313101353676,
                                      -499130576914971235, -2792644028872199845, 7418263500277021893,
                                      2694114786322560992, 4061258396109354054, -1849911890378420295,
                                      9118057679900777911, -3936386868030296687, 1604225206776741424,
                                      -3320942249089341849, 6737389433003496613};
std::array<int64_t, 100> R_ARITH_64 = {-8276309856613328811, -6767114504700108738, 29388598104882488,
                                       -8347403970586833788, -3544828600467562683, -1298165952702491454,
                                       -573563415284019549, 2720502832721861001, 6913904853441059293,
                                       2555488053472141989, -962083588325003090, 3547916251626501711,
                                       -6832831998176057755, -7248633753760733689, -5618383221648879018,
                                       -5391765772483732329, 4664764522739862106, -3087698935148103808,
                                       3459277373695706603, -8823150524458428845, -571894703657843745,
                                       -6270514179176170281, -8517025949848257723, 8253513181391768453,
                                       -7256432519525491927, 391282459926439480, -5738909333897241171,
                                       -6085658968261324463, 6026970917614162125, -8807114266831405233,
                                       -359750688328042123, -103604466428433939, -5574343515212980821,
                                       -2461904246423037823, -7226586320386460639, 5514202844915190709,
                                       -5560808381418376004, 1991545012844356425, -3029219260551040545,
                                       -5880231780155016727, 6790102401361765558, -232442980128390748,
                                       -7673755252571721470, 149689576650014659, -3855582802456313189,
                                       -6422251474445030185, -7319538979026301197, 43375819491230490,
                                       4067095951146399024, 5566762981945445426, 3380316064147053703,
                                       6326592332302779559, 8113639977376027382, -7369480799159297171,
                                       966243114147515701, -6946416253695514028, 2357046065215502907,
                                       -8234953644437983710, -7934548759483306436, 2105400710493894916,
                                       -9005388290168190670, 365645163741482283, -8280608950147955336,
                                       -3059435043961995739, -3748400235947939042, -7176388910736545336,
                                       -8144718550601439186, -3577359004419942045, 6339366923570150542,
                                       7495634606004170324, -4653490025228736624, 1801237228338093014,
                                       -77566043195538997, 723582532828161971, 3314200684287618500, -478209934713862152,
                                       -4777987683718814745, 7061213708150261499, 9186605732021141840,
                                       -729019047071794707, -8436203444188863471, -1734078300218198554,
                                       7511227023674679512, 6990818596810480936, -218336073768965596,
                                       -3413361850145131188, -7959493175363426853, 2222052874923314581,
                                       -6420779562516943667, 7515701854190147396, 3369452958378196388,
                                       -5620547966298519802, 6506602029511629209, -7098818170722380714,
                                       -8423302467637088130, 8155287111655515936, 2877204379700368458,
                                       -3876240582429451838, -9213106718258967179, -2480342612987621788};

#endif //MPC_PACKAGE_FIXED_AB_PAIRS_0_H
