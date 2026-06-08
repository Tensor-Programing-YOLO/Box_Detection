import os
import shutil
import random
from pathlib import Path


def preprocess_dataset():
    # 1. 경로 설정
    raw_images_dir = Path("raw_dataset/images")
    raw_labels_dir = Path("raw_dataset/labels")
    
    target_base = Path("dataset")
    dirs = {
        "train_img": target_base / "images/train",
        "val_img": target_base / "images/val",
        "train_lbl": target_base / "labels/train",
        "val_lbl": target_base / "labels/val",
    }

    # 타겟 디렉토리 생성
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 2. 수동 경계 설정 (새로운 상자가 시작되는 사진 번호)
    # 이 번호들을 기준으로 사진들이 그룹화되어 Train/Val로 나뉩니다.
    boundaries = [
    1, 50, 100, 150, 200, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 702, 751, 801, 850, 900, 929,
    931, 932, 935, 939, 940, 943, 945, 946, 952, 954,
    957, 958, 961, 963, 966, 969, 970, 972, 976, 979,
    980, 983, 984, 985, 986, 987, 988, 989, 990, 991,
    992, 993, 996, 997, 1001, 1004, 1005, 1007, 1008, 1009,
    1010, 1011, 1012, 1013, 1016, 1019, 1024, 1029, 1030, 1031,
    1032, 1033, 1034, 1035, 1037, 1040, 1043, 1045, 1047, 1050,
    1054, 1056, 1058, 1075, 1086, 1094, 1103, 1110, 1111, 1122,
    1146, 1152, 1155, 1158, 1160, 1163, 1166, 1175, 1178, 1181,
    1185, 1194, 1200, 1201, 1205, 1208, 1210, 1211, 1212, 1216,
    1219, 1223, 1226, 1227, 1228, 1231, 1232, 1235, 1236, 1237,
    1253, 1260, 1261, 1262, 1263, 1264, 1265, 1268, 1270, 1271,
    1272, 1274, 1279, 1280, 1281, 1282, 1283, 1284, 1287, 1288,
    1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1299,
    1300, 1301, 1302, 1303, 1304, 1305, 1307, 1309, 1310, 1311,
    1313, 1314, 1315, 1316, 1320, 1321, 1322, 1323, 1324, 1326,
    1327, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1339, 1340,
    1341, 1342, 1343, 1345, 1347, 1348, 1349, 1350, 1351, 1352,
    1353, 1354, 1355, 1356, 1357, 1358, 1359, 1362, 1363, 1365,
    1366, 1367, 1368, 1370, 1371, 1372, 1374, 1375, 1376, 1377,
    1383, 1386, 1387, 1388, 1389, 1390, 1391, 1393, 1394, 1395,
    1396, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1407, 1409,
    1410, 1411, 1412, 1413, 1415, 1416, 1417, 1418, 1419, 1420,
    1422, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432,
    1433, 1434, 1435, 1436, 1438, 1439, 1440, 1441, 1451, 1452, 
    1453, 1455, 1458, 1460, 1461, 1463, 1465, 1466, 1467, 1468, 
    1471, 1473, 1477, 1479, 1481, 1482, 1483, 1484, 1486, 1487,
    1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 
    1499, 1501, 1502, 1505, 1506, 1508, 1509, 1510, 1511, 1515,
    1517, 1518, 1519, 1520, 1521, 1522, 1523, 1526, 1527, 1528, 
    1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1538, 1539, 
    1542, 1547, 1548, 1549, 1550, 1552, 1553, 1554, 1556, 1558, 
    1559, 1560, 1561,1562, 1563, 1564, 1565, 1567, 1568, 1569, 
    1570, 1571, 1572, 1573, 1574, 1576, 1577, 1581, 1582, 1583, 
    1584, 1585, 1586, 1588, 1589, 1590, 1591, 1592, 1593, 
    1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 
    1604, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 
    1615, 1616, 1617, 1618, 1620, 1621, 1625,  1626, 1628, 1630,
    1631, 1632, 1636, 1637, 1638, 1641, 1643, 1644, 1646, 1647, 
    1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 
    1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667
    
    #1439, 1440, 1441, 1442, 1452,
    #1453, 1456, 1457, 1460, 1464, 1466, 1468, 1470, 1472, 1476,
    #1478, 1480, 1482, 1485, 1488, 1492, 1494, 1499, 1500, 1501,
    #1503, 1504, 1506, 1507, 1510, 1512, 1514, 1515, 1516, 1517,
    #1518, 1519, 1521, 1522, 1523, 1525, 1526, 1527, 1529, 1530,
    #1531, 1532, 1535, 1536, 1539, 1540, 1545, 1547, 1549, 1550,
    #1551, 1553, 1554, 1556, 1557, 1558, 1562, 1563, 1565, 1566,
    #1585, 1586, 1587, 1588, 1589, 1590, 1592, 1594, 1596, 1597,
    #1598, 1599, 1600, 1601, 1602, 1605, 1608, 1610, 1611, 1612,
    #1613, 1615, 1618, 1619, 1620, 1622, 1626, 1627, 1629, 1631,
    #1634, 1635, 1638, 1640, 1642, 1643, 1644, 1646, 1647, 1651,
    #1653, 1654, 1656, 1657, 1658, 1660, 1661, 1662, 1664, 1665,
    #1666, 1670
]
    

    boundary_set = set(boundaries)

    # 3. 데이터 그룹화
    image_files = sorted([f for f in raw_images_dir.glob("*.jpg")], key=lambda x: int(x.stem))
    groups = {}
    current_group_id = 0

    for img_path in image_files:
        try:
            # 파일명에서 숫자 추출 (예: 00001.jpg -> 1)
            file_num = int(img_path.stem)
        except ValueError:
            continue

        if file_num in boundary_set:
            current_group_id += 1
        
        if current_group_id not in groups:
            groups[current_group_id] = []
        groups[current_group_id].append(img_path)

    # 4. 상자 그룹 단위 셔플 및 분할 (8:2)
    group_ids = list(groups.keys())
    random.seed(40) # 재현성을 위해 시드 고정
    random.shuffle(group_ids)

    split_idx = int(len(group_ids) * 0.8)
    train_groups = group_ids[:split_idx]
    val_groups = group_ids[split_idx:]

    def copy_files(selected_groups, img_target, lbl_target):
        count = 0
        discard_count = 0
        ignore_class_id = 5  # 'contamination' 클래스 ID
        box_class_id = 0     # 'box' 클래스 ID
        
        for g_id in selected_groups:
            for img_path in groups[g_id]:
                lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
                
                # 조건 검사: contamination 포함 여부 및 box 개수 확인
                has_contamination = False
                box_count = 0
                
                if lbl_path.exists():
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            
                            class_id = int(parts[0])
                            
                            # 1. contamination 클래스 포함 여부 확인
                            if class_id == ignore_class_id:
                                has_contamination = True
                                break
                            
                            # 2. box 클래스 개수 카운트
                            if class_id == box_class_id:
                                box_count += 1
                
                # 필터링 조건: contamination이 있거나 box가 2개 이상이면 제외
                if has_contamination or box_count >= 2:
                    discard_count += 1
                    continue

                # 이미지 복사
                shutil.copy(img_path, img_target / img_path.name)
                
                # 라벨 복사
                if lbl_path.exists():
                    shutil.copy(lbl_path, lbl_target / lbl_path.name)
                
                count += 1
        return count, discard_count

    # 5. 파일 복사 실행
    train_count, train_discard = copy_files(train_groups, dirs["train_img"], dirs["train_lbl"])
    val_count, val_discard = copy_files(val_groups, dirs["val_img"], dirs["val_lbl"])

    # 결과 출력
    print(f"총 상자 그룹 수: {len(group_ids)}")
    print(f"Train 이미지 수: {train_count} (버려진 사진: {train_discard})")
    print(f"Val 이미지 수: {val_count} (버려진 사진: {val_discard})")
    print(f"총 제외된 contamination 사진 수: {train_discard + val_discard}")

if __name__ == "__main__":
    preprocess_dataset()
