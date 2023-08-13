from collections import defaultdict
from shapely.geometry import Polygon, Point


def bbox2poly(bbox):
    if len(bbox) > 0 and isinstance(bbox, list):
        return bbox
    x0, y0, x1, y1 = poly1
    poly1 = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    poly1 = Polygon(poly1)
    return poly1


def calculate_iou(polygon1: Polygon, polygon2: Polygon):
    """
    计算两个多边形的IoU
    poly1，poly2：是由表示边界四个点的坐标形成的列表，形式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    # 计算多边形的面积
    poly1_area = polygon1.area
    poly2_area = polygon2.area
    # 计算多边形的交集区域
    inter_area = polygon1.intersection(polygon2).area
    # 计算IoU
    iou = inter_area / (poly1_area + poly2_area - inter_area)
    return iou

def contains_center_point(point: Point, polygon: Polygon):
    return polygon.contains(point)

def matched_lines_with_cell(lines, cell_boxs):
    """
    规则性匹配，规则：
        1. cell_id_to_lines
        2. 先中心点匹配，匹配到多个就选择iou最大的
        3. 没匹配上就算了，说明可能不是表格（置信度高的呢？）
    lines: [[x0, x1, y0, y1], text, prob]
    boxs: [[x0, x1, y0, y1]]
    """
    cell_id_to_lines = [[] for _ in range(len(cell_boxs))]
    for line in lines:
        bbox, text, prob = line[:3]
        center = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2]
        center = Point(center)
        poly = bbox2poly(bbox)
        # 遍历
        matched_center_cells = []
        for cell_id, cell_box in enumerate(cell_boxs):
            cell_poly = bbox2poly(cell_box)
            if contains_center_point(center, cell_poly):
                matched_center_cells.append((cell_id, cell_poly))
        # 选择iou最大的
        best_matched_cell_id = -1
        if len(matched_center_cells) == 1:
            best_matched_cell_id = matched_center_cells[0][0]
        elif len(matched_center_cells) > 1:
            best_matched_cell_id = max(matched_center_cells, key=lambda x: calculate_iou(x[1], poly))[0]
        else:
            continue
        # 添加
        if best_matched_cell_id >= 0:
            cell_id_to_lines[best_matched_cell_id].append(line)
        else:
            cell_id_to_lines.append(line)


def convert_cells_to_doc(cell_id_to_lines, im_base64, w, h):
    img_labels = []
    ind = 0
    for box, text, _ in cell_id_to_lines:
        x0, x1, y0, y1 = box
        img_labels.append({
            "box": [x0, y0, x1, y1],
            "text": text,
            "words": [{"box": box, "text": text}],
            "id": ind,
        })
        ind += 1
    return {
        "lang": "zh",
        "version": "0.1",
        "split": "predict",
        "documents": [{
            "id": "0",
            "uid": "",
            "document": img_labels,
            "img": {
                "fname": "0-place-holder.jpg",
                "image_data": im_base64,
                "width": w,
                "height": h,
            }
        }],
    }