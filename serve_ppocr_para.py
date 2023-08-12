
# -*- coding: UTF-8 -*-
import os
import sys
import warnings
warnings.filterwarnings("ignore")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import json
import math
from flask import Flask, jsonify, request, url_for
from flask_cors import CORS
import base64
import numpy as np
import argparse
import cv2
from loguru import logger
from paddleocr import PaddleOCR,draw_ocr
from shapely.geometry import Polygon

ocr = PaddleOCR(
    use_angle_cls=False, lang='ch', 
    det_model_dir = "./inference/ch_PP-OCRv3_det_infer", 
    rec_model_dir = "./inference/ch_PP-OCRv3_rec_infer", # 没用
    det_limit_side_len=1920,det_limit_type="max",
    det_db_score_mode="slow",
    # use_gpu=False,
) # need to run only once to load model into memory

########################
## debug for one test ##
########################
# img_path = 'image/koreanImages/Resized_1654756658073.JPEG'
# result = ocr.ocr(img_path, cls=True, rec=True, det=True)
# for line in result:
#     print(line)


###########
## flask ##
###########

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'secret to test!'
GUID = 'de683f7810e84046ab5a8240a7cc0be3'  # 用于测试的guid


def bbox2poly(bbox):
    if len(bbox) > 0 and isinstance(bbox, list):
        return bbox
    x0, y0, x1, y1 = poly1
    poly1 = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return poly1


def calculate_iou(poly1, poly2):
    """
    计算两个多边形的IoU
    poly1，poly2：是由表示边界四个点的坐标形成的列表，形式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    poly1 = bbox2poly(poly1)
    poly2 = bbox2poly(poly2)
    # 创建多边形
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    # 计算多边形的面积
    poly1_area = polygon1.area
    poly2_area = polygon2.area
    # 计算多边形的交集区域
    inter_area = polygon1.intersection(polygon2).area
    # 计算IoU
    iou = inter_area / (poly1_area + poly2_area - inter_area)
    return iou


def matched_lines_with_cell(lines, cell_boxs):
    """
    规则性匹配，规则：
    「规则」
    lines: [[x0, x1, y0, y1], text, prob]
    boxs: [[x0, x1, y0, y1]]
    """


@app.route("/api/ocr", methods=["POST"])
def predict():

    try:
        img = request.form["image"]
    except:
        return jsonify({"code": 100})
    # base64解析
    img = base64.b64decode(str(img))
    image_data = np.frombuffer(img, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # print(image_data)
    # predict
    result = ocr.ocr(image_data)[0]
    res_info = []
    for line in result:
        bbox_4p, text, prob = line[0], line[1][0], line[1][1]
        # print(line)
        x0 = int(min(bbox_4p, key=lambda b4: b4[0])[0])
        x1 = int(max(bbox_4p, key=lambda b4: b4[0])[0])
        y0 = int(min(bbox_4p, key=lambda b4: b4[1])[1])
        y1 = int(max(bbox_4p, key=lambda b4: b4[1])[1])
        res_info.append([[x0, x1, y0, y1], text, prob])
    final_result = {
        "result": res_info,
        "code": 200,
    }
    return jsonify(final_result)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19300)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
