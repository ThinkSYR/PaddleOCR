
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

ocr = PaddleOCR(
    use_angle_cls=False, lang='ch', 
    det_model_dir = "./inference/ch_PP-OCRv3_det_infer", 
    rec_model_dir = "./inference/ch_PP-OCRv3_rec_infer",
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


@app.route("/api/textrecog", methods=["POST"])
def text_recognition():

    try:
        imgs = json.loads(request.form["image"])
    except:
        return jsonify({"code": 100})
    
    image_data = []
    logger.info("imgs: {}".format(len(imgs)))
    for img_ in imgs:
        # base64解析
        img_ = base64.b64decode(str(img_))
        img_ = np.frombuffer(img_, np.uint8)
        img_ = cv2.imdecode(img_, cv2.IMREAD_COLOR)
        image_data.append(img_)
    # print(image_data)
    # predict
    result = ocr.ocr(image_data, det=False, rec=True, cls=False)
    logger.info("result: {}".format(result))
    res_info = []
    for line in result:
        if len(line) == 0:
            res_info.append(["", 0])
            continue
        text, prob = line[0][0], line[0][1]
        res_info.append([text, prob])
    final_result = {
        "result": res_info,
        "code": 200,
    }
    return jsonify(final_result)



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

@app.route("/api/rotate", methods=["POST"])
def get_rotate():

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
    result = ocr.ocr(image_data, cls=True, rec=False, det=True)
    # 获取角度
    angle_list = []
    for box in result:
        # 顺时针四点双层list坐标，计算长边和短边
        box = np.int0(box)
        l1  = (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        l2  = (np.linalg.norm(box[1] - box[2]) + np.linalg.norm(box[3] - box[0])) / 2
        # 长宽比
        if l1 > l2:
            prop = l1 / l2
        else:
            prop = l2 / l1
            box[[0, 1, 2, 3], :] = box[[1, 2, 3, 0], :] # 错一位，换长边和短边
        if prop > 2:
            # 长边中点
            center0 = [(box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2]
            center1 = [(box[2][0] + box[3][0])/2, (box[2][1] + box[3][1])/2]
            # math.degrees输出范围-180~180
            degree  = 90 - math.degrees(math.atan2(center1[1] - center0[1], center1[0] - center0[0]))
            degree  = degree if degree <= 90 else degree - 180
            # print(degree)
            angle_list.append(degree)
    if len(angle_list) == 0:
        angle = 0
    else:
        angle_list = np.array(angle_list)
        angle_abs  = np.abs(angle_list)
        # 过滤，在一定区间内计算角度的分布，不能太分散
        range_list  = [[0, 15], [15, 30], [30, 45], [45, 60], [60, 75], [75, 90]]
        angle_count = [0, 0, 0, 0, 0, 0]
        for i, range_se in enumerate(range_list):
            angle_count[i] = np.sum(((angle_abs >= range_se[0]) & (angle_abs <= range_se[1])))
        # 集中
        angle_range = range_list[np.argmax(angle_count)]
        # print(angle_count, angle_range)
        angle_list = angle_list[(angle_abs >= angle_range[0]) & (angle_abs <= angle_range[1])]
        # 先判断角度的正负，不能有正有负会影响结果，以多数为准
        if (angle_list >= 0).sum() > (angle_list < 0).sum():
            angle = np.mean(np.abs(angle_list))
        else:
            angle = -np.mean(np.abs(angle_list))
    print(angle)

    final_result = {
        "degree": int(-angle), # 负方向输出
        "code": 200,
    }
    return jsonify(final_result)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
