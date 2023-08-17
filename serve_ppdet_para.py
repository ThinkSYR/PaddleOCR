
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

"""
这里修改了deploy/pdserving/ocr_reader.py，默认最小边长960
"""
ocr = PaddleOCR(
    use_angle_cls=False, lang='ch', 
    det_model_dir = "./inference/det_ft_para", 
    rec_model_dir = "./inference/ch_PP-OCRv3_rec_infer", # 没用
    det_limit_side_len=1920,det_limit_type="max",
    det_db_score_mode="slow",
    det_db_box_thresh=0.5,det_algorithm="DB++", # db++就这点不一样好像
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


@app.route("/api/detect_cell", methods=["POST"])
def detect():

    try:
        img = request.form["image_path"]
    except:
        return jsonify({"code": 100})
    # # base64解析
    # img = base64.b64decode(str(img))
    # image_data = np.frombuffer(img, np.uint8)
    # image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # print(image_data)
    # predict
    result = ocr.ocr(img, det=True, rec=False, cls=False)[0]
    res_info = [] # n*4*2
    for line in result:
        bbox_4p = line
        res_info.append(bbox_4p) # 4*2
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
