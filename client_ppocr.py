
import json
import base64
import requests
import uuid

# ocr
img_path = 'doc/deployment_en.png'
with open(img_path, 'rb') as f:
    im_base64 = base64.b64encode(f.read()).decode()
print(len(im_base64))

guid = str(uuid.uuid4())
params = {'guid': guid, 'fname': guid, 'image': im_base64}
result = requests.post('http://127.0.0.1:19200/api/ocr', data=params)
if result.status_code == 200:
    result = result.json()
    print(result)

# result = requests.post('http://127.0.0.1:9300/api/rotate', data=params)
# if result.status_code == 200:
#     result = result.json()
#     print(result)


# # 文本识别
# from copy import deepcopy
# img_path = 'doc/PaddleOCR_log.png'
# with open(img_path, 'rb') as f:
#     im_base64 = base64.b64encode(f.read()).decode()

# imgs = [deepcopy(im_base64) for _ in range(16)]
# print(len(imgs), type(imgs[0]))
# imgs = json.dumps(imgs)

# guid = str(uuid.uuid4()).replace("-", "")
# params = {'guid': guid, 'fname': guid, 'image': imgs, 'include_pts': 'true'}
# result = requests.post('http://127.0.0.1:9300/api/textrecog', data=params)
# if result.status_code == 200:
#     result = result.json()
#     print(result)


# 文本检测
# from copy import deepcopy
# img_path = 'doc/deployment_en.png'
# with open(img_path, 'rb') as f:
#     im_base64 = base64.b64encode(f.read()).decode()

# guid = str(uuid.uuid4()).replace("-", "")
# params = {'guid': guid, 'fname': guid, 'image': im_base64, 'include_pts': 'true'}
# result = requests.post('http://127.0.0.1:19300/api/detect_cell', data=params)
# if result.status_code == 200:
#     result = result.json()
#     print(result)