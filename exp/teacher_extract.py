import paddle
# 加载预训练模型
all_params = paddle.load("output/ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 模型的权重提取
s_params = {key[len("Teacher."):]: all_params[key] for key in all_params if "Teacher." in key}
# 查看模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "./output/ch_PP-OCRv3_det_distill_train/teacher.pdparams")