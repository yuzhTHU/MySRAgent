# 训练模型直接预测
python ./scripts/nn_tools/train_property_predictor.py


# 先训练模型预训练，再训练模型预测
python ./scripts/nn_tools/train_foundation_model.py --exp_name pretrain # 假设预训练的模型保存在 logs/nn_tools/train_foundation_model/pretrain/checkpoint.pth 目录下
python ./scripts/nn_tools/train_property_predictor.py --reload_checkpoint logs/nn_tools/train_foundation_model/pretrain/checkpoint.pth
