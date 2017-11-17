from chapter02.mynetwork import MyNetwork
model = MyNetwork()


import torch
# 保存模型的全部信息
torch.sava(model, 'model_path.ptm')

# 保存模型的参数信息
torch.save(model.state_dict(), 'model_state_path.ptm')

# 加载整个模型
load_model = torch.load('model_path.ptm')

# 加载模型的参数信息
model.load_state_dict('model_state_path.ptm')