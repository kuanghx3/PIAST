import numpy as np
import pandas as pd
import myfunctions as fn
from args import args, dev
import torch
from torch.utils.data import DataLoader
import proposed_model as model


adj = pd.read_csv(args.data_path + 'SZ247_adj.csv', index_col=0, header=0)
adj_dense = np.array(adj, dtype=float)
nodes = adj.shape[0]
adj_dense = torch.Tensor(adj_dense)
adj = adj_dense.to_sparse_coo().to(dev)

fn.seed_torch(2023)
occ, prc = fn.get_data(args)  # (t, nodes)
occ_train, occ_test = fn.division(args, occ)
prc_train, prc_test = fn.division(args, prc)
train_dataset = fn.MyDataset(args, occ_train, prc_train, dev)
test_dataset = fn.MyDataset(args, occ_test, prc_test, dev)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True)

model_proposed = model.PhysicsInformedNN(args, adj)
model_proposed.train(train_loader,args)
# torch.save(model_proposed, "./result_"+str(args.predict_time)+"/proposed_pinn.pt")
# result = model_proposed.predict(test_loader,args)
lambda_1_value = model_proposed.lambda_1.detach().cpu().numpy()
pd.DataFrame(data=lambda_1_value).to_csv("./result_"+str(args.predict_time)+"/lamda_1.csv")
pd.DataFrame(data=result).to_csv("./result_"+str(args.predict_time)+"/proposed_pinn.csv")
# result = model.test(model_proposed, test_loader, args)
