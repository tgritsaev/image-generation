import torch


max_lr, warmup_iters = 1e-3, 1000
linear_lambda = lambda iter: max_lr * (iter + 1) / warmup_iters
gamma = 0.999
exponential_lambda = lambda iter: max_lr * (gamma ** (iter + 1 - warmup_iters))
lr_lambda = lambda iter: linear_lambda(iter) if iter < warmup_iters else exponential_lambda(iter)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(None, lr_lambda)

for i in range(10000):
    lr_scheduler.step()
    print(lr_scheduler.get_last_lr()[0])
