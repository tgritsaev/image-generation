import torch


max_lr, warmup_iters = 1e-3, 10
optimizer = torch.optim.AdamW(torch.nn.Linear(10, 1).parameters(), max_lr)

linear_lambda = lambda iter: (iter + 1) / warmup_iters
gamma = 0.8
exponential_lambda = lambda iter: max_lr * (gamma ** (iter + 1 - warmup_iters))
lr_lambda = lambda iter: linear_lambda(iter) if iter < warmup_iters else exponential_lambda(iter)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for i in range(100):
    lr_scheduler.step()
    print(lr_scheduler.get_last_lr()[0])
