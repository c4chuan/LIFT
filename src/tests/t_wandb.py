# 首先确保安装了 wandb：
# pip install wandb

import wandb
import time
import random

def train(num_epochs=5, learning_rate=0.01):
    # 初始化一个 wandb 运行，project 名称可替换为你自己的项目名
    wandb.init(
        project="my-first-wandb-project2",
        name = "asdf"
    )

    # 模拟训练过程
    for epoch in range(10):
        # 假设训练和验证损失是不断下降/波动的随机数
        train_loss = random.uniform(0.5, 1.0) * (1.0 / (epoch + 1))
        val_loss = random.uniform(0.5, 1.0) * (1.0 / (epoch + 1)) + 0.1

        # 打印到控制台（可选）
        print(f"Epoch {epoch+1}/10 — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # 使用 wandb.log 记录指标
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
        })

        # 模拟每个 epoch 的耗时
        time.sleep(0.5)

    # 训练结束后，结束当前 run
    wandb.finish()

if __name__ == "__main__":
    # 调用训练函数，可修改超参数
    train(num_epochs=10, learning_rate=0.001)
