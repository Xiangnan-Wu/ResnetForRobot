import os

import swanlab
import torch
from torch import nn, optim

from data_utils import load_dataset
from model import RobotResNetRegressor


def train(model, device, dataloader, optimizer, criterion, epoch):
    model.train()
    for iter, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            print(
                f"Epoch: {epoch}, Iteration [{iter}/{len(dataloader)}], Loss: {loss.item()}"
            )
        swanlab.log({"train_loss": loss.item()})

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    torch.save(model.state_dict(), "checkpoint/model_epoch{}.pth".format(epoch))


if __name__ == "__main__":
    lr = 1e-5
    bz = 32
    num_epoches = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = load_dataset("test", bz)

    swanlab.init(
        experiment_name="ResnetForRobot",
        description="使用Resnet验证采集数据的可行性",
        config={
            "model": "Resnet18",
            "optim": "SGD",
            "lr": lr,
            "batch_size": bz,
            "num_epochs": num_epoches,
            "device": device,
        },
    )

    criterion = nn.MSELoss()
    model = RobotResNetRegressor(
        resnet_type="resnet_18", output_dim=8, input_channels=3, dropout_rate=0.05
    )
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.01)
    for i in range(num_epoches):
        train(
            model=model,
            device=device,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=i + 1,
        )
