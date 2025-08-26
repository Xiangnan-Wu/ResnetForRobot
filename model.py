from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # First Block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Second Block
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers: List[int], in_channels: int = 3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def _make_layer(self, block, channels: int, blocks: int, stride: int = 1):
        downsample = None
        # 当 stride 不为 1 或 通道数变化时，需要 downsample
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        # 第一个 block 可能需要 downsample
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion

        # 后续 block 不需要 downsample
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        # Resnet
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class RobotResNetRegressor(nn.Module):
    def __init__(
        self,
        resnet_type="resnet_18",
        output_dim: int = 8,
        input_channels=3,
        dropout_rate=0.1,
    ):
        super(RobotResNetRegressor, self).__init__()
        self.backbone = self._create_resnet(resnet_type, input_channels)
        feature_dim = self._get_feature_dim(resnet_type)

        self.position_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # xyz
        )
        # 旋转回归头 (quaternion)
        self.rotation_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # quaternion (w, x, y, z)
        )
        # 夹爪回归头
        self.gripper_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # gripper state
            nn.Sigmoid(),  # 将输出限制在[0,1]范围
        )
        self._initialize_regressor()

    def _create_resnet(self, resnet_type: str, input_channels):
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=input_channels)

    def _get_feature_dim(self, resnet_type):
        return 512

    def _initialize_regressor(self):
        for head in [self.position_head, self.rotation_head, self.gripper_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)

        position = self.position_head(features)
        rotation = self.rotation_head(features)
        gripper = self.gripper_head(features)

        rotation = F.normalize(rotation, p=2, dim=1)
        out = torch.cat([position, rotation, gripper], dim=1)
        return out

    def forward_separate(self, x: torch.Tensor) -> tuple:
        """分别返回位置、旋转和夹爪预测"""
        features = self.backbone(x)

        position = self.position_head(features)
        rotation = self.rotation_head(features)
        gripper = self.gripper_head(features)

        rotation = F.normalize(rotation, p=2, dim=1)

        return position, rotation, gripper


if __name__ == "__main__":
    model = RobotResNetRegressor(output_dim=8)

    B = 8
    channel = 3
    height, width = 224, 224
    x = torch.randn((B, channel, height, width))

    output = model(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
