import torch.nn as nn
import torchvision.models as models
import torch

class ViolenceDetector(nn.Module):
    def __init__(self):
        super(ViolenceDetector, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            x = self.cnn(x).view(B, T, -1)  # [B, T, 512]
        lstm_out, _ = self.lstm(x)
        output = self.classifier(lstm_out[:, -1])
        return torch.sigmoid(output).squeeze()
