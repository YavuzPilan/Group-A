import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1)  # 3 input channels
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 5 * 6, 7)  # Adjusted input size

        # Value head
        self.value_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * 5 * 6, 64)  # Adjusted input size
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Debugging: Print input shape

        # Feature extraction
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")  # Debugging: Print shape after conv1
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")  # Debugging: Print shape after conv2
        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")  # Debugging: Print shape after conv3

        # Policy head
        policy = F.relu(self.policy_conv(x))
        # print(f"Policy after conv: {policy.shape}")  # Debugging: Print shape after policy_conv
        policy = policy.view(policy.size(0), -1)  # Flatten correctly
        # print(f"Policy after flattening: {policy.shape}")  # Debugging: Print shape after flattening
        policy = F.softmax(self.policy_fc(policy), dim=1)
        # print(f"Policy after fc: {policy.shape}")  # Debugging: Print shape after policy_fc

        # Value head
        value = F.relu(self.value_conv(x))
        # print(f"Value after conv: {value.shape}")  # Debugging: Print shape after value_conv
        value = value.view(value.size(0), -1)  # Flatten correctly
        # print(f"Value after flattening: {value.shape}")  # Debugging: Print shape after flattening
        value = F.relu(self.value_fc1(value))
        # print(f"Value after fc1: {value.shape}")  # Debugging: Print shape after value_fc1
        value = torch.tanh(self.value_fc2(value))  # Remove scaling factor (0.5) for simplicity
        # print(f"Value after fc2: {value.shape}")  # Debugging: Print shape after value_fc2

        return policy, value


if __name__ == "__main__":
    # Example input for debugging
    model = Connect4Net()
    sample_input = torch.randn(8, 3, 6, 7, dtype=torch.float32)  # Corrected shape and dtype
    # print(f"Sample input shape: {sample_input.shape}")  # Debugging: Print sample input shape
    policy, value = model(sample_input)
    # print("Policy shape:", policy.shape)  # Expected: (8, 7)
    # print("Value shape:", value.shape)    # Expected: (8, 1)
