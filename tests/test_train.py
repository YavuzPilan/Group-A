import torch
import torch.nn as nn
import torch.optim as optim
from train import load_and_prepare_data, DEVICE, BATCH_SIZE, LEARNING_RATE
from model import Connect4Net
import os

# Global variables for tests
train_boards_tensor = None
train_policies_tensor = None
train_values_tensor = None
test_boards_tensor = None
test_policies_tensor = None
test_values_tensor = None
model = None
optimizer = None

def setup_module():
    """Setup function to load and preprocess data before running tests."""
    global train_boards_tensor, train_policies_tensor, train_values_tensor
    global test_boards_tensor, test_policies_tensor, test_values_tensor
    global model, optimizer

    # Load data
    train_boards, test_boards, train_policies, test_policies, train_values, test_values = load_and_prepare_data(skip=1, partition_rate=0.9)

    # Convert data to PyTorch tensors with correct channel dimensions
    train_boards_tensor = torch.tensor(train_boards, dtype=torch.float32).unsqueeze(1).repeat(1, 2, 1, 1).to(DEVICE)  # (batch, 2, 6, 7)
    train_policies_tensor = torch.tensor(train_policies, dtype=torch.float32).to(DEVICE)
    train_values_tensor = torch.tensor(train_values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    test_boards_tensor = torch.tensor(test_boards, dtype=torch.float32).unsqueeze(1).repeat(1, 2, 1, 1).to(DEVICE)  # (batch, 2, 6, 7)
    test_policies_tensor = torch.tensor(test_policies, dtype=torch.float32).to(DEVICE)
    test_values_tensor = torch.tensor(test_values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Initialize model and optimizer
    model = Connect4Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def teardown_module():
    """Teardown function to free memory after tests."""
    global train_boards_tensor, train_policies_tensor, train_values_tensor
    global test_boards_tensor, test_policies_tensor, test_values_tensor
    global model, optimizer

    del train_boards_tensor, train_policies_tensor, train_values_tensor
    del test_boards_tensor, test_policies_tensor, test_values_tensor
    del model, optimizer

def test_model_initialization():
    """Test if the Connect4Net model initializes correctly."""
    assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"

def test_forward_pass():
    """Test if the model produces outputs of correct shape."""
    batch_boards = train_boards_tensor[:BATCH_SIZE]  # Shape should now be (BATCH_SIZE, 2, 6, 7)

    predicted_policies, predicted_values = model(batch_boards)

    assert predicted_policies.shape == (BATCH_SIZE, 7), "Policy output should have shape (BATCH_SIZE, 7)"
    assert predicted_values.shape == (BATCH_SIZE, 1), "Value output should have shape (BATCH_SIZE, 1)"

def test_training_step():
    """Test if the optimizer updates model parameters during training."""
    model.train()
    optimizer.zero_grad()

    batch_boards = train_boards_tensor[:BATCH_SIZE]  # (BATCH_SIZE, 2, 6, 7)
    batch_policies = train_policies_tensor[:BATCH_SIZE]
    batch_values = train_values_tensor[:BATCH_SIZE]

    predicted_policies, predicted_values = model(batch_boards)

    # Compute losses
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    policy_loss = policy_loss_fn(predicted_policies, batch_policies.argmax(dim=1))
    value_loss = value_loss_fn(predicted_values, batch_values)

    total_loss = policy_loss + value_loss
    total_loss.backward()
    optimizer.step()

    assert total_loss.item() >= 0, "Total loss should be non-negative"

def test_model_saving():
    """Test if the model saves correctly."""
    save_path = "test_model.pth"
    torch.save(model.state_dict(), save_path)

    assert os.path.exists(save_path), "Model file was not saved"

    # Clean up test model file
    os.remove(save_path)
