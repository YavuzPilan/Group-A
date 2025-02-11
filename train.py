import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preparing import load_and_prepare_data
from model import Connect4Net
import csv

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
POLICY_LOSS_WEIGHT = 0.5
VALUE_LOSS_WEIGHT = 0.5


def preprocess_board_tensor(boards):
    """ Convert (N, 42) to (N, 3, 6, 7) """
    boards = boards.reshape(-1, 6, 7)  # Convert 42 -> (6, 7)
    boards_p1 = (boards == 1).float()  # Player 1
    boards_p2 = (boards == 2).float()  # Player 2
    boards_empty = (boards == 0).float()  # Empty spaces
    return torch.stack([boards_p1, boards_p2, boards_empty], dim=1)  # (N, 3, 6, 7)


def evaluate(model, test_loader, criterion_policy, criterion_value, epoch):
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    predictions = []

    with torch.no_grad():
        for batch_boards, batch_policies, batch_values in test_loader:
            predicted_policy, predicted_value = model(batch_boards)

            loss_policy = criterion_policy(predicted_policy, batch_policies)
            loss_value = criterion_value(predicted_value.squeeze(), batch_values.squeeze())
            loss = POLICY_LOSS_WEIGHT * loss_policy + VALUE_LOSS_WEIGHT * loss_value

            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

            # Save value predictions
            for pred, true in zip(predicted_value.squeeze().tolist(), batch_values.squeeze().tolist()):
                predictions.append([epoch, pred, true])

    # Write predictions to CSV
    with open("Data/value_predictions.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(predictions)

    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches

    print(f"Test Loss: {avg_loss:.4f}, Test Policy Loss: {avg_policy_loss:.4f}, Test Value Loss: {avg_value_loss:.4f}")
    return avg_loss


def train():
    print("Loading and preparing data...")
    train_boards, test_boards, train_policies, test_policies, train_values, test_values = load_and_prepare_data(skip=1,
                                                                                                                partition_rate=0.9)

    # Preprocess data (same as before)
    train_boards = torch.tensor(train_boards, dtype=torch.float32)
    train_boards = preprocess_board_tensor(train_boards)
    train_policies = torch.tensor(train_policies, dtype=torch.float32)
    train_values = torch.tensor(train_values, dtype=torch.float32).unsqueeze(1)

    test_boards = torch.tensor(test_boards, dtype=torch.float32)
    test_boards = preprocess_board_tensor(test_boards)
    test_policies = torch.tensor(test_policies, dtype=torch.float32)
    test_values = torch.tensor(test_values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_boards, train_policies, train_values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(test_boards, test_policies, test_values)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Connect4Net()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    print("Starting training loop...")
    with open("Data/loss_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test Loss"])  # CSV Header

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_boards, batch_policies, batch_values in train_loader:
            optimizer.zero_grad()
            predicted_policy, predicted_value = model(batch_boards)

            loss_policy = criterion_policy(predicted_policy, batch_policies)
            loss_value = criterion_value(predicted_value.squeeze(), batch_values.squeeze())
            loss = POLICY_LOSS_WEIGHT * loss_policy + VALUE_LOSS_WEIGHT * loss_value

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        test_loss = evaluate(model, test_loader, criterion_policy, criterion_value, epoch)

        print(f"Epoch {epoch + 1} completed. Avg Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save losses to CSV
        with open("Data/loss_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_epoch_loss, test_loss])

    torch.save(model.state_dict(), "connect4_model.pth")
    print("Model saved as connect4_model.pth")


if __name__ == "__main__":
    train()
