from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from channelwise_GRU_torch import *

embedding_dim = 128
hidden_units = 48
max_len = 10
epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test, vocab_sizes = preprocessing(max_len=max_len)
print(type(X_train), type(Y_train), type(X_test), type(Y_test))
model = Model(vocab_sizes, embedding_dim, hidden_units, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
print(type(X_train[0]))


dataset = TensorDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
model.to(device)

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, target in dataloader:
        inputs = [x.to(device) for x in inputs]
        target = target.to(device)

        optimizer.zero_grad()

        outputs = model(*inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

print('Training finished.')
# Print the model summary
print(model)