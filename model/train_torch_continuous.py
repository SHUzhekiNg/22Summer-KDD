from channelwise_GRU_continuous_torch import *

embedding_dim = 128
hidden_units = 512
max_len = 10
epochs = 20
batch_size = 32

ob = Metesre(embedding_dim=embedding_dim, hidden_units=hidden_units, max_len=max_len)

ob.model.load_state_dict(torch.load("model.pth"))
ob.test()

# ob.train(epoch=epochs)

