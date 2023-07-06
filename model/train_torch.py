# from channelwise_GRU_torch import *
# from SR_SAN_torch import *
from channelwise_GRU_torch_two import *

embedding_dim = 384
hidden_units = 768
max_len = 10
save_feq = 40
epochs = 200
batch_size = 512
lang = "es"

ob = Metesre(embedding_dim=embedding_dim, hidden_units=hidden_units, max_len=max_len, lang=lang)

ob.model.load_state_dict(torch.load("model/model.pth"))
ob.test()

# ob.train(epoch=epochs, batch_size=batch_size, save_feq=save_feq)

