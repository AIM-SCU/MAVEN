import torch
print(torch.version.cuda)  # Should not be None
print(torch.cuda.is_available())  # Should be True if your driver is OK
