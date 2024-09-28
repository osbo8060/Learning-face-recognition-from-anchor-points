import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Shows the current GPU device ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Shows the GPU name