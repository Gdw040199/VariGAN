# Description: This script is used to test the number of trainable parameters in the model

from models import Generator, Discriminator


model = Generator([3, 256, 256],9)
#model = Discriminator([3, 256, 256])

# get the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"trainable para: {trainable_params:,}")
print(f"total para: {total_params:,}")
print(f"forzen para: {total_params - trainable_params:,}")

# M for million
print(f"trainable para: {trainable_params/1e6:.2f}M")
print(f"total para: {total_params/1e6:.2f}M")