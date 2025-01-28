import torch
import torch.nn as nn


#########################################################################
############################  models.py  ###################################

## Define the parameter initialization function
def weights_init_normal(m):
    classname = m.__class__.__name__  ## m is a formal parameter that can theoretically pass many things. To enable multiple arguments, each module must provide its own name. This line returns the name of m.
    if classname.find("Conv") != -1:  ## find(): Checks if "Conv" is in classname. Returns -1 if not found; otherwise, returns 0.
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  ## m.weight.data represents the weights to be initialized. nn.init.normal_(): Random initialization using a normal distribution with mean 0 and standard deviation 0.02.
        if hasattr(m, "bias") and m.bias is not None:  ## hasattr(): Checks if m contains the attribute "bias" and if it is not empty.
            torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_(): Initializes the bias as a constant 0.
    elif classname.find("BatchNorm2d") != -1:  ## find(): Checks if "BatchNorm2d" is in classname. Returns -1 if not found; otherwise, returns 0.
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)  ## m.weight.data represents the weights to be initialized. nn.init.normal_(): Random initialization using a normal distribution with mean 1.0 and standard deviation 0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_(): Initializes the bias as a constant 0.

##############################
## Self-Attention Mechanism
##############################
class EfficientSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):  # Added reduction_ratio
        super(EfficientSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Define query, key, and value convolutional layers
        self.query = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # Downsample the feature map
        x_reduced = nn.functional.interpolate(x, scale_factor=1/self.reduction_ratio, mode='nearest')
        _, _, reduced_height, reduced_width = x_reduced.size()

        # Compute query, key, and value
        query = self.query(x_reduced).view(batch_size, -1, reduced_height * reduced_width).permute(0, 2, 1)  # (B, H'*W', C//r)
        key = self.key(x_reduced).view(batch_size, -1, reduced_height * reduced_width)  # (B, C//r, H'*W')
        value = self.value(x_reduced).view(batch_size, -1, reduced_height * reduced_width)  # (B, C, H'*W')

        # Compute attention scores
        attention = torch.bmm(query, key)  # (B, H'*W', H'*W')
        attention = torch.softmax(attention, dim=-1)  # Normalize along rows

        # Weighted sum
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H'*W')
        out = out.view(batch_size, C, reduced_height, reduced_width)

        # Upsample back to original size
        out = nn.functional.interpolate(out, size=(height, width), mode='nearest')

        # Add residual connection
        out = self.gamma * out + x  # Learnable scaling parameter gamma
        return out


##############################
## Residual Block
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(  ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),  ## ReflectionPad2d(): Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),  ## Convolution
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d(): Normalizes over HxW for each image, used in style transfer
            nn.ReLU(inplace=True),  ## Non-linear activation
            nn.ReflectionPad2d(1),  ## ReflectionPad2d(): Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),  ## Convolution
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d(): Normalizes over HxW for each image, used in style transfer
        )

    def forward(self, x):  ## Input is an image
        return x + self.block(x)  ## Output is the image plus the residual output of the network


##############################
## Generator Network (GeneratorResNet)
##############################
class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()

        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling part
        for _ in range(3):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                EfficientSelfAttention(out_features),  # Introduce self-attention mechanism
            ]
            in_features = out_features

        # Residual blocks part
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling part
        for _ in range(3):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                EfficientSelfAttention(out_features),  # Introduce self-attention mechanism
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape  ## input_shape: (3, 256, 256)

        # Output shape is (1, 16, 16)
        self.output_shape = (1, 16, 16)

        def depthwise_separable_conv(in_filters, out_filters, stride=1, padding=1):
            """Depthwise separable convolution block"""
            layers = [
                # Depthwise convolution
                nn.Conv2d(in_filters, in_filters, kernel_size=3, stride=stride, padding=padding, groups=in_filters, bias=False),
                nn.InstanceNorm2d(in_filters),
                nn.LeakyReLU(0.2, inplace=True),
                # Pointwise convolution
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return layers

        self.model = nn.Sequential(
            # First layer: Regular convolution
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer: Depthwise separable convolution
            *depthwise_separable_conv(64, 128, stride=2, padding=1),

            # Third layer: Depthwise separable convolution
            *depthwise_separable_conv(128, 256, stride=2, padding=1),

            # Fourth layer: Depthwise separable convolution
            *depthwise_separable_conv(256, 512, stride=2, padding=1),

            # Fifth layer: Depthwise separable convolution (maintain spatial dimensions)
            *depthwise_separable_conv(512, 512, stride=1, padding=1),

            # Sixth layer: Depthwise separable convolution (maintain spatial dimensions)
            *depthwise_separable_conv(512, 512, stride=1, padding=1),

            # Final layer: Regular convolution, output channels = 1, output size = (1, 16, 16)
            nn.Conv2d(512, 1, kernel_size=16, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Use Sigmoid to map output to [0, 1] range
        )

    def forward(self, img):
        return self.model(img)