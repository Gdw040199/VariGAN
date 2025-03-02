import torch
import torch.nn as nn


#########################################################################
############################  models.py  ###################################


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
        self.gamma = nn.Parameter(torch.ones(1)*0.05) #this can be initialized to 0 or a small positive value

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
        super().__init__()
        self.in_features = in_features

        # The convolution layer that defines the branch
        self.conv_Q = nn.Conv2d(in_features, in_features // 4, 1)  # Conv1(Q)
        self.conv_K = nn.Conv2d(in_features, in_features // 4, 1)  # Conv1(K)
        self.conv_V = nn.Conv2d(in_features, in_features // 4, 1)  # Conv1(V)

        # Define the concatenated channel adjustment layer
        self.conv_adapt = nn.Conv2d(3 * (in_features // 4), in_features // 4, 1)  # 调整通道数

        # Define the final 3x3 convolution layer
        self.conv_final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features // 2, in_features, 3),  # 输出通道恢复为 in_features
            nn.InstanceNorm2d(in_features)
        )

        # 3x3 convolution of the original residual path
        self.conv_original = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        # Original Res path
        h_prev = self.conv_original(x)

        # New path in the formula
        Q = self.conv_Q(x)  # [B, C/4, H, W]
        K = self.conv_K(x)  # [B, C/4, H, W]
        V = self.conv_V(x)  # [B, C/4, H, W]

        # Generate K^T (spatial dimension transpose)
        K_T = K.permute(0, 1, 3, 2)  # [B, C/4, W, H]

        # Concatenate Q, K, K^T
        concat_KKT = torch.cat([K, K_T], dim=1)  # [B, C/2, H, W]
        concat_Q_KKT = torch.cat([Q, concat_KKT], dim=1)  # [B, 3C/4, H, W]
        adapted = self.conv_adapt(concat_Q_KKT)  # [B, C/4, H, W]
        concat_V_adapted = torch.cat([V, adapted], dim=1)  # [B, C/2, H, W]

        # Final 3x3 convolution
        h_new = self.conv_final(concat_V_adapted)

        # Return the sum of the original path and the new path
        return x + h_prev + h_new  # Residual connection


##############################
## Generator Network (GeneratorResNet)
##############################
class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()

        channels = input_shape[0]
        out_features = 64

        # initial layer
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features

        # downsample part
        self.down1 = nn.Sequential(
            nn.Conv2d(in_features, out_features*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features*2),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(out_features*2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_features*2, out_features*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features*4),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(out_features*4),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(out_features*4, out_features*8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features*8),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(out_features*8),
        )
        in_features = out_features * 8

        # Resnet blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_features) for _ in range(num_residual_blocks)]
        )

        # Up-sampling part (key modification point)
        # ---------------------------
        # Modification note: Add channel compression convolution after each upper sampling layer to process the feature map after concat
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, in_features//2, 3, padding=1),      # 512 -> 256
            nn.InstanceNorm2d(in_features//2),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(in_features//2),
        )
        # Added channel compression layer (feature after processing concat)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),  # deal with the concatenated 512 channels
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),        # 256 -> 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(128),
        )
        self.up2_conv = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),  # deal with the concatenated 256 channels
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),         # 128 -> 64
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            EfficientSelfAttention(64),
        )
        self.up3_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),     # deal with the concatenated 128 channels
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # ---------------------------

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(64, channels, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        # Initial layer
        x_initial = self.initial(x)  # [B, 64, H, W]

        # Downsample
        d1 = self.down1(x_initial)   # [B, 128, H/2, W/2]
        d2 = self.down2(d1)          # [B, 256, H/4, W/4]
        d3 = self.down3(d2)          # [B, 512, H/8, W/8]

        # Residual blocks
        res = self.res_blocks(d3)    # [B, 512, H/8, W/8]

        # Upsampling (key modification point)
        # ---------------------------
        # First level upsampling
        u1 = self.up1(res)           # [B, 256, H/4, W/4]
        u1 = torch.cat([u1, d2], dim=1)  #  [B, 256+256=512, H/4, W/4]
        u1 = self.up1_conv(u1)       #  [B, 256, H/4, W/4]

        # Second level upsampling
        u2 = self.up2(u1)            # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, d1], dim=1)  #  [B, 128+128=256, H/2, W/2]
        u2 = self.up2_conv(u2)       # [B, 128, H/2, W/2]

        # Third level upsampling
        u3 = self.up3(u2)            # [B, 64, H, W]
        u3 = torch.cat([u3, x_initial], dim=1)  # [B, 64+64=128, H, W]
        u3 = self.up3_conv(u3)       #  [B, 64, H, W]
        # ---------------------------

        # Output layer
        output = self.output(u3)
        return output


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