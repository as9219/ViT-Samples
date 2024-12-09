from torch import nn, randn, cat

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = embed_dim,
                kernel_size = patch_size,
                stride = patch_size
            ),
            nn.Flatten(2)
        )
        
        self.cls_token = nn.Parameter(randn(size=(1, in_channels, embed_dim)), requires_grad = True)
        self.position_embeddings = nn.Parameter(randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) #-1 means that the dimension is not mutable

        x = self.patcher(x).permute(0, 2, 1)
        x = cat([cls_token, x], dim=1)

        x = self.position_embeddings + x # dont need this if .to(device) is enabled in sample tensor in ViT class????
        x = self.dropout(x)
        
        return x