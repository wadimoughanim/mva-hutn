from transformer import ViT
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(AutoEncoder, self).__init__()
        self.P = P
        self.L = L
        self.size = size
        self.patch = patch
        self.dim = dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim * P) // patch ** 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim * P) // patch ** 2, momentum=0.5),
        )

        # Vision Transformer
        self.vtrans = ViT(image_size=size, patch_size=patch, dim=(dim * P), depth=2, heads=8, mlp_dim=12, pool='cls')
        
        # Upscale to original size
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        
        # Smooth and apply softmax for abundance maps
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(-1, self.P, self.size * self.size)  # Adjusted for batch processing
        abu_est = self.upscale(cls_emb).view(-1, self.P, self.size, self.size)  # Adjusted for batch processing
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        return abu_est, re_result

if __name__ == "__main__":
    P = 3 
    L = 10 
    size = 32 
    patch = 4  
    dim = 512  

    model = AutoEncoder(P=P, L=L, size=size, patch=patch, dim=dim)
    
    model.apply(model.weights_init)
    mock_data = torch.randn(1, L, size, size)  

    abu_est, re_result = model(mock_data)

    print(f"Shape of abundance estimates: {abu_est.shape}")
    print(f"Shape of reconstructed image: {re_result.shape}")