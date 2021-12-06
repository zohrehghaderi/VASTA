from torch import nn
import torch
from model.Swin import SwinTransformer3D
import copy
class swin_encoder(nn.Module):
    def __init__(self , device , drop , checkpoint_encoder):
        super().__init__()
        checkpoint = checkpoint_encoder
        self.device=device
        self.label= 'demo/label_map_k400.txt'

        self.model=SwinTransformer3D(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=(8, 7, 7),
            patch_size=(2, 4, 4),
            drop_path_rate=0.1,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=drop,
            attn_drop_rate=drop,

            patch_norm=True)

        checkpoint = torch.load(checkpoint, map_location=device)
        state_dict2 = copy.deepcopy(checkpoint['state_dict'])
        state_dict = dict()
        for key in checkpoint['state_dict']:
            if 'backbone.' in key:
                new_key = key.replace('backbone.', '')
                state_dict[new_key] = state_dict2.pop(key)

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.max_testing_views = None



    def forward(self, imgs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        feat = self.model.forward(imgs)

        # perform spatio-temporal pooling
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        feat = avg_pool(feat)
        # squeeze dimensions
        feat = feat.view(batches, feat.shape[1], feat.shape[2])
        feat = feat.permute(0, 2, 1)
        return feat





