import torch
import torch.nn as nn


class DINOv2Featurizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model.eval()
        self.config = {}

    def get_cls_token(self, img):
        pass

    def forward(self, img, include_cls):
        feature_dict = self.model.forward_features(img)
        _, _, h, w = img.shape
        new_h, new_w = h // 14, w // 14
        b, _, c = feature_dict["x_norm_patchtokens"].shape
        spatial_tokens = feature_dict["x_norm_patchtokens"].permute(0, 2, 1).reshape(b, c, new_h, new_w)

        if include_cls:
            return spatial_tokens, feature_dict["x_norm_clstoken"]
        else:
            return spatial_tokens


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from shared import norm, crop_to_divisor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("../../samples/dog_man_1_crop.jpg")
    load_size = 224  # * 3
    transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])

    model = DINOv2Featurizer().cuda()

    results = model(transform(image).cuda().unsqueeze(0), include_cls=False)

    print(results.shape)
