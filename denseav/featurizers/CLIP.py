import clip
import torch
from torch import nn


class CLIPFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        self.model.eval().cuda()
        self.config = {}

    def get_cls_token(self, img):
        return self.model.encode_image(img).to(torch.float32)

    def forward(self, img, include_cls):
        features = self.model.get_visual_features(img, include_cls)
        new_features = []
        for i in range(2):
            t = features[i]
            if isinstance(t, torch.Tensor):
                new_features.append(t.to(torch.float32))
            else:
                new_features.append(t)

        return new_features


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    from shared import norm, crop_to_divisor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open("../samples/lex1.jpg")
    load_size = 224  # * 3
    transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        # T.CenterCrop(load_size),
        T.ToTensor(),
        lambda x: crop_to_divisor(x, 16),
        norm])

    model = CLIPFeaturizer().cuda()

    results = model(transform(image).cuda().unsqueeze(0))

    print(clip.available_models())
