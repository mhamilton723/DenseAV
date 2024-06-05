import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, HubertModel, HubertConfig
from transformers.pytorch_utils import Conv1D

class HubertAudioTransform():

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    def __call__(self, audio):
        return self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)


def copy_conv(l):
    new_l = Conv1D()


class Hubert(nn.Module):
    def __init__(self):
        super().__init__()
        model1 = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        config = model1.config
        del model1
        config.layer_norm_eps = 1e-4
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", config=config)
        self.config = dict()


    def forward(self, audio, include_cls):
        outputs = self.model(audio)
        # outputs = deepspeed.checkpointing.checkpoint(self.model, audio)

        patch_tokens = outputs.last_hidden_state.permute(0, 2, 1).unsqueeze(2)

        # return patch_tokens
        if include_cls:
            return patch_tokens, None
        else:
            return patch_tokens

    def get_last_params(self):
        return self.model.encoder.layers[-1].parameters()


if __name__ == "__main__":
    import librosa
    from shared import pca, remove_axes
    import matplotlib.pyplot as plt
    from pytorch_lightning import seed_everything

    audio, _ = librosa.load("../../samples/example.wav", sr=16000)
    audio = torch.from_numpy(audio).unsqueeze(0).to("cuda")

    model = Hubert().to("cuda")
    embeddings = model.forward(audio, include_cls=False)

    print(embeddings.shape)
    seed_everything(0)

    with torch.no_grad():
        [pca_feats], _ = pca([embeddings])
        pca_feats = torch.broadcast_to(
            pca_feats, (pca_feats.shape[0], pca_feats.shape[1], 25, pca_feats.shape[3]))
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        axes[1].imshow(pca_feats.cpu().squeeze(0).permute(1, 2, 0))
        remove_axes(axes)
        plt.tight_layout()
        plt.show()
        print("here")
