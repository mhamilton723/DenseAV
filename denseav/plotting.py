import os
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import torchvision
from moviepy.editor import VideoFileClip, AudioFileClip
from IPython.display import HTML, display
from base64 import b64encode
from denseav.shared import pca


def write_video_with_audio(video_frames, audio_array, video_fps, audio_fps, output_path):
    """
    Writes video frames and audio to a specified path.

    Parameters:
    - video_frames: torch.Tensor of shape (num_frames, height, width, channels)
    - audio_array: torch.Tensor of shape (num_samples, num_channels)
    - video_fps: int, frames per second of the video
    - audio_fps: int, sample rate of the audio
    - output_path: str, path to save the final video with audio
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    temp_audio_path = output_path.replace('.mp4', '_temp_audio.wav')
    video_options = {
        'crf': '23',
        'preset': 'slow',
        'bit_rate': '1000k'}

    if audio_array is not None:
        torchvision.io.write_video(
            filename=temp_video_path,
            video_array=video_frames,
            fps=video_fps,
            options=video_options
        )

        wavfile.write(temp_audio_path, audio_fps, audio_array.cpu().to(torch.float64).permute(1, 0).numpy())
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(temp_audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', verbose=False)
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    else:
        torchvision.io.write_video(
            filename=output_path,
            video_array=video_frames,
            fps=video_fps,
            options=video_options
        )


def alpha_blend_layers(layers):
    blended_image = layers[0]
    for layer in layers[1:]:
        rgb1, alpha1 = blended_image[:, :3, :, :], blended_image[:, 3:4, :, :]
        rgb2, alpha2 = layer[:, :3, :, :], layer[:, 3:4, :, :]
        alpha_out = alpha2 + alpha1 * (1 - alpha2)
        rgb_out = (rgb2 * alpha2 + rgb1 * alpha1 * (1 - alpha2)) / alpha_out.clamp(min=1e-7)
        blended_image = torch.cat([rgb_out, alpha_out], dim=1)
    return (blended_image[:, :3] * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def _prep_sims_for_plotting(sim_by_head, frames):
    with torch.no_grad():
        results = defaultdict(list)
        n_frames, _, vh, vw = frames.shape

        sims = sim_by_head.max(dim=1).values

        n_audio_feats = sims.shape[-1]
        for frame_num in range(n_frames):
            selected_audio_feat = int((frame_num / n_frames) * n_audio_feats)

            selected_sim = F.interpolate(
                sims[frame_num, :, :, selected_audio_feat].unsqueeze(0).unsqueeze(0),
                size=(vh, vw),
                mode="bicubic")

            results["sims_all"].append(selected_sim)

            for head in range(sim_by_head.shape[1]):
                selected_sim = F.interpolate(
                    sim_by_head[frame_num, head, :, :, selected_audio_feat].unsqueeze(0).unsqueeze(0),
                    size=(vh, vw),
                    mode="bicubic")
                results[f"sims_{head + 1}"].append(selected_sim)

        results = {k: torch.cat(v, dim=0) for k, v in results.items()}
        return results


def get_plasma_with_alpha():
    plasma = plt.cm.plasma(np.linspace(0, 1, 256))
    alphas = np.linspace(0, 1, 256)
    plasma_with_alpha = np.zeros((256, 4))
    plasma_with_alpha[:, 0:3] = plasma[:, 0:3]
    plasma_with_alpha[:, 3] = alphas
    return mcolors.ListedColormap(plasma_with_alpha)


def get_inferno_with_alpha_2(alpha=0.5, k=30):
    k_fraction = k / 100.0
    custom_cmap = np.zeros((256, 4))
    threshold_index = int(k_fraction * 256)
    custom_cmap[:threshold_index, :3] = 0  # RGB values for black
    custom_cmap[:threshold_index, 3] = alpha  # Alpha value
    remaining_inferno = plt.cm.inferno(np.linspace(0, 1, 256 - threshold_index))
    custom_cmap[threshold_index:, :3] = remaining_inferno[:, :3]
    custom_cmap[threshold_index:, 3] = alpha  # Alpha value
    return mcolors.ListedColormap(custom_cmap)


def get_inferno_with_alpha():
    plasma = plt.cm.inferno(np.linspace(0, 1, 256))
    alphas = np.linspace(0, 1, 256)
    plasma_with_alpha = np.zeros((256, 4))
    plasma_with_alpha[:, 0:3] = plasma[:, 0:3]
    plasma_with_alpha[:, 3] = alphas
    return mcolors.ListedColormap(plasma_with_alpha)


red_cmap = mcolors.LinearSegmentedColormap('RedMap', segmentdata={
    'red': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    'blue': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
})

blue_cmap = mcolors.LinearSegmentedColormap('BlueMap', segmentdata={
    'red': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    'green': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    'blue': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
    'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
})


def plot_attention_video(sims_by_head, frames, audio, video_fps, audio_fps, output_filename):
    prepped_sims = _prep_sims_for_plotting(sims_by_head, frames)
    n_frames, _, vh, vw = frames.shape
    sims_all = prepped_sims["sims_all"].clamp_min(0)
    sims_all -= sims_all.min()
    sims_all = sims_all / sims_all.max()
    cmap = get_inferno_with_alpha()
    layer1 = torch.cat([frames, torch.ones(n_frames, 1, vh, vw)], axis=1)
    layer2 = torch.tensor(cmap(sims_all.squeeze().detach().cpu())).permute(0, 3, 1, 2)
    write_video_with_audio(
        alpha_blend_layers([layer1, layer2]),
        audio,
        video_fps,
        audio_fps,
        output_filename)


def plot_2head_attention_video(sims_by_head, frames, audio, video_fps, audio_fps, output_filename):
    prepped_sims = _prep_sims_for_plotting(sims_by_head, frames)
    sims_1 = prepped_sims["sims_1"]
    sims_2 = prepped_sims["sims_2"]

    n_frames, _, vh, vw = frames.shape

    mask = sims_1 > sims_2
    sims_1 *= mask
    sims_2 *= (~mask)

    sims_1 = sims_1.clamp_min(0)
    sims_1 -= sims_1.min()
    sims_1 = sims_1 / sims_1.max()

    sims_2 = sims_2.clamp_min(0)
    sims_2 -= sims_2.min()
    sims_2 = sims_2 / sims_2.max()

    layer1 = torch.cat([frames, torch.ones(n_frames, 1, vh, vw)], axis=1)
    layer2_head1 = torch.tensor(red_cmap(sims_1.squeeze().detach().cpu())).permute(0, 3, 1, 2)
    layer2_head2 = torch.tensor(blue_cmap(sims_2.squeeze().detach().cpu())).permute(0, 3, 1, 2)

    write_video_with_audio(
        alpha_blend_layers([layer1, layer2_head1, layer2_head2]),
        audio,
        video_fps,
        audio_fps,
        output_filename)


def plot_feature_video(image_feats,
                       audio_feats,
                       frames,
                       audio,
                       video_fps,
                       audio_fps,
                       video_filename,
                       audio_filename):
    with torch.no_grad():
        image_feats_ = image_feats.cpu()
        audio_feats_ = audio_feats.cpu()
        [red_img_feats, red_audio_feats], _ = pca([
            image_feats_,
            audio_feats_.tile(image_feats_.shape[0], 1, 1, 1)
        ])
        _, _, vh, vw = frames.shape
        red_img_feats = F.interpolate(red_img_feats, size=(vh, vw), mode="bicubic")
        red_audio_feats = red_audio_feats[0].unsqueeze(0)
        red_audio_feats = F.interpolate(red_audio_feats, size=(50, red_img_feats.shape[0]), mode="bicubic")

    write_video_with_audio(
        (red_img_feats.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8),
        audio,
        video_fps,
        audio_fps,
        video_filename)

    red_audio_feats_expanded = red_audio_feats.tile(red_img_feats.shape[0], 1, 1, 1)
    red_audio_feats_expanded = F.interpolate(red_audio_feats_expanded, scale_factor=6, mode="bicubic")
    for i in range(red_img_feats.shape[0]):
        center_index = i * 6
        min_index = max(center_index - 2, 0)
        max_index = min(center_index + 2, red_audio_feats_expanded.shape[-1])
        red_audio_feats_expanded[i, :, :, min_index:max_index] = 1

    write_video_with_audio(
        (red_audio_feats_expanded.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8),
        audio,
        video_fps,
        audio_fps,
        audio_filename)


def display_video_in_notebook(path):
    mp4 = open(path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url))
