import csv
import os
import tempfile

import gradio as gr
import requests
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from featup.util import norm
from torchaudio.functional import resample

from denseav.plotting import plot_attention_video, plot_2head_attention_video, plot_feature_video
from denseav.shared import norm, crop_to_divisor, blur_dim
from os.path import join

if __name__ == "__main__":

    # os.environ['TORCH_HOME'] = '/tmp/.cache'
    # os.environ['GRADIO_EXAMPLES_CACHE'] = '/tmp/gradio_cache'
    # sample_images_dir = "/tmp/samples"
    sample_videos_dir = "samples"


    def download_video(url, save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as file:
            file.write(response.content)


    base_url = "https://marhamilresearch4.blob.core.windows.net/denseav-public/samples/"
    sample_videos_urls = {
        "puppies.mp4": base_url + "puppies.mp4",
        "peppers.mp4": base_url + "peppers.mp4",
        "boat.mp4": base_url + "boat.mp4",
        "elephant2.mp4": base_url + "elephant2.mp4",

    }

    # Ensure the directory for sample videos exists
    os.makedirs(sample_videos_dir, exist_ok=True)

    # Download each sample video
    for filename, url in sample_videos_urls.items():
        save_path = os.path.join(sample_videos_dir, filename)
        # Download the video if it doesn't already exist
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            download_video(url, save_path)
        else:
            print(f"{filename} already exists. Skipping download.")

    csv.field_size_limit(100000000)
    options = ['language', "sound_and_language", "sound"]
    load_size = 224
    plot_size = 224

    video_input = gr.Video(label="Choose a video to featurize", height=480)
    model_option = gr.Radio(options, value="language", label='Choose a model')

    video_output1 = gr.Video(label="Audio Video Attention", height=480)
    video_output2 = gr.Video(label="Multi-Head Audio Video Attention (Only Availible for sound_and_language)",
                             height=480)
    video_output3 = gr.Video(label="Visual Features", height=480)
    video_output4 = gr.Video(label="Audio Features", height=480)

    models = {o: torch.hub.load("mhamilton723/DenseAV", o) for o in options}


    def process_video(video, model_option):
        model = models[model_option].cuda()

        original_frames, audio, info = torchvision.io.read_video(video, end_pts=10, pts_unit='sec')
        sample_rate = 16000

        if info["audio_fps"] != sample_rate:
            audio = resample(audio, info["audio_fps"], sample_rate)
        audio = audio[0].unsqueeze(0)

        img_transform = T.Compose([
            T.Resize(load_size, Image.BILINEAR),
            lambda x: crop_to_divisor(x, 8),
            lambda x: x.to(torch.float32) / 255,
            norm])

        frames = torch.cat([img_transform(f.permute(2, 0, 1)).unsqueeze(0) for f in original_frames], axis=0)

        plotting_img_transform = T.Compose([
            T.Resize(plot_size, Image.BILINEAR),
            lambda x: crop_to_divisor(x, 8),
            lambda x: x.to(torch.float32) / 255])

        frames_to_plot = plotting_img_transform(original_frames.permute(0, 3, 1, 2))

        with torch.no_grad():
            audio_feats = model.forward_audio({"audio": audio.cuda()})
            audio_feats = {k: v.cpu() for k, v in audio_feats.items()}
            image_feats = model.forward_image({"frames": frames.unsqueeze(0).cuda()}, max_batch_size=2)
            image_feats = {k: v.cpu() for k, v in image_feats.items()}

            sim_by_head = model.sim_agg.get_pairwise_sims(
                {**image_feats, **audio_feats},
                raw=False,
                agg_sim=False,
                agg_heads=False
            ).mean(dim=-2).cpu()

            sim_by_head = blur_dim(sim_by_head, window=3, dim=-1)
            print(sim_by_head.shape)

        temp_video_path_1 = tempfile.mktemp(suffix='.mp4')

        plot_attention_video(
            sim_by_head,
            frames_to_plot,
            audio,
            info["video_fps"],
            sample_rate,
            temp_video_path_1)

        if model_option == "sound_and_language":
            temp_video_path_2 = tempfile.mktemp(suffix='.mp4')

            plot_2head_attention_video(
                sim_by_head,
                frames_to_plot,
                audio,
                info["video_fps"],
                sample_rate,
                temp_video_path_2)

        else:
            temp_video_path_2 = None

        temp_video_path_3 = tempfile.mktemp(suffix='.mp4')
        temp_video_path_4 = tempfile.mktemp(suffix='.mp4')

        plot_feature_video(
            image_feats["image_feats"].cpu(),
            audio_feats['audio_feats'].cpu(),
            frames_to_plot,
            audio,
            info["video_fps"],
            sample_rate,
            temp_video_path_3,
            temp_video_path_4,
        )
        return temp_video_path_1, temp_video_path_2, temp_video_path_3, temp_video_path_4

        return temp_video_path_1, temp_video_path_2, temp_video_path_3


    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("## Visualizing Sound and Language with DenseAV")
            gr.Markdown(
                "This demo allows you to explore the inner attention maps of DenseAV's dense multi-head contrastive operator.")
            with gr.Row():
                with gr.Column(scale=1):
                    model_option.render()
                with gr.Column(scale=3):
                    video_input.render()
            with gr.Row():
                submit_button = gr.Button("Submit")
            with gr.Row():
                gr.Examples(
                    examples=[
                        [join(sample_videos_dir, "puppies.mp4"), "sound_and_language"],
                        [join(sample_videos_dir, "peppers.mp4"), "language"],
                        [join(sample_videos_dir, "elephant2.mp4"), "language"],
                        [join(sample_videos_dir, "boat.mp4"), "language"]

                    ],
                    inputs=[video_input, model_option]
                )
            with gr.Row():
                video_output1.render()
                video_output2.render()
                video_output3.render()

        submit_button.click(fn=process_video, inputs=[video_input, model_option],
                            outputs=[video_output1, video_output2])

    # demo.launch(server_name="0.0.0.0", server_port=6006, debug=True)

    demo.launch(server_name="0.0.0.0", server_port=6006, debug=True)
    # demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
