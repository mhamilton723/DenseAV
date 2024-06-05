import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_average_precision
from tqdm import tqdm

from constants import *
from shared import unnorm, remove_axes


def prep_heatmap(sims, masks, h, w):
    masks = masks.to(torch.float32)
    hm = torch.einsum("bhwt,bt->bhw", sims, masks) / masks.sum(-1).reshape(-1, 1, 1)
    hm -= hm.min()
    hm /= hm.max()
    return F.interpolate(hm.unsqueeze(1), (h, w), mode="bilinear").squeeze(1)


def iou(prediction, target):
    prediction = prediction > 0.0
    target = target > 0.5
    intersection = torch.logical_and(prediction, target).sum().float()
    union = torch.logical_or(prediction, target).sum().float()
    if union == 0:
        return 1.0
    return (intersection / union).item()  # Convert to Python scalar


def multi_iou(prediction, target, k=20):
    prediction = torch.tensor(prediction)
    target = torch.tensor(target)
    target = target > 0.5

    thresholds = torch.linspace(prediction.min(), prediction.max(), k)
    hard_pred = prediction.unsqueeze(0) > thresholds.reshape(k, 1, 1, 1, 1)
    target = torch.broadcast_to(target.unsqueeze(0), hard_pred.shape)

    # Calculate IoU for each threshold
    intersection = torch.logical_and(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.logical_or(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.where(union == 0, torch.tensor(1.0), union)  # Avoid division by zero
    iou_scores = intersection / union

    # Find the best IoU and corresponding threshold
    best_iou, best_idx = torch.max(iou_scores, dim=0)
    # best_threshold = thresholds[best_idx]
    # print(best_threshold)
    return best_iou  # , best_threshold.item()


def get_paired_heatmaps(
        model,
        results,
        class_ids,
        timing,
        class_names=None):
    sims = model.sim_agg.get_pairwise_sims(
        results,
        raw=False,
        agg_sim=False,
        agg_heads=True
    ).squeeze(1).mean(-2)

    prompt_classes = torch.tensor(list(class_ids))
    gt = results["semseg"] == prompt_classes.reshape(-1, 1, 1)
    basic_masks = results[AUDIO_MASK]  # BxT
    _, fullh, fullw = gt.shape
    basic_heatmaps = prep_heatmap(sims, basic_masks, fullh, fullw)

    if timing is not None:
        prompt_timing = np.array(list(timing))
        raw_timing = torch.tensor([json.loads(t) for t in prompt_timing])
        timing = torch.clone(raw_timing)
        timing[:, 0] -= .2
        timing[:, 1] += .2
        total_length = (results['total_length'] / 16000)[0]
        fracs = timing / total_length
        bounds = basic_masks.shape[1] * fracs
        bounds[:, 0] = bounds[:, 0].floor()
        bounds[:, 1] = bounds[:, 1].ceil()
        bounds = bounds.to(torch.int64)
        advanced_masks = (F.one_hot(bounds, basic_masks.shape[1]).cumsum(-1).sum(-2) == 1).to(basic_masks)
        advanced_heatmaps = prep_heatmap(sims, advanced_masks, fullh, fullw)

    metrics = defaultdict(list)
    unique_classes = torch.unique(prompt_classes)

    should_plot = class_names is not None

    if should_plot:
        prompt_names = np.array(list(class_names))

    for prompt_class in tqdm(unique_classes):
        subset = torch.where(prompt_classes == prompt_class)[0]
        gt_subset = gt[subset]
        basic_subset = basic_heatmaps[subset]
        metrics["basic_ap"].append(binary_average_precision(basic_subset.flatten(), gt_subset.flatten()))
        metrics["basic_iou"].append(multi_iou(basic_subset.flatten(), gt_subset.flatten()))

        if timing is not None:
            advanced_subset = advanced_heatmaps[subset]
            metrics["advanced_ap"].append(binary_average_precision(advanced_subset.flatten(), gt_subset.flatten()))
            metrics["advanced_iou"].append(multi_iou(advanced_subset.flatten(), gt_subset.flatten()))

        if should_plot:
            prompt_class_subset = prompt_classes[subset]
            name_subset = prompt_names[subset]
            print(prompt_class, name_subset, prompt_class_subset)
            n_imgs = min(len(subset), 5)
            if n_imgs > 1:
                fig, axes = plt.subplots(n_imgs, 5, figsize=(4 * 5, n_imgs * 3))
                frame_subset = unnorm(results[IMAGE_INPUT][subset].squeeze(1)).permute(0, 2, 3, 1)
                semseg_subset = results["semseg"][subset]
                for img_num in range(n_imgs):
                    axes[img_num, 0].imshow(frame_subset[img_num])
                    axes[img_num, 1].imshow(basic_subset[img_num])
                    axes[img_num, 2].imshow(advanced_subset[img_num])
                    axes[img_num, 3].imshow(gt_subset[img_num])
                    axes[img_num, 4].imshow(semseg_subset[img_num], cmap="tab20", interpolation='none')

                axes[0, 0].set_title("Image")
                class_name = name_subset[0].split(",")[0]
                axes[0, 1].set_title(f"{class_name} Basic Heatmap")
                axes[0, 2].set_title(f"{class_name} Advanced Heatmap")
                axes[0, 3].set_title("True Mask")
                axes[0, 4].set_title("True Seg")
                remove_axes(axes)
                plt.tight_layout()
                plt.show()

    return metrics, unique_classes
