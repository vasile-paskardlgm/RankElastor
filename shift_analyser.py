# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import torch
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.dataloaders import RankDataLoader
import src
import gc
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from typing import Dict
from scipy.stats import gaussian_kde

####################################
####### Erank Computation
####################################
def effective_rank_entropy(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: (d1, d2) tensor on CPU
    return: scalar effective rank
    """
    s = torch.linalg.svdvals(x)
    s = s + eps
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


####################################
####### Collcect Internal Representations
####################################
def collect_unimixer_internal_representations_one_batch(
    model: torch.nn.Module,
    batch_data,
    save_path: str,
) -> None:
    """
    Collect internal representations and batch-wise effective rank
    for ONE batch, then save to disk (CPU-safe).
    """

    model.eval()

    storage: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inp, out):
            # out: (B, d1, d2)
            out_cpu = out.detach().cpu()
            B = out_cpu.shape[0]

            eranks = torch.empty(B)
            for b in range(B):
                eranks[b] = effective_rank_entropy(out_cpu[b])

            storage[name] = {
                "repr": out_cpu,   # (B, d1, d2)
                "erank": eranks,   # (B,)
            }
        return hook

    # ---- Register hooks ----
    hooks.append(
        model.embedding_layer.register_forward_hook(
            make_hook("embedding_layer")
        )
    )

    hooks.append(
        model.embedding_tokenization.register_forward_hook(
            make_hook("embedding_tokenization")
        )
    )

    for i, layer in enumerate(model.backbone):
        hooks.append(
            layer.token_mixing.register_forward_hook(
                make_hook(f"backbone__{i}__token_mixing")
            )
        )
        hooks.append(
            layer.token_interaction.register_forward_hook(
                make_hook(f"backbone__{i}__token_interaction")
            )
        )

    # ---- Forward pass (ONE batch) ----
    with torch.no_grad():
        # batch_data = batch_data.to(device)
        _ = model.forward(batch_data)

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Save (CPU-safe) ----
    torch.save(storage, save_path)


####################################
####### Plot Erank Distribution
####################################
def plot_erank_transition(
    erank_a: np.ndarray,
    erank_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_name: str,
    font_size: int = 14,
    threshold_ratio: float = 0.05,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    os.makedirs("./figures", exist_ok=True)

    kde_a = gaussian_kde(erank_a)
    kde_b = gaussian_kde(erank_b)

    # ---- Step 1: evaluate on a wide initial grid ----
    combined = np.concatenate([erank_a, erank_b])
    lo, hi = combined.min(), combined.max()
    pad = 0.5 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 2000)

    y_a = kde_a(grid)
    y_b = kde_b(grid)

    # ---- Step 2: threshold-based support ----
    thr_a = threshold_ratio * y_a.max()
    thr_b = threshold_ratio * y_b.max()

    idx_a = np.where(y_a >= thr_a)[0]
    idx_b = np.where(y_b >= thr_b)[0]

    xmin = min(grid[idx_a[0]], grid[idx_b[0]])
    xmax = max(grid[idx_a[-1]], grid[idx_b[-1]])

    # ---- Small safety margin ----
    margin = 0.05 * (xmax - xmin + 1e-6)
    xmin -= margin
    xmax += margin

    # ---- Final high-res grid ----
    x = np.linspace(xmin, xmax, 600)
    y_a = kde_a(x)
    y_b = kde_b(x)

    y_max = max(y_a.max(), y_b.max())

    plt.figure(figsize=(6.5, 4.5))

    # ---- Curves ----
    plt.plot(x, y_a, color="darkgreen", linewidth=2, label=label_a)
    plt.plot(x, y_b, color="darkblue", linewidth=2, label=label_b)

    # ---- Filled areas ----
    plt.fill_between(x, y_a, color="green", alpha=0.30)
    plt.fill_between(x, y_b, color="blue", alpha=0.30)

    # ---- base statistics ----
    mean_a = float(np.mean(erank_a))
    mean_b = float(np.mean(erank_b))

    x_center = 0.5 * (mean_a + mean_b)
    delta = 0.35 * abs(mean_b - mean_a)

    # ---- enforce minimum visible length ----
    min_len = 0.08 * (xmax - xmin)   # tunable (5–10% works well)
    arrow_len = max(delta, min_len)

    x_start_arrow = x_center - arrow_len / 2 + 0.25
    x_end_arrow   = x_center + arrow_len / 2 + 0.25

    # ---- enforce semantic direction rules ----
    force_left_to_right = label_b.startswith("Mixing")
    force_right_to_left = label_b.startswith("FFN")

    if force_left_to_right and x_start_arrow > x_end_arrow:
        x_start_arrow, x_end_arrow = x_end_arrow, x_start_arrow

    if force_right_to_left and x_start_arrow < x_end_arrow:
        x_start_arrow, x_end_arrow = x_end_arrow, x_start_arrow

    # ---- vertical placement ----
    y_arrow = 1.08 * max(y_a.max(), y_b.max())

    # ---- draw arrow ----
    plt.annotate(
        "",
        xy=(x_end_arrow, y_arrow),
        xytext=(x_start_arrow, y_arrow),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=3.3,
            alpha=0.9
        )
    )

    # ---- Axes ----
    plt.xlim(xmin, xmax)
    plt.ylim(0, y_max * 1.35)

    plt.xlabel("Effective Rank", fontsize=font_size+9.5)
    plt.ylabel("Density", fontsize=font_size+9.5)

    plt.title(f"{label_a} to {label_b}", fontsize=font_size + 9.5)

    plt.legend(loc="upper left", fontsize=font_size + 6)
    plt.tick_params(axis="both", labelsize=font_size + 6.5)

    plt.tight_layout()
    plt.savefig(f"./figures/{save_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()



if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='UniMixer_criteo_permute_pooling', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--batch_num', type=int, default=0, help='Which batch to analyze')
    parser.add_argument("--retest", action="store_true", help="Recompute and save internal representations")
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    # set_logger(params)
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)

    save_path = f"./save_data/{args['expid']}.pt"
    os.makedirs("./save_data", exist_ok=True)

    ## Data Collections
    if args['retest'] or not os.path.exists(save_path):
        print("[INFO] Computing and saving internal representations...")
        print("Load trained model from {}...".format(model.checkpoint))
        model.load_weights(model.checkpoint)

        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        for batch_data in test_gen:
            collect_unimixer_internal_representations_one_batch(
            model=model,
            batch_data=batch_data,
            save_path=save_path,
            )
            break

    else:
        print("[INFO] Loading cached internal representations...")

    ## Plot
    data = torch.load(f"./save_data/{args['expid']}.pt")

    stages = [
        ("embedding_layer", "Raw"),
        ("embedding_tokenization", "Tokenization"),
        ("backbone__0__token_mixing", "Mixing 1"),
        ("backbone__0__token_interaction", "FFN 1"),
        ("backbone__1__token_mixing", "Mixing 2"),
        ("backbone__1__token_interaction", "FFN 2"),
    ]

    transition_indices = [
        # consecutive transitions
        (i, i + 1) for i in range(len(stages) - 1)
    ] + [
        # additional global transitions
        (0, 2),  # Raw -> Mixing 1
        (0, 5),  # Raw -> FFN 2
    ]

    transition_indices = list(dict.fromkeys(transition_indices))

    font_size = 14

    for i, j in transition_indices:
        key_a, name_a = stages[i]
        key_b, name_b = stages[j]

        erank_a = data[key_a]["erank"].numpy()
        erank_b = data[key_b]["erank"].numpy()

        save_name = f"{args['expid']}_{name_a}_to_{name_b}"

        plot_erank_transition(
            erank_a=erank_a,
            erank_b=erank_b,
            label_a=name_a,
            label_b=name_b,
            save_name=save_name,
            font_size=font_size,
        )



r'''Save data structure:
data = torch.load("unimixer_internal_repr.pt")

data.keys()
# dict_keys([
#   'embedding_layer',
#   'embedding_tokenization',
#   'backbone__0__token_mixing',
#   'backbone__0__token_interaction',
#   ...
# ])

data["backbone__3__token_mixing"]["repr"].shape   # (B, d1, d2)
data["backbone__3__token_mixing"]["erank"].shape  # (B,)
'''