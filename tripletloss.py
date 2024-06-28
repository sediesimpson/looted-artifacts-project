from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from datasets import MetricDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from video import PLTVideoWriter


class FaceNet(nn.Module):
    def __init__(self, dim=2, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        self.resnet = resnet50(weights=weights)
        self.resnet.requires_grad_(False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return F.normalize(self.resnet(x))


def lfw_figure(dataset: MetricDataset):
    num_rows = 10
    num_cols = 10
    size = dataset.train.values.shape[-1]
    print(size)

    pixels = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)
    indices = np.random.choice(len(dataset.train.values), num_rows * num_cols, replace=False)
    for r in range(num_rows):
        for c in range(num_cols):
            i = r * num_cols + c
            img = dataset.train.values[indices[i]].transpose(1, 2, 0) * 255
            pixels[r*size:(r+1)*size, c*size:(c+1)*size] = img.astype(np.uint8)

    plt.figure(figsize=(num_cols, num_rows))
    plt.imshow(pixels)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("../figures/lfw.pdf")


def lfw_circle(dataset: MetricDataset, num_ids=6):
    counts = np.bincount(dataset.train.ids)
    print(len(counts))
    print((counts > 1).sum(), "with 2 or more images")
    ids = np.nonzero(counts > 2)[0]
    ids = np.random.choice(ids, num_ids, replace=False)

    circle = np.linspace(0, 2 * np.pi, 100)
    radius = 1

    angles = np.linspace(0, 2 * np.pi, num_ids + 1)[:num_ids] + np.random.uniform(-0.15, 0.15, size=num_ids)
    plt.figure(figsize=(15, 9))
    gs = plt.GridSpec(3, 5)

    ax = plt.subplot(gs[:, 2:])
    ax.plot(np.cos(circle) * radius, np.sin(circle) * radius, 'k')
    
    for i, (image_id, angle) in enumerate(zip(ids, angles)):
        idx = dataset.train.ids == image_id
        images = dataset.train.values[idx]
        images = np.moveaxis(images, 1, 3)
        r = i // 2
        c = i % 2
        ax_img = plt.subplot(gs[r, c])
        ax_img.imshow(images[0])
        ax_img.axis('off')

        angle_noise = np.linspace(-0.25, 0.25, len(images))
        radius_noise = np.random.uniform(-0.15, 0.15, size=len(images))
        for image, an, rn in zip(images, angle_noise, radius_noise):
            x = np.cos(angle + an) * (radius - rn)
            y = np.sin(angle + an) * (radius - rn)
            ax.imshow(image, extent=(x-0.15, x+0.15, y-0.15, y+0.15), zorder=2)

    ax.set_xlim(-radius - .25, radius + .25)
    ax.set_ylim(-radius - .25, radius + .25)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("../figures/lfw_circle.pdf")


def lfw_minitrain():
    dataset = MetricDataset.lfw()
    counts = np.bincount(dataset.train.ids)
    id_by_counts = [[] for _ in range(counts.max())]
    for i in range(len(counts)):
        id_by_counts[counts[i] - 1].append(i)

    train_ids = id_by_counts[-1] + id_by_counts[-2]

    idx = []
    for i in train_ids:
        idx.append(np.where(dataset.train.ids == i)[0])
    
    idx = np.concatenate(idx)
    idx.sort()
    
    values = dataset.train.values[idx]
    ids = dataset.train.ids[idx] 

    np.savez("lfw_minitrain.npz", values=values, ids=ids, labels=dataset.train.labels)


Snapshot = NamedTuple("Snapshot", [("step", int), ("metric", float), ("embeddings", np.ndarray)])


def facenet():
    alpha = 0.7
    batch_size = 64
    data = np.load("lfw_minitrain.npz")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # this is so small we can precompute all valid triplets
    ids = data["ids"]
    idx_by_id = {}
    for i, id in enumerate(ids):
        if id not in idx_by_id:
            idx_by_id[id] = []

        idx_by_id[id].append(i)

    triplets = []
    for id, idx in idx_by_id.items():
        for a in range(len(idx)):
            for p in range(a + 1, len(idx)):
                for n, other in enumerate(ids):
                    if id == other:
                        continue
                    triplets.append((idx[a], idx[p], n))
    
    for a, p, n in triplets:
        assert ids[a] == ids[p]
        assert ids[a] != ids[n]

    triplets = torch.LongTensor(triplets).to(device)

    weights = ResNet50_Weights.DEFAULT
    model = FaceNet(weights=weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=1e-5)

    images = torch.from_numpy(data["values"]).to(device)
    preprocess = weights.transforms()

    images = preprocess(images)

    def get_valid_triplets(embeddings: torch.Tensor):
        triplet_embeddings = embeddings[triplets]
        ap = torch.linalg.norm(triplet_embeddings[:, 0] - triplet_embeddings[:, 1], dim=1)
        an = torch.linalg.norm(triplet_embeddings[:, 0] - triplet_embeddings[:, 2], dim=1)
        metric = ap - an + alpha
        valid = (metric > 0 & (ap < an))
        return torch.where(valid)[0], metric.relu().sum()

    step = 0
    snapshots = []
    for i in range(100):
        model.resnet.fc.eval()
        embeddings = model(images)
        valid_triplets, metric = get_valid_triplets(embeddings)
        valid_triplets = valid_triplets.cpu().numpy()
        model.resnet.fc.train()

        np.random.shuffle(valid_triplets)

        print("valid triplets:", len(valid_triplets), "metric:", metric.item())

        snapshots.append(Snapshot(step, metric.item(), embeddings.detach().cpu().numpy()))

        if len(valid_triplets) == 0:
            break

        for start in tqdm(range(0, len(valid_triplets), batch_size), f"Epoch {i}"):
            end = min(start + batch_size, len(valid_triplets))
            batch = torch.from_numpy(valid_triplets[start:end]).to(device)
            batch_triplets = triplets[batch]

            optimizer.zero_grad()
            embeddings = model(images)
            triplet_embeddings = embeddings[batch_triplets]
            ap = torch.linalg.norm(triplet_embeddings[:, 0] - triplet_embeddings[:, 1], dim=1)
            an = torch.linalg.norm(triplet_embeddings[:, 0] - triplet_embeddings[:, 2], dim=1)
            metric = ap - an + alpha
            loss = F.relu(metric).mean()
            loss.backward()

            optimizer.step()

            step += 1
            if step % 5 == 0:
                _, metric = get_valid_triplets(embeddings)
                snapshots.append(Snapshot(step, metric.item(), embeddings.detach().cpu().numpy()))

    results = {"snapshots": snapshots, "net": model.state_dict()}
    torch.save(results, "facenet.results")


def facenet_video():
    data = np.load("lfw_minitrain.npz")
    results = torch.load("facenet.results")
    snapshots = results["snapshots"]

    steps = [s.step for s in snapshots]
    metrics = [s.metric for s in snapshots]

    ids = data["ids"]
    values = data["values"]
    values = np.moveaxis(values, 1, 3)

    pixels = np.zeros((4*250, 2*250, 3), dtype=np.uint8)
    unique_ids = np.unique(ids)
    id_to_image = {}
    id_to_label = {}
    exemplars = []
    for label, id in enumerate(unique_ids):
        idx = np.where(ids == id)[0][0]
        r = label // 2
        c = label % 2
        pixels[r*250:(r+1)*250, c*250:(c+1)*250] = (values[idx] * 255).astype(np.uint8)
        id_to_image[id] = values[idx]
        id_to_label[id] = label
        exemplars.append(idx)

    labels = np.array([id_to_label[id] for id in ids])

    def plot(fig: plt.Figure, step: int):
        snapshot = snapshots[step]
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 2])

        circle = np.linspace(0, 2 * np.pi, 100)
        radius = 1

        images_ax = fig.add_subplot(gs[0, 0])
        images_ax.imshow(pixels)
        images_ax.axis('off')

        circle_ax = fig.add_subplot(gs[0, 1])
        circle_ax.plot(np.cos(circle) * radius, np.sin(circle) * radius, 'k', alpha=0.5, dashes=(5, 5))
        circle_ax.scatter(snapshot.embeddings[:, 0], snapshot.embeddings[:, 1], c=labels, cmap='jet', alpha=1)

        cmap = plt.get_cmap('jet')
        for i, idx in enumerate(exemplars):
            x, y = snapshot.embeddings[idx]
            image = values[idx]
            x0 = x - 0.15
            x1 = x + 0.15
            y0 = y - 0.15
            y1 = y + 0.15
            circle_ax.imshow(image, extent=(x0, x1, y0, y1), zorder=2 + 2*i)
            color = cmap(i / len(exemplars))[:3]
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor=color, facecolor='none', zorder=2 + 2*i+1)
            circle_ax.add_patch(rect)

        circle_ax.set_xlim(-radius - .25, radius + .25)
        circle_ax.set_ylim(-radius - .25, radius + .25)
        circle_ax.axis('off')
        circle_ax.set_title("Metric Space")

        loss_ax = fig.add_subplot(gs[0, 2])
        loss_ax.plot(steps, metrics, "b-")
        loss_ax.plot(snapshot.step, snapshot.metric, 'bo')
        loss_ax.set_title("Triplet Loss")
        
        fig.tight_layout()

    with PLTVideoWriter("../videos/facenet.mp4", 10, 4, 150, plot, framerate=5) as video:
        for step in range(len(snapshots)):
            video.write_frame(step=step)


def facenet_circle():
    data = np.load("lfw_minitrain.npz")
    results = torch.load("facenet.results")
    snapshots = results["snapshots"]

    steps = [s.step for s in snapshots]
    metrics = [s.metric for s in snapshots]

    ids = data["ids"]
    values = data["values"]
    values = np.moveaxis(values, 1, 3)

    pixels = np.zeros((4*250, 2*250, 3), dtype=np.uint8)
    unique_ids = np.unique(ids)
    id_to_image = {}
    id_to_label = {}
    exemplars = []
    for label, id in enumerate(unique_ids):
        idx = np.where(ids == id)[0][0]
        r = label // 2
        c = label % 2
        pixels[r*250:(r+1)*250, c*250:(c+1)*250] = (values[idx] * 255).astype(np.uint8)
        id_to_image[id] = values[idx]
        id_to_label[id] = label
        exemplars.append(idx)

    labels = np.array([id_to_label[id] for id in ids])

    def plot(fig: plt.Figure, step: int):
        snapshot = snapshots[step]
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 2])

        circle = np.linspace(0, 2 * np.pi, 100)
        radius = 1

        images_ax = fig.add_subplot(gs[0, 0])
        images_ax.imshow(pixels)
        images_ax.axis('off')

        circle_ax = fig.add_subplot(gs[0, 1])
        circle_ax.plot(np.cos(circle) * radius, np.sin(circle) * radius, 'k', alpha=0.5, dashes=(5, 5))
        circle_ax.scatter(snapshot.embeddings[:, 0], snapshot.embeddings[:, 1], c=labels, cmap='jet', alpha=1)

        cmap = plt.get_cmap('jet')
        for i, idx in enumerate(exemplars):
            x, y = snapshot.embeddings[idx]
            image = values[idx]
            x0 = x - 0.15
            x1 = x + 0.15
            y0 = y - 0.15
            y1 = y + 0.15
            circle_ax.imshow(image, extent=(x0, x1, y0, y1), zorder=2 + 2*i)
            color = cmap(i / len(exemplars))[:3]
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor=color, facecolor='none', zorder=2 + 2*i+1)
            circle_ax.add_patch(rect)

        circle_ax.set_xlim(-radius - .25, radius + .25)
        circle_ax.set_ylim(-radius - .25, radius + .25)
        circle_ax.axis('off')
        circle_ax.set_title("Metric Space")

        loss_ax = fig.add_subplot(gs[0, 2])
        loss_ax.plot(steps, metrics, "b-")
        loss_ax.plot(snapshot.step, snapshot.metric, 'bo')
        loss_ax.set_title("Triplet Loss")

    fig = plt.figure(figsize=(10, 4))
    plot(fig, 0)
    fig.savefig("../figures/facenet_0.pdf")

    fig.clf()
    plot(fig, -1)
    fig.savefig("../figures/facenet_1.pdf")


if __name__ == "__main__":
    # np.random.seed(20080524)
    # dataset = MetricDataset.lfw()
    # lfw_figure(dataset)
    # lfw_circle(dataset)
    lfw_minitrain()
    facenet()
    #facenet_video()
    facenet_circle()
