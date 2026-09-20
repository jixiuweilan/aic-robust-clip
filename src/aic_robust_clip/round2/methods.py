"""Deterministic TURN/FINE and SNSCL project variants (not paper replicas)."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MethodError(ValueError):
    pass


def gmm(values, *, high=False, max_iter=100):
    """Two-component 1D EM, quantile initialization, explicit convergence."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        raise MethodError("GMM requires >=8 finite, nonconstant scores")
    means = np.quantile(x, [.25, .75])
    variances = np.full(2, max(float(x.var()), 1e-6))
    weights = np.full(2, .5)
    previous = None
    for _ in range(max_iter):
        logp = (np.log(weights) - .5 * np.log(2 * np.pi * variances)
                - .5 * (x[:, None] - means) ** 2 / variances)
        maximum = logp.max(axis=1, keepdims=True)
        normalizer = maximum + np.log(np.exp(logp - maximum).sum(axis=1, keepdims=True))
        posterior = np.exp(logp - normalizer)
        likelihood = float(normalizer.mean())
        if not np.isfinite(posterior).all():
            raise MethodError("nonfinite GMM posterior")
        if previous is not None and abs(likelihood - previous) <= 1e-6:
            if abs(means[0] - means[1]) <= 1e-12:
                raise MethodError("degenerate GMM components have no reliable ordering")
            return posterior[:, np.argmax(means) if high else np.argmin(means)]
        mass = posterior.sum(axis=0)
        if (mass <= 1e-12).any():
            raise MethodError("collapsed GMM component")
        means = (posterior * x[:, None]).sum(axis=0) / mass
        variances = np.maximum((posterior * (x[:, None] - means) ** 2).sum(axis=0) / mass, 1e-6)
        weights, previous = mass / len(x), likelihood
    raise MethodError("GMM did not converge in 100 iterations")


def fine_scores(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) < 1e-12).any():
        raise MethodError("invalid FINE features")
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    if len(x) < x.shape[1]:
        # Same squared projection via the smaller sample Gram matrix.
        # Avoid a 512x512 eigensolve for every small long-tail class.
        values, vectors = np.linalg.eigh(x @ x.T)
        return max(0., float(values[-1])) * vectors[:, -1] ** 2
    _, vectors = np.linalg.eigh(x.T @ x)
    return (x @ vectors[:, -1]) ** 2


def select(labels, losses, features=None, *, method):
    labels, losses = np.asarray(labels), np.asarray(losses, dtype=np.float64)
    if labels.ndim != 1 or losses.shape != labels.shape or not np.isfinite(losses).all():
        raise MethodError("invalid complete scoring pass")
    if method == "snscl":
        probability = gmm(losses)  # Paper uses a global loss mixture.
        return np.ones(len(labels), dtype=bool), probability, {"scope": "global"}
    selected, probability, report = np.ones(len(labels), dtype=bool), np.full(len(labels), np.nan), {}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        scores = fine_scores(np.asarray(features)[indices]) if method == "fine" else losses[indices]
        reason = "fewer_than_8" if len(indices) < 8 else "constant" if np.ptp(scores) <= 1e-12 else None
        if reason:
            report[str(label)] = {"status": "unfiltered", "reason": reason, "count": len(indices), "selected": len(indices)}
            continue
        p = gmm(scores, high=method == "fine")
        keep = p >= .6
        if not keep.any():
            raise MethodError(f"class {label}: zero selected samples")
        selected[indices], probability[indices] = keep, p
        report[str(label)] = {"status": "filtered", "count": len(indices), "selected": int(keep.sum())}
    return selected, probability, report


def corrected_labels(previous, predictions, observed, reliability):
    weight = torch.where(reliability > .5, torch.ones_like(reliability), reliability)
    target = (1 - weight[:, None]) * predictions + weight[:, None] * observed
    return .99 * previous + .01 * target, weight


class StochasticFeature(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(),
                                     nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, 2 * dimension))

    def forward(self, features, *, generator):
        mu, logvar = self.network(features).float().chunk(2, dim=-1)
        # No silent variance clamp: nonfinite distributions stop the candidate.
        sigma = torch.exp(.5 * logvar)
        epsilon = torch.randn(mu.shape, generator=generator, device="cpu").to(mu.device)
        sample = mu + sigma * epsilon
        kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=-1)
        return sample, kl


class ClassQueue(nn.Module):
    def __init__(self, classes, dimension=128, capacity=32):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("features", torch.zeros(classes, capacity, dimension))
        self.register_buffer("count", torch.zeros(classes, dtype=torch.long))
        self.register_buffer("pointer", torch.zeros(classes, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, features, labels, weights, *, generator):
        if features.ndim != 2 or features.shape[1] != self.features.shape[2] or len(features) != len(labels) or len(features) != len(weights):
            raise MethodError("queue feature/label/weight coverage mismatch")
        if not torch.isfinite(features).all() or not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise MethodError("invalid queue input")
        for feature, label, weight in zip(features, labels, weights):
            c = int(label)
            if not 0 <= c < len(self.count):
                raise MethodError("queue label outside class map")
            if float(torch.rand((), generator=generator)) >= float(weight):
                continue  # Rejection changes neither storage nor cursor.
            p = int(self.pointer[c])
            self.features[c, p].copy_(feature.detach())
            self.pointer[c] = (p + 1) % self.capacity
            self.count[c] = min(int(self.count[c]) + 1, self.capacity)

    def loss(self, queries, labels):
        valid = torch.arange(self.capacity, device=self.count.device)[None, :] < self.count[:, None]
        keys = self.features[valid].detach()
        if not len(keys):
            return queries.sum(dim=-1) * 0
        classes = torch.arange(len(self.count), device=self.count.device)[:, None].expand_as(valid)[valid]
        logits = queries.float() @ keys.float().T / .07
        positives = labels[:, None] == classes[None, :]
        logp = F.log_softmax(logits, dim=1)
        return -(logp * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)


class SNSCL(nn.Module):
    def __init__(self, student, class_count, *, seed=17):
        super().__init__()
        dimension = student.encoder.output_dim
        self.stochastic = StochasticFeature(dimension)
        self.projector = nn.Linear(dimension, 128)
        self.momentum = copy.deepcopy(student).requires_grad_(False).eval()
        self.momentum_stochastic = copy.deepcopy(self.stochastic).requires_grad_(False).eval()
        self.momentum_projector = copy.deepcopy(self.projector).requires_grad_(False).eval()
        self.queue = ClassQueue(class_count)
        self.generator = torch.Generator().manual_seed(seed)

    def loss(self, features, labels):
        sample, kl = self.stochastic(features, generator=self.generator)
        queries = F.normalize(self.projector(sample).float(), dim=-1)
        return .1 * self.queue.loss(queries, labels) + 1e-4 * kl

    @torch.no_grad()
    def successful_update(self, student, images, labels, weights):
        for target, source in ((self.momentum, student), (self.momentum_stochastic, self.stochastic),
                               (self.momentum_projector, self.projector)):
            for p, q in zip(target.parameters(), source.parameters()):
                p.mul_(.999).add_(q.detach(), alpha=.001)
            for p, q in zip(target.buffers(), source.buffers()):
                p.copy_(q)
        # Bound GPU image memory: callers retain CPU microbatches for one update.
        for pixels, hard, weight in zip(images, labels, weights):
            features = self.momentum.encoder(pixels.to(next(student.parameters()).device), no_grad=True)
            sample, _ = self.momentum_stochastic(features, generator=self.generator)
            keys = F.normalize(self.momentum_projector(sample).float(), dim=-1)
            self.queue.enqueue(keys, hard, weight, generator=self.generator)


class MethodState:
    def __init__(self, method, sample_ids, labels, classes):
        if method not in {"ce", "turn", "fine", "snscl"} or len(sample_ids) != len(set(sample_ids)) or not sample_ids:
            raise MethodError("invalid method/train IDs")
        self.method, self.ids = method, list(sample_ids)
        self.labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.observed = F.one_hot(self.labels, classes).float()
        self.soft = self.observed.clone()
        self.weights = torch.ones(len(self.ids))
        self.selected = torch.ones(len(self.ids), dtype=torch.bool)
        self.completed_epochs, self.report = 0, {}

    def indices(self, ids):
        if any(s not in self.index for s in ids):
            raise MethodError("non-train ID entered method state/queue")
        return torch.tensor([self.index[s] for s in ids])

    def rescore(self, ids, losses, features, probabilities, *, completed_epochs):
        if list(ids) != self.ids:
            raise MethodError("scoring must cover exactly the ordered train IDs")
        previous = self.selected.clone()
        if self.method in {"turn", "fine"} or self.method == "snscl" and completed_epochs >= 5:
            keep, probability, self.report = select(self.labels.numpy(), losses, features, method=self.method)
            self.selected = torch.from_numpy(keep)
            if self.method == "snscl":
                self.soft, self.weights = corrected_labels(self.soft, torch.as_tensor(probabilities), self.observed,
                                                           torch.from_numpy(probability).float())
        self.report["changed"] = int((previous != self.selected).sum())
        self.completed_epochs = completed_epochs

    def state_dict(self):
        return {"version": "round2-method-v1", "method": self.method, "ids": self.ids, "labels": self.labels,
                "soft": self.soft, "weights": self.weights, "selected": self.selected,
                "completed_epochs": self.completed_epochs, "report": self.report}

    def load_state_dict(self, value):
        if set(value) != set(self.state_dict()) or value["version"] != "round2-method-v1" or value["method"] != self.method or value["ids"] != self.ids or not torch.equal(value["labels"], self.labels):
            raise MethodError("method identity/state missing or incompatible")
        for name in ("soft", "weights", "selected"):
            tensor, reference = value[name], getattr(self, name)
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all():
                raise MethodError(f"invalid method tensor: {name}")
        if ((value["soft"] < 0) | (value["soft"] > 1)).any() or not torch.allclose(value["soft"].sum(1), torch.ones(len(self.ids)), atol=1e-5):
            raise MethodError("invalid soft label probabilities")
        if ((value["weights"] < 0) | (value["weights"] > 1)).any() or not value["selected"].any():
            raise MethodError("invalid reliability/selection")
        if self.method != "snscl" and (not torch.equal(value["soft"], self.observed) or not torch.equal(value["weights"], torch.ones_like(self.weights))):
            raise MethodError("unexpected soft-label state for CE/TURN/FINE")
        if self.method in {"ce", "snscl"} and not value["selected"].all():
            raise MethodError("CE/SNSCL must preserve all training IDs")
        if any(not value["selected"][self.labels == label].any() for label in self.labels.unique()):
            raise MethodError("checkpoint selection removed an entire class")
        if type(value["completed_epochs"]) is not int or not 0 <= value["completed_epochs"] <= 30:
            raise MethodError("invalid method phase")
        for name in ("soft", "weights", "selected", "completed_epochs", "report"):
            setattr(self, name, copy.deepcopy(value[name]))
