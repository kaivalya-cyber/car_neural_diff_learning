import numpy as np


class SensorAugmenter:
    def __init__(self, dropout_prob: float = 0.05, noise_std: float = 0.02,
                 occlusion_prob: float = 0.1, block_size: int = 3):
        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.occlusion_prob = occlusion_prob
        self.block_size = block_size
        self.augmentations_applied = []

    def augment(self, readings: np.ndarray) -> np.ndarray:
        aug = readings.copy().astype(np.float32)
        applied = []

        if np.random.random() < self.dropout_prob:
            dropout_mask = np.random.random(len(aug)) < 0.3
            aug[dropout_mask] = 1.0
            applied.append("dropout")

        if self.noise_std > 0:
            noise = np.random.randn(len(aug)) * self.noise_std
            aug = np.clip(aug + noise, 0.0, 1.0)
            applied.append("noise")

        if np.random.random() < self.occlusion_prob and len(aug) > self.block_size:
            start = np.random.randint(0, len(aug) - self.block_size)
            aug[start:start + self.block_size] = 1.0
            applied.append("occlusion")

        self.augmentations_applied.append(applied)
        return aug

    def reset_stats(self):
        self.augmentations_applied = []

    def stats(self) -> dict:
        if not self.augmentations_applied:
            return {"total_steps": 0}
        total = len(self.augmentations_applied)
        counts = {}
        for applied in self.augmentations_applied:
            for a in applied:
                counts[a] = counts.get(a, 0) + 1
        return {
            "total_steps": total,
            "augmentation_rates": {k: v / total for k, v in counts.items()},
        }


class MultiSensorAugmenter:
    def __init__(self, base_count: int = 16, augment_count: int = 8):
        self.base_count = base_count
        self.augment_count = augment_count
        self.augmenter = SensorAugmenter()

    def augment_batch(self, readings: np.ndarray) -> np.ndarray:
        aug = readings.copy()
        for i in range(len(aug)):
            aug[i] = self.augmenter.augment(aug[i])
        return aug


def main():
    augmenter = SensorAugmenter(dropout_prob=0.3, noise_std=0.05, occlusion_prob=0.2)
    print("Sensor Augmentation Demo")
    print("=" * 50)

    for step in range(20):
        clean = np.random.rand(16) * 0.8 + 0.1
        clean[:3] = 0.05
        augmented = augmenter.augment(clean)
        changes = np.sum(np.abs(augmented - clean) > 0.01)
        if changes > 0:
            print(f"  Step {step:3d}: {changes} sensors modified | "
                  f"mean Δ={np.mean(np.abs(augmented - clean)):.4f}")

    stats = augmenter.stats()
    print(f"\nStats: {stats['total_steps']} steps processed")
    for aug, rate in stats.get("augmentation_rates", {}).items():
        print(f"  {aug}: {rate:.0%} of steps")
    print("=" * 50)


if __name__ == "__main__":
    main()
