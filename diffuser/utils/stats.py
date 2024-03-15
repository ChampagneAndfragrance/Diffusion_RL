import numpy as np

def get_stats_batch(ground_truth_list, sample_list, total_area, batch_size, n_samples):
    min_dists = []
    dist_averages = []
    for gts, samples in zip(ground_truth_list, sample_list):
        g = np.reshape(gts, (batch_size, n_samples, gts.shape[-2], gts.shape[-1]))
        s = np.reshape(samples, (batch_size, n_samples, samples.shape[-2], samples.shape[-1]))

        dist_averages.append(np.linalg.norm(gts - samples, axis=-1))

        dist = np.linalg.norm(g - s, axis=-1)

        indices = np.argmin(dist.sum(axis=-1), axis=-1)
        best_dist = dist[np.arange(batch_size), indices]

        # best_dist = np.amin(dist, axis=1)
        # dist = np.linalg.norm(gts - samples, axis=-1)
        min_dists.append(best_dist)
    dist_t = np.concatenate(min_dists, axis=0)

    dist_averages = np.concatenate(dist_averages, axis=0)
    return dist_t, dist_averages