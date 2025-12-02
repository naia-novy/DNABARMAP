from collections import defaultdict
from scipy.ndimage import convolve1d

from dnabarmap.utils import hot_degenerate_base_mapping, import_cupy_numpy
np = import_cupy_numpy()

from scipy.ndimage import gaussian_filter1d

def sequences_to_array(sequences, max_len):
    # Convert string based DNA sequences to N x 4 array (int encoding)
    assert max_len is not None
    seq_array = np.full((len(sequences), max_len, 4), np.nan, dtype=np.float32)
    for i, sequence in enumerate(sequences):
        indices = [hot_degenerate_base_mapping[base] for base in sequence]
        indices = np.asarray(indices, dtype=seq_array.dtype)
        seq_array[i, :len(indices), :] = indices
    return seq_array

def reference_to_array(reference):
    # Convert string based reference DNA sequences to N x 4 array (int encoding)
    indices = [hot_degenerate_base_mapping[base] for base in reference]
    ref = np.array(indices)
    return ref[np.newaxis, np.newaxis, :, :]

def score_sequences_simple(sequence_array, reference_array):
    # Ensure reference shape matches
    if reference_array.ndim == sequence_array.ndim - 1:
        reference_array = np.broadcast_to(reference_array, sequence_array.shape)

    # Masks
    ref_mask = (reference_array != 6) & (reference_array != 0)
    seq_mask = sequence_array != 0

    # Masked arrays (without NaNs)
    ref_array = np.where(ref_mask, reference_array, 0)
    seq_array = np.where(seq_mask, sequence_array, 0)

    # Valid index mask
    valid_indices = ~np.isnan(ref_array).any(axis=-1) & ~np.isnan(seq_array).any(axis=-1)

    # Correct matches (only where ref==1 and seq==1)
    correct = np.sum((seq_array == 1) & (ref_array == 1), axis=-1)

    # Total possibilities
    possibilities = np.clip(np.sum(ref_array, axis=-1), 1e-8, None)

    # Base score
    score = (correct / possibilities)

    # Apply NaN for invalid
    score = np.where(valid_indices, score, np.nan)

    return score


def compute_adjacency_score(seqs, refs, max_run):
    # Ensure contiguous float32
    seqs = np.ascontiguousarray(seqs, dtype=np.float32)
    refs = np.ascontiguousarray(refs, dtype=np.float32)

    # probs = np.exp(refs.sum(axis=-1))
    probs = refs.sum(axis=-1)
    wins = (seqs == refs) & (seqs == 1)

    scores = wins.sum(axis=-1) / (probs + 1e-8)  # avoid div by zero
    scores = np.ascontiguousarray(scores, dtype=np.float32)

    final_scores = np.zeros_like(scores)

    slices = [np.pad(scores[..., i:], ((0, 0), (0, 0), (0, 0), (0,i)), constant_values=0.01) for i in range(max_run)]

    result_fw = np.prod(np.stack(slices[:max_run], axis=0), axis=0)
    final_scores += (result_fw)

    return final_scores


def find_best_rolls_batch(seqs, ref):
    # Parameters
    max_run = 10
    max_shift = seqs.shape[-2]
    min_shift = 0
    provided_range = np.arange(min_shift, max_shift + 1)

    # provided_range = np.arange(min_shift, max_shift + 1)
    n_rolls = len(provided_range)
    n_strands, n_seqs, seq_len, seq_dim = seqs.shape

    # Precompute rolled sequences in one big array
    # rolled_all = np.empty((n_strands, n_rolls, n_seqs, max_shift, seq_dim), dtype=seqs.dtype)
    rolled_all = np.empty((n_strands, n_rolls, n_seqs, ref.shape[-2], seq_dim), dtype=seqs.dtype)
    for idx, shift in enumerate(provided_range):
        # rolled_all[:,idx] = np.roll(seqs, shift=shift, axis=2)[:, :, :ref.shape[-2]]
        rolled = np.roll(seqs, shift=-shift, axis=2)
        if shift > 0:
            rolled[:, :, -shift:] = 0  # zero out wrapped portion
        rolled_all[:, idx] = rolled[:, :, :ref.shape[-2]]

    adjacency_matrix = compute_adjacency_score(rolled_all, ref[np.newaxis], max_run=max_run)
    adjacency_score = adjacency_matrix.sum(axis=(-1))
    smoothed = gaussian_filter1d(adjacency_score, axis=-2, sigma=10)

    # Pick best strand per sequence
    direction = np.argmax(np.max(smoothed, axis=1), axis=0)
    best_rolls = np.argmax(smoothed, axis=1)[direction, np.arange(len(direction))]

    return best_rolls, direction


def roll_batch(batch_array, roll_values):
    # Apply batched roll to array
    rolled = batch_array.copy()
    # Group sequence indices by their roll shift
    shift_groups = defaultdict(list)
    for idx, shift in enumerate(roll_values):
        if shift == 0:
            continue
        key = int(shift)  # ensure hashable
        shift_groups[key].append(idx)

    # Apply roll per unique shift
    for shift, indices in shift_groups.items():
        rolled_batch = np.roll(batch_array[indices], shift=-shift, axis=1)  # axis=1 assumes time or sequence axis
        rolled[indices] = rolled_batch

    return rolled

