from collections import defaultdict

from dnabarmap.utils import hot_degenerate_base_mapping, import_cupy_numpy
np = import_cupy_numpy()

from scipy.ndimage import gaussian_filter1d

def sequences_to_array(sequences, max_len):
    # Convert string based DNA sequences to N x 4 array (int encoding)
    assert max_len is not None
    seq_array = np.full((len(sequences), max_len, 4), np.nan, dtype=np.float32)
    for i, sequence in enumerate(sequences):
        shift = int((max_len - len(sequence)) / 2)
        indices = [hot_degenerate_base_mapping[base] for base in sequence]
        indices = np.asarray(indices, dtype=seq_array.dtype)
        seq_array[i, shift:len(indices)+shift, :] = indices
    return seq_array

def reference_to_array(reference, max_len):
    # Convert string based reference DNA sequences to N x 4 array (int encoding)
    assert max_len is not None
    shift = int((max_len - len(reference))/2)
    ref = np.full((4, max_len), np.nan, dtype=np.float32)
    indices = [hot_degenerate_base_mapping[base] for base in reference]
    indices = np.array(indices)
    ref[:, shift:shift+indices.shape[0]] = indices.transpose()
    return ref[:, np.newaxis, :]

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
    probs = refs.sum(axis=-1)[..., np.newaxis]
    wins = np.logical_and(seqs == refs,  seqs != 0)
    scores = wins / probs

    zero_mask = np.logical_or(np.isnan(refs), np.isnan(seqs))
    indel_mask = np.logical_or(seqs == 6, refs == 6)
    scores[indel_mask] = 0.01 # Add small smoothing factor for multiplication chains
    scores[zero_mask] = 0.0
    scores = scores.sum(axis=-1)

    if len(scores.shape) == 4:
        d, r, E, B = scores.shape
        final_scores = np.zeros((d, r, E, B), dtype=np.float32)
        slices = [np.pad(scores[..., i:], ((0, 0), (0,0), (0, 0), (0,i))) for i in range(max_run)]
    elif len(scores.shape) == 3:
        a, E, B= scores.shape
        final_scores = np.zeros((a, E, B), dtype=np.float32)
        slices = [np.pad(scores[..., i:], ((0, 0), (0,0), (0,i))) for i in range(max_run)]
    else:
        E, B= scores.shape
        final_scores = np.zeros((E, B), dtype=np.float32)
        slices = [np.pad(scores[..., i:], ((0, 0), (0,i))) for i in range(max_run)]

    for run_len in range(1, max_run + 1):
        result_fw = np.prod(np.stack(slices[:run_len], axis=0), axis=0)
        result_rv = np.prod(np.stack(slices[-run_len:], axis=0), axis=0)
        final_scores += result_fw + result_rv

    return final_scores

def find_best_rolls_batch(seqs, refs):
    # Parameters
    max_shift = min(50, seqs.shape[2] // 2)
    provided_range = np.arange(-max_shift, max_shift + 1)
    n_rolls = len(provided_range)
    n_strands, n_seqs, seq_len, seq_dim = seqs.shape

    # Precompute rolled sequences in one big array
    rolled_all = np.empty((n_strands, n_rolls, n_seqs, seq_len, seq_dim), dtype=seqs.dtype)
    for idx, shift in enumerate(provided_range):
        rolled_all[:,idx] = np.roll(seqs, shift=shift, axis=2)

    adjacency_matrix = compute_adjacency_score(rolled_all, refs[:, np.newaxis], max_run=10)
    adjacency_score = adjacency_matrix.sum(axis=(-1))

    # Pick best roll per sequence
    direction = np.argmax(np.mean(adjacency_score, axis=1), 0)

    top_arrays = adjacency_score[direction, :, np.arange(direction.shape[0])]
    if hasattr(top_arrays, "get"):  # CuPy array
        arr = top_arrays.get()  # move to NumPy
        smoothed = gaussian_filter1d(arr, axis=-1, sigma=1)
        smoothed = np.asarray(smoothed) # send back to CuPy
    else:
        # Run with numpy only
        smoothed = gaussian_filter1d(top_arrays, axis=-1, sigma=1)

    best_rolls_idx = np.argmax(smoothed, axis=-1)
    best_rolls = provided_range[best_rolls_idx]  # shape: (2, n_seqs)

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
        rolled_batch = np.roll(batch_array[indices], shift=shift, axis=1)  # axis=1 assumes time or sequence axis
        rolled[indices] = rolled_batch

    return rolled

