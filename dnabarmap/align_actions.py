from collections import defaultdict

from dnabarmap.utils import hot_degenerate_base_mapping, import_cupy_numpy
import numpy as numpy # Needed for insert functions
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


def score_sequences(sequence_array, reference_array):
    # Score sequences by rewarding alignments that have long stretches of adjacent matches
    ref_mask = (reference_array != 6) & (reference_array != 0)
    seq_mask = sequence_array != 0
    ref_array = reference_array * ref_mask
    seq_array = sequence_array * seq_mask

    valid_indices = np.logical_and(~np.isnan(ref_array).any(axis=-1), ~np.isnan(seq_array).any(axis=-1))
    score = compute_adjacency_score(sequence_array, reference_array, max_run=5)
    score = np.where(valid_indices, score, np.nan)

    return score

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

def cached_intervals(seq_len, min_len, a, b):
    # full upper‐triangular once, keep as cache for efficiency
    starts, ends = np.triu_indices(seq_len + 1, k=min_len)
    mask = (ends + abs(b) <= seq_len) & (starts - abs(a) >= 0) & (ends != starts)
    return starts[mask], ends[mask]

def generate_intervals(
    seq_len,
    min_len,
    a, b,
    valid_bound=None):
    # Make intervals of possible pairwise positional shifts

    starts, ends = cached_intervals(seq_len, min_len, a, b)
    batch_allowed = valid_bound.any(axis=0)  # shape: (seq_len+1,)
    keep = batch_allowed[starts] & batch_allowed[ends]

    return starts[keep], ends[keep]

def prepare_cumsums(arr, A, B, relevant_range):
    # Calculate the sum of shifts to identify regions where overlap improves
    a = int(relevant_range[A]); b = int(relevant_range[B])
    sub_center = arr[A, B]
    batch_size, seq_len = sub_center.shape
    cumsum_center = np.zeros((batch_size, seq_len + 1))
    cumsum_center[:, 1:] = np.cumsum(sub_center, axis=-1)

    return cumsum_center, a, b

def best_single_interval(arr, A, B, relevant_range, valid, valid_bound, min_len=1):
    # Extract the best combination of pairwise shifts that improves the score
    batch_size, seq_len = valid.shape

    cumsum_center, a, b = prepare_cumsums(arr, A, B, relevant_range)
    starts, ends = generate_intervals(seq_len, min_len, a, b, valid_bound)
    interval_sums = np.subtract(cumsum_center[:, ends], cumsum_center[:, starts])

    vb_t = np.transpose(valid_bound, (1, 0))  # shape: (L+1, B)
    valid_starts = vb_t[starts].T  # shape: (B, num_intervals)
    valid_ends = vb_t[ends].T

    scores = np.full((batch_size, starts.shape[0]), -np.inf)
    raw_score = interval_sums
    valid = valid_starts & valid_ends
    np.copyto(scores, raw_score, where=valid)

    # Return top `max_patience` per batch
    best_idx = np.argmax(scores, axis=1)   # shape (batch_size,)
    starts_out = starts[best_idx]
    ends_out   = ends[best_idx]
    gains_out  = scores[np.arange(batch_size), best_idx]
    # Return as 2D arrays with trailing dim = 1 for consistency
    return (starts_out[:,None],
            ends_out[:,None],
            gains_out[:,None])

def get_precise_topk(scores: np.ndarray, patience: np.ndarray) -> np.ndarray:
    # Precise (tie breaking) of top k wins for tradeoff of efficiency and accuracy
    B, L = scores.shape
    max_patience = int(patience.max())

    topk_part = np.argpartition(scores, -max_patience, axis=1)[:, -max_patience:]  # shape (B, max_k)

    # if int(max_patience) > 0:
    #     topk_part = np.argpartition(scores, -max_patience, axis=1)[:, -max_patience:]  # shape (B, max_k)
    # else:
    #     topk_part = scores.max(axis=1)
    best_idx = np.empty(B, dtype=np.int32)

    # Stage 2: for each unique k, do a tiny stable sort on just those max_k values
    for k in np.unique(patience):
        rows = np.where(patience == k)[0]  # which rows need “kᵢ = k”
        part = topk_part[rows]  # shape (G, max_k)
        sub = scores[rows]  # shape (G, L)
        vals = np.take_along_axis(sub, part, axis=1)  # shape (G, max_k)

        # stable descending sort of those k‑candidates
        order = np.argsort(-vals, axis=1, kind="stable")  # shape (G, max_k)

        # pick the (k-1)-th element in that sorted list
        pick = order[:, k-1]                             # shape (G,)

        # map back to original column indices
        best_idx[rows] = part[np.arange(len(rows)), pick]

    return best_idx

def expand_bounds(mask, n):
    expanded = np.zeros_like(mask)
    n = int(n)
    for shift in range(-n, n+1):
        if shift < 0:
            expanded[:, :shift] |= mask[:, -shift:]
        elif shift > 0:
            expanded[:, shift:] |= mask[:, :-shift]
        else:
            expanded |= mask
    return expanded


def find_suggestions(seqs, refs, patience, valid_bound):
    # Wrapper to compare sequences and references and make suggestions for where indels should be made
    max_shifts = [1,2,3]
    ps = [(1/i)**2 for i in max_shifts]
    ps = [i/sum(ps) for i in ps]
    max_shift = np.random.choice(max_shifts, p=ps, size=1)[0]
    valid_bound = expand_bounds(valid_bound, max_shift)
    relevant_range = np.arange(-int(max_shift), int(max_shift) + 1, dtype=np.int32)

    n_rolls = len(relevant_range)
    n_seqs, seq_len, _ = seqs.shape
    center_idx = n_rolls // 2

    result = np.zeros((n_rolls, n_seqs, seq_len))
    valid_indices = ~np.isnan(refs).any(axis=-1)

    for idx, shift in enumerate(relevant_range):
        rolled = np.roll(seqs, shift=shift, axis=1)
        sc = score_sequences_simple(rolled, refs)
        sc = np.nan_to_num(sc, nan=0)
        result[idx] = np.roll(sc, -shift, axis=-1)

    result_mask = np.isnan(result)
    result[result_mask] = 0

    diff = result - result[center_idx]
    diff[:, ~valid_indices] = 0.0
    all_pairwise_sums = (diff[:, None] + diff[None, :])/2

    pair_indices = [(i, j) for i in range(n_rolls) for j in range(n_rolls) if not (i == j == center_idx)]
    total_candidates = len(pair_indices)
    all_suggestions = np.full((n_seqs, total_candidates, 5), -np.inf)

    for idx, (i, j) in enumerate(pair_indices):
        s1s, s2s, gains = best_single_interval(all_pairwise_sums, i, j, relevant_range, valid_indices,
                                               valid_bound, min_len=0)
        for batch_idx in range(n_seqs):
            start = idx
            end = (idx + 1)
            all_suggestions[batch_idx, start:end, 0] = s1s[batch_idx]
            all_suggestions[batch_idx, start:end, 1] = s2s[batch_idx]
            all_suggestions[batch_idx, start:end, 2] = gains[batch_idx]
            all_suggestions[batch_idx, start:end, 3] = relevant_range[i]
            all_suggestions[batch_idx, start:end, 4] = relevant_range[j]

        # Select the k-th best suggestion per sequence
        scores = all_suggestions[:, :, 2]  # shape (B, N)
        best_idx = get_precise_topk(scores, np.clip(patience, 0, all_suggestions[:,:,2].shape[1]))  # shape (B,)
        best = all_suggestions[np.arange(n_seqs), best_idx]

    return best


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
        smoothed = gaussian_filter1d(arr, axis=-1, sigma=2)
        smoothed = np.asarray(smoothed) # send back to CuPy
    else:
        # Run with numpy only
        smoothed = gaussian_filter1d(top_arrays, axis=-1, sigma=2)

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


###

def remove_excess_nans(array, count, strategy='all'):
    is_4d = array.shape[-1] == 4
    batch_size, seq_len = array.shape[:2]

    # Pre-allocate result array
    new_shape = (3, batch_size, seq_len - count) + ((array.shape[-1],) if is_4d else ())
    result = np.empty(new_shape)

    for j in range(batch_size):
        # Find NaN indices efficiently
        nan_mask = np.isnan(array[j])
        nan_indices = (np.where(np.any(nan_mask, axis=-1))[0] if is_4d
                       else np.where(nan_mask)[0])

        if len(nan_indices) < count:
            raise ValueError(f"Not enough NaNs to remove in sample {j}: needed {count}, found {len(nan_indices)}")

        # Three removal strategies
        if strategy == 'right':
            indices_to_remove = [
                nan_indices[-count:],  # Remove from right
            ]
        elif strategy == 'center':
            indices_to_remove = [
                nan_indices[:count],  # Remove from center
            ]
        elif strategy == 'left':
            indices_to_remove = [nan_indices[:count]  # Remove from left (will be rolled)
            ]
        else:
            indices_to_remove = [
                nan_indices[-count:],  # Remove from right
                nan_indices[:count],  # Remove from center
                nan_indices[:count]  # Remove from left (will be rolled)
            ]

        for strategy_idx, indices in enumerate(indices_to_remove):
            removed = np.delete(array[j], indices, axis=0)
            # Apply roll only for left strategy
            if strategy_idx == 2:
                removed = np.roll(removed, shift=-count, axis=0)
            result[strategy_idx, j] = removed

    return result


def remove_nans_and_align(seq_, ref_, target_len, strategy='all'):
    seq, ref = seq_.copy(), ref_.copy()

    # Early return if already correct size
    if seq.shape[0] == ref.shape[0] == target_len:
        return seq[np.newaxis], ref[np.newaxis]

    # Calculate length differences
    diffs = [arr.shape[0] - target_len for arr in [seq, ref]]
    seq_diff, ref_diff = diffs

    # Remove excess NaNs from arrays that are too long
    arrays = [seq, ref]
    for i, (arr, diff) in enumerate(zip(arrays, diffs)):
        if diff > 0:
            arrays[i] = remove_excess_nans(arr[np.newaxis], diff, strategy=strategy)

    seq, ref = arrays

    # Ensure all arrays have 2 dimensions via broadcasting
    target_shape = ref.shape if len(ref.shape) == 4 else seq.shape # this may cause errors when batching
    for i, arr in enumerate([seq, ref]):
        if len(arr.shape) != 4:
            arrays[i] = np.broadcast_to(arr, target_shape).copy()
    seq, ref = arrays[:2]

    # Check for invalid length reductions
    if any(diff < 0 for diff in [ref_diff, seq_diff]):
        raise ValueError("Cannot handle arrays shorter than target length")

    # Find positions to remove (duplicates and mismatched gaps)
    remove_positions = _find_positions_to_remove(seq, ref)

    if remove_positions.size > 0:
        seq, ref = _process_position_removal(seq, ref, remove_positions)

    return seq, ref


def _find_positions_to_remove(seq, ref):
    # Duplicate indels (both seq and ref have gap=6)
    duplicates = np.argwhere(np.logical_and((seq == 6).all(axis=-1),
                                            (ref == 6).all(axis=-1)))

    # Mismatched gaps (one has NaN, other has gap=6)
    mismatched = np.argwhere(np.logical_or(
        np.logical_and(np.isnan(ref).all(axis=-1), (seq == 6).all(axis=-1)),
        np.logical_and(np.isnan(seq).all(axis=-1), (ref == 6).all(axis=-1))
    ))

    # Combine and deduplicate positions
    all_positions = [arr for arr in [duplicates, mismatched] if arr.size > 0]
    return np.unique(np.concatenate(all_positions), axis=0) if all_positions else np.array([])


def _process_position_removal(seq, ref, remove_positions):
    # Sort positions for removal from highest to lowest index
    keys = np.stack([-remove_positions[:, 2],
                     remove_positions[:, 1],
                     remove_positions[:, 0]])
    sorted_positions = remove_positions[np.lexsort(keys)][::-1]

    # Group positions by (s, i) coordinates
    position_groups = _group_positions_by_coordinates(sorted_positions)

    # Process each group
    for s, i, positions in position_groups:
        seq[s, i], ref[s, i] = _remove_and_insert_nans(
            seq[s, i], ref[s, i], positions)

    return seq, ref


def _group_positions_by_coordinates(positions):
    """Group positions by their (s, i) coordinates."""
    if len(positions) == 0:
        return []

    groups = []
    current_coords = positions[0][:2]
    current_group = [positions[0][2]]

    for pos in positions[1:]:
        coords = pos[:2]
        if np.array_equal(coords, current_coords):
            current_group.append(pos[2])
        else:
            groups.append((*current_coords, current_group))
            current_coords = coords
            current_group = [pos[2]]

    # Add final group
    groups.append((*current_coords, current_group))
    return groups


def _remove_and_insert_nans(seq_slice, ref_slice, positions):
    """Remove positions from slices and insert NaNs at the end of existing NaN region."""
    # Remove positions from all arrays
    arrays = [seq_slice.copy(), ref_slice.copy()]
    for i in range(len(arrays)):
        arrays[i] = np.delete(arrays[i], positions, axis=0)

    # Find where to insert NaNs (at end of existing NaN region)
    nan_mask = np.logical_and(np.isnan(arrays[0]).any(axis=-1),
                              np.isnan(arrays[1]).any(axis=-1))
    nan_positions = np.argwhere(nan_mask)
    insert_pos = nan_positions[-1][0] if nan_positions.size > 0 else arrays[0].shape[0]

    # Insert NaNs for each removed position
    try:
        GPU_available = np.cuda.runtime.getDeviceCount() # Will fail if no GPU
        # Send to CPU
        arr0 = arrays[0].get()
        arr1 = arrays[1].get()
        insert_pos = insert_pos.get()
        # Do CPU inserts
        for _ in range(len(positions)):
            arr0 = numpy.insert(arr0, insert_pos, np.nan, axis=0)
            arr1 = numpy.insert(arr1, insert_pos, np.nan, axis=0)
        # Send back to GPU
        arrays[0] = np.asarray(arr0)
        arrays[1] = np.asarray(arr1)
    except:
        for _ in range(len(positions)):
            arrays[0] = np.insert(arrays[0], insert_pos, np.nan, axis=0)
            arrays[1] = np.insert(arrays[1], insert_pos, np.nan, axis=0)

    return arrays


def shift(true_shift1, true_shift2, seq, ref, p1, p2):
    if true_shift1 > 0:
        ref = numpy.insert(ref, [p1] * true_shift1, 6, axis=0)
    elif true_shift1 < 0:
        seq = numpy.insert(seq, [p1] * abs(true_shift1), 6, axis=0)

    if true_shift2 > 0:
        ref = numpy.insert(ref, [p2] * true_shift2, 6, axis=0)
    else:
        seq = numpy.insert(seq, [p2] * abs(true_shift2), 6, axis=0)

    return ref, seq

def apply_alignment_vectorized(sequence_array, reference_array, sug, target_len):
    # Rapid alignment of seq and barcode given suggestion for indels
    results_seq, results_ref = [], []
    for idx in range(sug.shape[0]):
        s = sug[idx]
        ps = {int(s[0]):int(s[-2]), int(s[1]):int(s[-1])}
        p1, p2 = max(ps.keys()), min(ps.keys()) # do backend first to maintain alignment
        true_shift1, true_shift2 = ps[p1], ps[p2]
        seq, ref = sequence_array[idx].copy(), reference_array[idx].copy()

        # Insert NaNs for each removed position
        try:
            GPU_available = np.cuda.runtime.getDeviceCount()  # Will fail if no GPU
            # Send to CPU
            ref = ref.get()
            seq = seq.get()

            ref, seq = shift(true_shift1, true_shift2, seq, ref, p1, p2) # Do CPU inserts

            # Send back to GPU
            seq = np.asarray(seq)
            ref = np.asarray(ref)
        except:
            ref, seq = shift(true_shift1, true_shift2, seq, ref, p1, p2)

        new_seq, new_ref = remove_nans_and_align(seq, ref, target_len)

        # Apply alignment processing
        new_seq = new_seq.reshape((-1, new_seq.shape[-2], 4))
        new_ref = new_ref.reshape((-1, new_ref.shape[-2], 4))
        results_seq.append(new_seq)
        results_ref.append(new_ref)

    # find the maximum “depth” in your list
    max_depth = max(arr.shape[0] for arr in results_seq)

    padded = []
    for arr in results_seq:
        depth, L, C = arr.shape
        pad_amt = max_depth - depth

        if pad_amt > 0:
            # Duplicate the first entry `pad_amt` times
            first_entry = arr[0:1]  # shape (1, L, C)
            pad_block = np.repeat(first_entry, pad_amt, axis=0)  # shape (pad_amt, L, C)
            padded_arr = np.concatenate([arr, pad_block], axis=0)
        else:
            padded_arr = arr

        padded.append(padded_arr)

    stacked_seqs = np.stack(padded, axis=1)  # shape: (max_depth, batch_size, L, C)

    max_depth = max(arr.shape[0] for arr in results_ref)

    padded = []
    for arr in results_ref:
        depth, L, C = arr.shape
        pad_amt = max_depth - depth

        if pad_amt > 0:
            first_entry = arr[0:1]
            pad_block = np.repeat(first_entry, pad_amt, axis=0)
            padded_arr = np.concatenate([arr, pad_block], axis=0)
        else:
            padded_arr = arr

        padded.append(padded_arr)

    stacked_refs = np.stack(padded, axis=1)

    return (stacked_seqs, stacked_refs)
