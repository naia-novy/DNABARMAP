from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from dnabarmap.utils import *
np = import_cupy_numpy()
from numba import njit, prange
import numba as nb


def sequences_to_array(sequences, max_len):
    # Convert string based DNA sequences to N x 4 array (int encoding)
    assert max_len is not None
    seq_array = np.full((len(sequences), max_len, 4), np.nan, dtype=np.float32)
    for i, sequence in enumerate(sequences):
        shift = int((max_len - len(sequence)) / 2)
        indices = [hot_degenerate_base_mapping[base] for base in sequence]
        seq_array[i, shift:len(indices)+shift, :] = indices
    return seq_array

def reference_to_array(reference, max_len=None):
    # Convert string based reference DNA sequences to N x 4 array (int encoding)
    if max_len is None:
        max_len = int(len(reference)*1.25)

    shift = int((max_len - len(reference))/2)
    ref = np.full((4, max_len), np.nan, dtype=np.float32)
    indices = [hot_degenerate_base_mapping[base] for base in reference]
    indices = np.array(indices)
    ref[:, shift:shift+indices.shape[0]] = indices.transpose()
    return ref[:, np.newaxis, :]


def flatten_array(arr):
    """
    Make arr C-contiguous and return:
      - flat: array shaped (n_items, seq_len, seq_dim)
      - leading_shape: tuple of leading dims (product = n_items)
      - original_shape: arr.shape
    Assumes arr.ndim >= 2 and last two dims are (seq_len, seq_dim).
    """
    if arr.ndim < 2:
        raise ValueError("Array must have at least 2 dimensions (seq_len, seq_dim).")

    # ensure C-contiguous copy
    arr_c = np.ascontiguousarray(arr)
    shape = arr_c.shape
    seq_len = shape[-2]
    seq_dim = shape[-1]
    leading_shape = shape[:-2]
    if len(leading_shape) == 0:
        n_items = 1
    else:
        n_items = int(np.prod(leading_shape))

    flat = arr_c.reshape(n_items, seq_len, seq_dim)
    return flat, tuple(leading_shape), shape


# @njit(parallel=True)  # Uncomment when using
# def score_sequences_simple(flat_seq, flat_ref, seq_shape, n_items, ref_n, match_multiplier=1.0, indel_penalty=0.0):
#     shape = flat_seq.shape
#     seq_len = shape[-2]
#     seq_dim = shape[-1]
#
#     scores_flat = np.empty((n_items, seq_len), dtype=np.float64)
#     for idx in prange(n_items):
#         # Map sequence index to reference index
#         ref_idx = idx % ref_n
#
#         for i in range(seq_len):
#             correct = 0.0
#             total = 0.0
#             indels = 0.0
#             for d in range(seq_dim):
#                 a = flat_seq[idx, i, d]
#                 b = flat_ref[ref_idx, i, d]
#
#                 # Skip positions with NaN
#                 if np.isnan(a) or np.isnan(b):
#                     continue
#
#                 # Count total bases in seq (non-zero)
#                 if a != 0:
#                     total += a
#                     # Correct match
#                     if b != 0 and b != 6 and a == 1 and b == 1:
#                         correct += 1
#                     # Indel detected
#                     if a == 6 or b == 6:
#                         indels += 1
#
#             if total == 0:
#                 score = 0
#             else:
#                 score = (correct / total) * match_multiplier
#                 score -= indel_penalty*indels
#             scores_flat[idx, i] = score
#
#     # Reshape back to original leading dims, dropping only last dim
#     return scores_flat.reshape(seq_shape[:-1])


def build_broadcast_index_map(seq_leading, ref_leading):
    """
    Build an index map of length prod(seq_leading) mapping each flattened seq-leading
    multi-index to the corresponding flattened ref-leading index under NumPy broadcasting rules.

    Both seq_leading and ref_leading are tuples (could be empty tuples).
    Broadcasting alignment is done on the right (same as numpy).
    """
    # lengths
    Ls = len(seq_leading)
    Lr = len(ref_leading)

    # pad ref_leading on the left with ones to match length of seq_leading
    if Lr < Ls:
        padded = (1,) * (Ls - Lr) + tuple(ref_leading)
    else:
        padded = tuple(ref_leading[-Ls:]) if Lr > Ls else tuple(ref_leading)

    # sanity check: after padding, lengths must match
    assert len(padded) == Ls

    # precompute multipliers for flat indexing of the ref_padded shape
    # multiplier[j] = product of padded[j+1:]
    multipliers = [1] * Ls
    for j in range(Ls - 1, -1, -1):
        if j == Ls - 1:
            multipliers[j] = 1
        else:
            multipliers[j] = multipliers[j + 1] * padded[j + 1]

    total_items = 1 if Ls == 0 else int(np.prod(seq_leading))
    mapping = np.empty(total_items, dtype=np.int64)

    # iterate over every multi-index for seq_leading and compute corresponding ref flat index
    idx = 0
    for multi in np.ndindex(*seq_leading) if Ls > 0 else [()]:
        # compute ref_flat index:
        ref_idx = 0
        for j, coord in enumerate(multi):
            if padded[j] == 1:
                ref_coord = 0
            else:
                ref_coord = coord
            ref_idx += int(ref_coord) * int(multipliers[j])
        mapping[idx] = ref_idx
        idx += 1

    return mapping



def prepare_for_numba(seq_array, ref_array):
    """
    - seq_array: shape (..., seq_len, seq_dim)
    - ref_array: shape (maybe different leading dims ..., seq_len, seq_dim)
    Returns:
      flat_seq_int: shape (n_items, seq_len, seq_dim) dtype=int8 with NaN -> -1
      flat_ref_int: shape (ref_n, seq_len, seq_dim) dtype=int8 with NaN -> -1
      mapping: int64 array length n_items mapping seq-item -> ref-item (broadcast logic)
      out_shape: shape of desired output (seq_array.shape[:-1])
    """
    # Ensure float inputs as given; convert sentinel -1 for NaN
    seq_c = np.ascontiguousarray(seq_array)
    ref_c = np.ascontiguousarray(ref_array)

    # get shapes and flatten
    seq_shape = seq_c.shape
    seq_len = seq_shape[-2]
    seq_dim = seq_shape[-1]
    # n_items = product of leading dims (all except last two)
    n_items = int(np.prod(seq_shape[:-2])) if len(seq_shape) > 2 else 1
    flat_seq = seq_c.reshape(n_items, seq_len, seq_dim)

    ref_shape = ref_c.shape
    ref_leading = ref_shape[:-2]
    ref_n = int(np.prod(ref_leading)) if len(ref_leading) > 0 else 1
    flat_ref = ref_c.reshape(ref_n, seq_len, seq_dim)

    # convert to small integers with NaN-> -1 (choose dtype suitably)
    # -- if values can be 0..6 you can safely cast to int8 after NaN mapping
    flat_seq_int = np.where(np.isnan(flat_seq), -1, flat_seq).astype(np.int8)
    flat_ref_int = np.where(np.isnan(flat_ref), -1, flat_ref).astype(np.int8)

    # mapping: for simple broadcasting where ref_leading may be smaller than seq leading dims:
    # if seq_leading = (a,b,c) and ref_leading = (a,1,c) we need per-item mapping.
    # simplest: use the mapping builder you already have (build_broadcast_index_map)
    seq_leading = seq_shape[:-2]
    mapping = build_broadcast_index_map(seq_leading, ref_leading)  # from your earlier helper

    out_shape = seq_shape[:-1]  # want to drop last dim
    return flat_seq_int, flat_ref_int, mapping.astype(np.int64), out_shape

@nb.njit(parallel=True, fastmath=True)
def score_sequences_simple(flat_seq_int, flat_ref_int, mapping,
                              match_multiplier=1.0, indel_penalty=0.0):
    # flat_seq_int: (n_items, seq_len, seq_dim) int8 with -1 sentinel
    # flat_ref_int: (ref_n, seq_len, seq_dim) int8
    n_items, seq_len, seq_dim = flat_seq_int.shape
    scores_flat = np.empty((n_items, seq_len), dtype=np.float64)

    for idx in prange(n_items):
        ref_idx = mapping[idx]   # already computed mapping

        # Common-case: seq_dim == 4 -> unroll for speed
        if seq_dim == 4:
            for i in range(seq_len):
                correct = 0.0
                total = 0.0
                indels = 0.0

                a0 = flat_seq_int[idx, i, 0]; b0 = flat_ref_int[ref_idx, i, 0]
                a1 = flat_seq_int[idx, i, 1]; b1 = flat_ref_int[ref_idx, i, 1]
                a2 = flat_seq_int[idx, i, 2]; b2 = flat_ref_int[ref_idx, i, 2]
                a3 = flat_seq_int[idx, i, 3]; b3 = flat_ref_int[ref_idx, i, 3]

                # check a0/b0
                if a0 != -1 and b0 != -1:
                    if a0 != 0:
                        total += a0
                        if b0 != 0 and b0 != 6 and a0 == 1 and b0 == 1:
                            correct += 1.0
                        if a0 == 6 or b0 == 6:
                            indels += 1.0
                # a1/b1
                if a1 != -1 and b1 != -1:
                    if a1 != 0:
                        total += a1
                        if b1 != 0 and b1 != 6 and a1 == 1 and b1 == 1:
                            correct += 1.0
                        if a1 == 6 or b1 == 6:
                            indels += 1.0
                # a2/b2
                if a2 != -1 and b2 != -1:
                    if a2 != 0:
                        total += a2
                        if b2 != 0 and b2 != 6 and a2 == 1 and b2 == 1:
                            correct += 1.0
                        if a2 == 6 or b2 == 6:
                            indels += 1.0
                # a3/b3
                if a3 != -1 and b3 != -1:
                    if a3 != 0:
                        total += a3
                        if b3 != 0 and b3 != 6 and a3 == 1 and b3 == 1:
                            correct += 1.0
                        if a3 == 6 or b3 == 6:
                            indels += 1.0

                if total == 0:
                    scores_flat[idx, i] = np.nan
                else:
                    score = (correct / total) * match_multiplier
                    score -= indel_penalty * indels
                    scores_flat[idx, i] = score

        else:
            # general case
            for i in range(seq_len):
                correct = 0.0
                total = 0.0
                indels = 0.0
                for d in range(seq_dim):
                    a = flat_seq_int[idx, i, d]
                    b = flat_ref_int[ref_idx, i, d]
                    if a == -1 or b == -1:
                        continue
                    if a != 0:
                        total += a
                        if b != 0 and b != 6 and a == 1 and b == 1:
                            correct += 1.0
                        if a == 6 or b == 6:
                            indels += 1.0
                if total == 0:
                    scores_flat[idx, i] = np.nan
                else:
                    score = (correct / total) * match_multiplier
                    score -= indel_penalty * indels
                    scores_flat[idx, i] = score

    return scores_flat




def score_sequences_wrapper(seq_array, ref_array, match_multiplier=1.0, indel_penalty=0.0):
    """
    Wrapper that prepares flat arrays and mapping and calls the Numba scoring function.
    Returns an array shaped like seq_array.shape[:-1] (i.e. drop only the last dim seq_dim).
    """
    # flatten seq_array
    flat_seq, seq_leading, seq_shape = flatten_array(seq_array)
    # flatten ref_array
    flat_ref, ref_leading, ref_shape = flatten_array(ref_array)

    # build mapping from flattened seq items -> flattened ref items
    # seq_leading and ref_leading are tuples of leading dims (maybe empty)
    # If ref has fewer leading dims, build_broadcast_index_map pads ref on left (right-alignment done inside)
    # To broadcast properly we must align from the right; build_broadcast_index_map handles that by padding on the left.
    mapping = build_broadcast_index_map(seq_leading, ref_leading)

    # ensure flat_ref is contiguous (flatten_array does this already)
    flat_ref = np.ascontiguousarray(flat_ref)
    flat_seq = np.ascontiguousarray(flat_seq)

    # call numba core
    out = score_sequences_simple(flat_seq, flat_ref, mapping, match_multiplier, indel_penalty)

    # out is shaped (n_items, seq_len) -> reshape to seq_array.shape[:-1]
    return out.reshape(seq_shape[:-1])


# def score_sequences_simple(sequence_array, reference_array, sum_sequences,
#                                 match_multiplier=1.0, indel_penalty=0.0, binary=False):
#     # Precompute constants
#     match_multiplier = abs(match_multiplier)
#     indel_penalty = -abs(indel_penalty)
#
#     # Ensure reference shape matches
#     if reference_array.ndim == sequence_array.ndim - 1:
#         reference_array = np.broadcast_to(reference_array, sequence_array.shape)
#
#     # Masks
#     ref_mask = (reference_array != 6) & (reference_array != 0)
#     seq_mask = sequence_array != 0
#
#     # Masked arrays (without NaNs)
#     ref_array = np.where(ref_mask, reference_array, 0)
#     seq_array = np.where(seq_mask, sequence_array, 0)
#
#     # Valid index mask
#     valid_indices = ~np.isnan(ref_array).any(axis=-1) & ~np.isnan(seq_array).any(axis=-1)
#
#     # Correct matches (only where ref==1 and seq==1)
#     correct = np.sum((seq_array == 1) & (ref_array == 1), axis=-1)
#
#     # Total possibilities
#     possibilities = np.clip(np.sum(ref_array, axis=-1), 1e-8, None)
#
#     # Base score
#     score = (correct / possibilities) * match_multiplier
#
#     # Indel mask
#     indel_mask = np.any((sequence_array == 6) | (reference_array == 6), axis=-1)
#
#     # Apply NaN for invalid
#     score = np.where(valid_indices, score, np.nan)
#
#     if binary:
#         score = (score > 0).astype(int)
#         if sum_sequences:
#             return np.nansum(score, axis=-1) / np.maximum(valid_indices.sum(axis=-1), 1)
#         else:
#             return score
#     else:
#         score[indel_mask] = indel_penalty
#         if sum_sequences:
#             return np.nansum(score, axis=-1) / np.maximum(valid_indices.sum(axis=-1), 1)
#         else:
#             return score


def score_sequences(sequence_array, reference_array, sum_sequences,
                    match_multiplier=1.0, indel_penalty=0.0, binary=False):
    # Score sequences by rewarding alignments that have long stretches of adjacent matches
    ref_mask = (reference_array != 6) & (reference_array != 0)
    seq_mask = sequence_array != 0
    ref_array = reference_array * ref_mask
    seq_array = sequence_array * seq_mask

    valid_indices = np.logical_and(~np.isnan(ref_array).any(axis=-1), ~np.isnan(seq_array).any(axis=-1))
    match_multiplier = abs(match_multiplier)
    indel_penalty = -abs(indel_penalty)

    correct = np.sum(np.logical_and(seq_array == ref_array, seq_array==1), axis=-1)
    probs = np.clip(np.sum(ref_array, axis=-1), a_min=1e-8, a_max=None)

    reformat = False
    if len(correct.shape) != 3:
        correct = correct[np.newaxis]
        probs = probs[np.newaxis]
        reformat = True
    score = compute_adjacency_score(correct, probs,
                                                        max_run=3, squared_scores=False)
    if reformat:
        score = score[0]

    score *= match_multiplier
    if len(reference_array.shape) == len(sequence_array.shape) -1:
        reference_array = np.broadcast_to(reference_array, sequence_array.shape)
    indel_mask = np.any(np.logical_or.reduce([
            sequence_array == 6,
            reference_array == 6]), axis=-1)

    score = np.where(valid_indices, score, np.nan)

    if binary:
        score = (score > 0).astype(int)
        if sum_sequences:
            return np.nansum(score, axis=-1) / valid_indices.sum(axis=-1)
        else:
            return score
    else:
        score[indel_mask] = indel_penalty
        if sum_sequences:
            return np.nansum(score, axis=-1) / valid_indices.sum(axis=-1)
        else:
            return score


        # # Make a view of shape (B, L-run_len+1, run_len)
        # try:
        #     # cupy version of sliding function
        #     windows = np.lib.stride_tricks.as_strided(wins, shape=run_len)
        #
        # except:
        #     # numpy version of sliding function
        #     windows = np.lib.stride_tricks.sliding_window_view(wins, window_shape=run_len, axis=-1)
def compute_adjacency_score(wins, probs, max_run, squared_scores=False):
    if len(wins.shape) == 4:
        d, E, B, L = wins.shape
        csum = np.cumsum(wins, axis=-1, dtype=int)
        csum = np.pad(csum, ((0,0), (0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)
        scores = np.zeros((d, E, B, L), dtype=float)
        inv_probs = 1.0 / np.clip(probs, 1e-8, None)

        for run_len in range(1, min(max_run, L) + 1):
            seg_sum = csum[:, :, :, run_len:] - csum[:, :, :, :-run_len]
            valid_runs = (seg_sum == run_len)

            weight = run_len ** 2 if squared_scores else run_len
            scores[:, :, :, :L - run_len + 1] += weight * valid_runs
    else:
        E, B, L = wins.shape
        csum = np.cumsum(wins, axis=-1, dtype=int)
        csum = np.pad(csum, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)

        scores = np.zeros((E, B, L), dtype=float)
        inv_probs = 1.0 / np.clip(probs, 1e-8, None)

        for run_len in range(1, min(max_run, L) + 1):
            seg_sum = csum[:, :, run_len:] - csum[:, :, :-run_len]  # shape (E,B,L-run_len+1)
            valid_runs = (seg_sum == run_len)

            weight = run_len**2 if squared_scores else run_len
            scores[:, :, :L-run_len+1] += weight * valid_runs

    scores *= inv_probs
    return scores


def cached_intervals(seq_len, min_len, a, b):
    # full upper‐triangular once, keep as cache for efficiency
    starts, ends = np.triu_indices(seq_len + 1, k=min_len)
    mask = (ends + abs(b) <= seq_len) & (starts - abs(a) >= 0)
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
    cumsum_flank = np.zeros_like(cumsum_center)
    if a * b > 0:
        cumsum_flank = cumsum_center.copy()
    else:
        diff = a + b
        if abs(a) >= abs(b):
            new_a, new_b = diff, 0
        else:
            new_a, new_b = 0, diff
        A2 = int(np.where(relevant_range == new_a)[0][0])
        B2 = int(np.where(relevant_range == new_b)[0][0])
        sub_flank = arr[A2, B2]
        cumsum_flank[:, 1:] = np.cumsum(sub_flank, axis=-1)
    return cumsum_center, cumsum_flank, a, b

def best_single_interval(arr, A, B, relevant_range, valid, valid_bound,
                         max_patience, min_len=0, indel_penalty=0.0):
    # Extract the best combination of pairwise shifts that improves the score
    batch_size, seq_len = valid.shape

    cumsum_center, cumsum_flank, a, b = prepare_cumsums(arr, A, B, relevant_range)
    starts, ends = generate_intervals(seq_len, min_len, a, b, valid_bound)

    interval_sums = np.subtract(cumsum_center[:, ends], cumsum_center[:, starts])
    left_flank_starts = np.clip(starts - abs(a), 0, seq_len)
    left_flank_ends = np.clip(starts, 0, seq_len)
    right_flank_starts = np.clip(ends, 0, seq_len)
    right_flank_ends = np.clip(ends + abs(b), 0, seq_len)

    left_flank = np.subtract(cumsum_flank[:, left_flank_ends], cumsum_flank[:, left_flank_starts])
    right_flank = np.subtract(cumsum_flank[:, right_flank_ends], cumsum_flank[:, right_flank_starts])

    flank_sums = np.add(left_flank, right_flank)

    center_mean = interval_sums
    flank_mean  = flank_sums

    vb_t = np.transpose(valid_bound, (1, 0))  # shape: (L+1, B)
    valid_starts = vb_t[starts].T  # shape: (B, num_intervals)
    valid_ends = vb_t[ends].T

    scores = np.full((batch_size, starts.shape[0]), -np.inf)
    raw_score = np.add(center_mean, flank_mean)
    pen = indel_penalty * (abs(a) + abs(b))
    valid = valid_starts & valid_ends
    np.copyto(scores, raw_score - pen, where=valid)

    # Return top `max_patience` per batch
    if max_patience == 1:
        # trivial top-1 case
        best_idx = np.argmax(scores, axis=1)   # shape (batch_size,)
        starts_out = starts[best_idx]
        ends_out   = ends[best_idx]
        gains_out  = scores[np.arange(batch_size), best_idx]
        # Return as 2D arrays with trailing dim = 1 for consistency
        return (starts_out[:,None],
                ends_out[:,None],
                gains_out[:,None])
    else:
        topk_part = np.argpartition(scores, -max_patience, axis=1)[:, -max_patience:]  # (B, K)
        top_scores = np.take_along_axis(scores, topk_part, axis=1)                      # (B, K)
        sort_order = np.argsort(-top_scores, axis=1, kind="stable")                    # (B, K)
        sorted_indices = np.take_along_axis(topk_part, sort_order, axis=1)             # (B, K)

        s1s = starts[sorted_indices]
        s2s = ends[sorted_indices]
        gains = np.take_along_axis(scores, sorted_indices, axis=1)
        return s1s, s2s, gains  # each (B, K)


def get_precise_topk(scores: np.ndarray, patience: np.ndarray) -> np.ndarray:
    # Precise (tie breaking) of top k wins for tradeoff of efficiency and accuracy
    B, L = scores.shape
    max_patience = patience.max()

    topk_part = np.argpartition(scores, -max_patience, axis=1)[:, -max_patience:]  # shape (B, max_k)
    best_idx = np.empty(B, dtype=int)

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


def find_suggestions(seqs, refs, match_multiplier, indel_penalty, patience, valid_bound):
    # Wrapper to compare sequences and references and make suggestions for where indels should be made
    max_shift = 1
    relevant_range = np.arange(-max_shift, max_shift + 1, dtype=int)

    n_rolls = len(relevant_range)
    n_seqs, seq_len, _ = seqs.shape
    center_idx = n_rolls // 2

    result = np.zeros((n_rolls, n_seqs, seq_len))
    valid_indices = ~np.isnan(refs).any(axis=-1)

    # ref_flat, ref_n = flatten_array(refs) # Flatten arrays now since numba cant handle it

    for idx, shift in enumerate(relevant_range):
        rolled = np.roll(seqs, shift=shift, axis=1)
        # rolled_flat, n_items = flatten_array(rolled) # Flatten arrays now since numba cant handle it
        # sc = score_sequences_simple(rolled_flat, ref_flat, seqs.shape, n_items, ref_n, match_multiplier, indel_penalty)
        sc = score_sequences_wrapper(rolled, refs, match_multiplier, indel_penalty)

        sc = np.nan_to_num(sc, nan=0)
        result[idx] = np.roll(sc, -shift, axis=-1)

    result_mask = np.isnan(result)
    result[result_mask] = 0

    diff = result - result[center_idx]
    diff[:, ~valid_indices] = 0.0
    all_pairwise_sums = (diff[:, None] + diff[None, :])/2

    pair_indices = [(i, j) for i in range(n_rolls) for j in range(n_rolls) if not (i == j == center_idx)]

    max_patience = 1
    total_candidates = len(pair_indices) * max_patience
    all_suggestions = np.full((n_seqs, total_candidates, 5), -np.inf)

    for idx, (i, j) in enumerate(pair_indices):
        s1s, s2s, gains = best_single_interval(all_pairwise_sums, i, j, relevant_range, valid_indices, valid_bound, max_patience,
                                               min_len=0, indel_penalty=indel_penalty)
        for batch_idx in range(n_seqs):
            start = idx * max_patience
            end = (idx + 1) * max_patience
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




def find_best_rolls_batch(seqs, refs, match_multiplier, indel_penalty):
    # Parameters
    max_shift = min(30, seqs.shape[2] // 2)
    provided_range = np.arange(-max_shift, max_shift + 1)
    n_rolls = len(provided_range)
    n_strands, n_seqs, seq_len, seq_dim = seqs.shape

    # Precompute rolled sequences in one big array
    rolled_all = np.empty((n_strands, n_rolls, n_seqs, seq_len, seq_dim), dtype=seqs.dtype)
    for idx, shift in enumerate(provided_range):
        rolled_all[:,idx] = np.roll(seqs, shift=shift, axis=2)

    # Flatten arrays now since numba cant handle it
    # rolled_flat, n_items = flatten_array(rolled_all)
    # adj_refs, ref_n = flatten_array(refs[:,np.newaxis])
    # scores = score_sequences_simple(rolled_flat, adj_refs, rolled_all.shape, n_items, ref_n, match_multiplier, indel_penalty)

    scores = score_sequences_wrapper(rolled_all, refs[:, np.newaxis], match_multiplier, indel_penalty)

    # # Score all rolls in one go — vectorized score_sequences_simple
    # scores = score_sequences_simple(rolled_all, refs[:,np.newaxis], match_multiplier, indel_penalty)

    # NaN → 0, compute cum_scores
    result_mask = np.isnan(scores)
    scores[result_mask] = 0
    # cum_scores = scores.sum(axis=-1)  # sum over seq_len

    # Compute adjacency scores (batched)
    wins = scores > 0
    probs = np.nan_to_num(refs.sum(axis=-1)[0], nan=1)[np.newaxis,np.newaxis]
    adjacency_matrix = compute_adjacency_score(wins, probs, max_run=3, squared_scores=False).sum(axis=-1)

    # Smooth across roll axis
    smoothed = gaussian_filter1d(adjacency_matrix, sigma=1, axis=1)

    # Pick best roll per sequence
    n_strands, n_rolls, n_seqs = smoothed.shape
    best_rolls_idx = np.argmax(smoothed, axis=1)  # shape: (2, n_seqs)

    # Now grab the corresponding best scores and rolls
    strand_idx = np.arange(n_strands)[:, None]  # shape: (2, 1)
    seq_idx = np.arange(n_seqs)[None, :]  # shape: (1, n_seqs)
    best_scores = smoothed[strand_idx, best_rolls_idx, seq_idx]  # shape: (2, n_seqs)
    best_rolls = provided_range[best_rolls_idx]  # shape: (2, n_seqs)

    return best_rolls, best_scores

def roll_batch(batch_array, roll_values):
    # Apply batched roll to array
    rolled = batch_array.copy()
    # Group sequence indices by their roll shift
    shift_groups = defaultdict(list)
    for idx, shift in enumerate(roll_values):
        if shift == 0:
            continue
        shift_groups[shift].append(idx)

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
    sorted_positions = remove_positions[np.lexsort((-remove_positions[:, 2],
                                                    -remove_positions[:, 1],
                                                    -remove_positions[:, 0]))][::-1]

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
    for _ in range(len(positions)):
        arrays[0] = np.insert(arrays[0], insert_pos, np.nan, axis=0)
        arrays[1] = np.insert(arrays[1], insert_pos, np.nan, axis=0)

    return arrays

def apply_alignment_vectorized(sequence_array, reference_array, sug, target_len):
    # Rapid alignment of seq and barcode given suggestion for indels
    results_seq, results_ref = [], []
    for idx in range(sug.shape[0]):
        s = sug[idx]
        ps = {int(s[0]):int(s[-2]), int(s[1]):int(s[-1])}
        p1, p2 = max(ps.keys()), min(ps.keys()) # do backend first to maintain alignment
        true_shift1, true_shift2 = ps[p1], ps[p2]
        seq, ref = sequence_array[idx].copy(), reference_array[idx].copy()

        if true_shift1 > 0:
            ref = np.insert(ref, [p1] * true_shift1, 6, axis=0)
        elif true_shift1 < 0:
            seq = np.insert(seq, [p1] * abs(true_shift1), 6, axis=0)

        if true_shift2 > 0:
            ref = np.insert(ref, [p2] * true_shift2, 6, axis=0)
        else:
            seq = np.insert(seq, [p2]*abs(true_shift2), 6, axis=0)

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
