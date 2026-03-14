import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
from pathlib import Path

from dnabarmap.align_actions import *
from dnabarmap.utils import (read_fastq, read_fastqgz, write_full_fastq,
                             degenerate_nucleotide_mapping, reverse_complement,
                             nuc_dict)

MEGA_BATCH_SIZE = 100_000  # Max sequences to load/process at a time
ALIGN_MATCH_SCORE = 3
ALIGN_MISMATCH_PENALTY = -3
ALIGN_GAP_PENALTY = -4

def decode_alignment(sequence, reference=None, extra=0):
    """Convert one-hot encoded sequence array or alignment back to nucleotide sequence."""
    sequences = []
    if reference is not None:
        sequences.append(reference[0,0,0])
    sequences.append(sequence)
    decoded_sequences = []
    for input in sequences:
        result = []
        for row in input:
            key = tuple(int(i) if not np.isnan(i) else 0 for i in row)
            nucleotide = degenerate_nucleotide_mapping.get(key, '-')
            result.append(nucleotide)
        decoded_sequences.append(''.join(result))

    if reference is not None:
        nonred_ref = decoded_sequences[0][:]
        decoded_sequences[0] = ''.join([val for i,val in enumerate(nonred_ref) if val != '-'])
        decoded_sequences[1] = ''.join([val for i,val in enumerate(decoded_sequences[1]) if nonred_ref[i] != '-'])
        decoded_sequences[0] = decoded_sequences[0].replace('-', 'N')
        decoded_sequences[1] = decoded_sequences[1].replace('-', 'N')
    else:
        decoded_sequences[0] = decoded_sequences[0].replace('-', 'N')

    return decoded_sequences


def _extract_coarse_window(sequence, roll, barcode_len, extra):
    """Extract a fixed-width window around the coarse roll position."""
    start = int(roll) - int(extra)
    end = int(roll) + int(barcode_len) + int(extra)

    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    start = max(0, start)
    end = min(len(sequence), end)

    window = sequence[start:end]
    if left_pad:
        window = 'N' * left_pad + window
    if right_pad:
        window = window + 'N' * right_pad
    return window


def _align_window_to_template(window, barcode_template, template_allowed):
    """
    Semiglobally align a coarse barcode window to the degenerate template.
    Returns a template-length normalized barcode plus query start/end.
    """
    template_len = len(barcode_template)
    window_len = len(window)
    neg_inf = -10 ** 9

    scores = np.full((template_len + 1, window_len + 1), neg_inf, dtype=np.int32)
    trace = np.zeros((template_len + 1, window_len + 1), dtype=np.int8)

    scores[0, :] = 0  # free query prefix
    for i in range(1, template_len + 1):
        scores[i, 0] = scores[i - 1, 0] + ALIGN_GAP_PENALTY
        trace[i, 0] = 1

    for i in range(1, template_len + 1):
        allowed = template_allowed[i - 1]
        for j in range(1, window_len + 1):
            query_base = window[j - 1]
            diag = scores[i - 1, j - 1] + (
                ALIGN_MATCH_SCORE if query_base in allowed else ALIGN_MISMATCH_PENALTY
            )
            up = scores[i - 1, j] + ALIGN_GAP_PENALTY
            left = scores[i, j - 1] + ALIGN_GAP_PENALTY

            best = diag
            step = 0
            if up > best:
                best = up
                step = 1
            if left > best:
                best = left
                step = 2

            scores[i, j] = best
            trace[i, j] = step

    end_j = int(np.argmax(scores[template_len, :]))
    score = int(scores[template_len, end_j])
    i = template_len
    j = end_j
    normalized = []

    while i > 0:
        step = trace[i, j] if j > 0 else 1
        if step == 0:
            normalized.append(window[j - 1])
            i -= 1
            j -= 1
        elif step == 1:
            normalized.append('N')
            i -= 1
        else:
            j -= 1

    normalized.reverse()
    return ''.join(normalized), j, end_j, score


def _normalize_barcode_window(window, barcode_template, extra, template_allowed):
    """
    Normalize the center barcode to template coordinates while preserving the
    same amount of flanking context used by the original extractor.
    """
    normalized, start_j, end_j, score = _align_window_to_template(
        window, barcode_template, template_allowed)

    left_context = window[max(0, start_j - extra):start_j]
    right_context = window[end_j:min(len(window), end_j + extra)]

    if len(left_context) < extra:
        left_context = 'N' * (extra - len(left_context)) + left_context
    if len(right_context) < extra:
        right_context = right_context + 'N' * (extra - len(right_context))

    return left_context + normalized + right_context, score


def initialize_sequences(sequences, barcode_template, data,
                         synthetic_data_available, seq_limit_for_debugging, batch_size, **kwargs):
    sequence_lengths = [len(i) for i in sequences]
    max_len = int(np.quantile(sequence_lengths, 0.9))

    # Initialize top and bottom seq arrays and top reference array
    sequences_B = [reverse_complement(i) for i in sequences]
    sequences_A = [s[:max_len] for s in sequences]
    sequences_B = [rs[:max_len] for rs in sequences_B]

    # Initialize arrays
    reference_array = reference_to_array(barcode_template)
    seq_A_array = sequences_to_array(sequences_A, max_len)
    seq_B_array = sequences_to_array(sequences_B, max_len)
    seq_stacked = np.stack([seq_A_array, seq_B_array], axis=0)

    # Score top and bottom strand alignments to orient and approximately position sequences
    score_array = np.zeros((2, seq_stacked.shape[1]))
    directions = np.zeros(seq_stacked.shape[1], dtype=np.int32)
    best_rolls = np.zeros(seq_stacked.shape[1], dtype=np.int32)
    for batch_idx in range(0, seq_stacked.shape[1], batch_size):
        batch_end = min(batch_idx + batch_size, seq_stacked.shape[1])

        # Find best roll and score for both strands simultaneously
        rolls, sub_directions = find_best_rolls_batch(
            seq_stacked[:, batch_idx:batch_end],
            reference_array)

        directions[batch_idx:batch_end] = sub_directions
        best_rolls[batch_idx:batch_end] = rolls

    # Gather sequences corresponding to best strand
    batch_idxs = np.arange(score_array.shape[1])
    best_sequences = seq_stacked[directions.astype(np.int32), batch_idxs]
    best_sequences = roll_batch(best_sequences, best_rolls.astype(np.int32))

    if synthetic_data_available:
        print(f'If using synthetic data, number of incorrectly oriented sequences: {directions.astype(np.int32).sum()}')
        report_alignment_result(best_sequences[:, :reference_array.shape[-2]], reference_array, data, seq_limit_for_debugging,
                                range(best_sequences.shape[0]))

    return best_sequences, directions, best_rolls


def report_alignment_result(best_sequences, reference_array, data, seq_limit_for_debugging, indices, plot=False, extra=0):
    # Print alignment to true barcode and barcode reference
    results = []
    decoded_sequences = []
    for test_idx in indices:
        decoded_reference, decoded_sequence = decode_alignment(best_sequences[test_idx], reference_array[np.newaxis], extra=extra)
        decoded_sequences.append(decoded_sequence)
        test_seq = data.true_barcode.to_list()[:seq_limit_for_debugging][test_idx]

        score = sum([test_seq[i] == decoded_sequence[i] for i in range(len(test_seq))])
        print("Alignment score: ", score, "\nAlignment suggestion:\n",
              'Refnc:',''.join([i for i in decoded_reference]),
              '\n','Align:', ''.join([i for i in decoded_sequence]),
              '\n','True_:', data.true_barcode.to_list()[:seq_limit_for_debugging][test_idx])

        results.append(score)

    if plot:
        sns.histplot(results, bins=20)
        plt.show()


def count_total_sequences(input_fn):
    """Count total sequences in a file without loading them all into memory."""
    if input_fn.endswith('.pkl'):
        data = pd.read_pickle(input_fn)
        return len(data.synthetic_sequence)
    elif input_fn.endswith('.fastq'):
        count = 0
        with open(input_fn, 'r') as f:
            for line in f:
                count += 1
        return count // 4
    elif input_fn.endswith('.fastq.gz'):
        import gzip
        count = 0
        with gzip.open(input_fn, 'rt') as f:
            for line in f:
                count += 1
        return count // 4
    else:
        raise ValueError('Input file must be either a .pkl, .fastq, or .fastq.gz file')


def load_data_chunk(input_fn, start, chunk_size, seq_limit_for_debugging):
    """Load a chunk of sequences from the input file.

    Returns sequences, headers, data, and the actual number of sequences loaded.
    """
    end = start + chunk_size

    if input_fn.endswith('.pkl'):
        data = pd.read_pickle(input_fn)
        total = len(data.synthetic_sequence)
        if seq_limit_for_debugging is not None:
            total = min(total, seq_limit_for_debugging)
        end = min(end, total)
        if start >= total:
            return None, None, None, 0
        sequences = data.synthetic_sequence.to_list()[start:end]
        headers = None
        return sequences, headers, data, len(sequences)

    elif input_fn.endswith('.fastq'):
        sequences = []
        headers = []
        with open(input_fn, 'r') as f:
            seq_idx = 0
            line_in_record = 0
            current_header = None
            current_seq = None
            for line in f:
                line = line.strip()
                if line_in_record == 0:
                    current_header = line
                elif line_in_record == 1:
                    current_seq = line
                elif line_in_record == 3:
                    if seq_idx >= start and seq_idx < end:
                        sequences.append(current_seq)
                        headers.append(current_header)
                    seq_idx += 1
                    if seq_limit_for_debugging is not None and seq_idx >= seq_limit_for_debugging:
                        break
                    if seq_idx >= end:
                        break
                line_in_record = (line_in_record + 1) % 4
        return sequences, headers, None, len(sequences)

    elif input_fn.endswith('.fastq.gz'):
        import gzip
        sequences = []
        headers = []
        with gzip.open(input_fn, 'rt') as f:
            seq_idx = 0
            line_in_record = 0
            current_header = None
            current_seq = None
            for line in f:
                line = line.strip()
                if line_in_record == 0:
                    current_header = line
                elif line_in_record == 1:
                    current_seq = line
                elif line_in_record == 3:
                    if seq_idx >= start and seq_idx < end:
                        sequences.append(current_seq)
                        headers.append(current_header)
                    seq_idx += 1
                    if seq_limit_for_debugging is not None and seq_idx >= seq_limit_for_debugging:
                        break
                    if seq_idx >= end:
                        break
                line_in_record = (line_in_record + 1) % 4
        return sequences, headers, None, len(sequences)

    else:
        raise ValueError('Input file must be either a .pkl, .fastq, or .fastq.gz file')


def load_data(input_fn, seq_limit_for_debugging, batch_size):
    """Load all data at once (kept for backward compatibility)."""
    if input_fn.endswith('.fastq'):
        sequences, headers = read_fastq(input_fn, seq_limit_for_debugging)
        data = None
    elif input_fn.endswith('.fastq.gz'):
        sequences, headers = read_fastqgz(input_fn, seq_limit_for_debugging)
        data = None
    elif input_fn.endswith('.pkl'):
        data = pd.read_pickle(input_fn)
        if seq_limit_for_debugging is None:
            seq_limit_for_debugging = len(data.synthetic_sequence)
        assert batch_size <= seq_limit_for_debugging

        if seq_limit_for_debugging > 0:
            sequences = data.synthetic_sequence.to_list()[:seq_limit_for_debugging]
        else:
            sequences = data.synthetic_sequence.to_list()
            seq_limit_for_debugging = len(sequences)
        headers = None
    else:
        raise ValueError('Input file must be either a .pkl or .fastq file')

    return sequences, headers, data, seq_limit_for_debugging


def align(input_fn, output_fn, reoriented_fn, seq_limit_for_debugging, batch_size, barcode_template,
          synthetic_data_available, extra,
          **kwargs):

    assert os.path.exists(input_fn)
    if synthetic_data_available:
        assert input_fn.endswith('.pkl')

    # Count total sequences to process
    total_sequences = count_total_sequences(input_fn)
    if seq_limit_for_debugging is not None:
        total_sequences = min(total_sequences, seq_limit_for_debugging)

    print(f'Total sequences to process: {total_sequences}')
    print(f'Processing in chunks of {MEGA_BATCH_SIZE}')

    reference_array = reference_to_array(barcode_template)[np.newaxis]

    # Process in mega-batches of MEGA_BATCH_SIZE
    all_passed_seqs = []
    all_directions = []
    global_offset = 0

    for chunk_start in range(0, total_sequences, MEGA_BATCH_SIZE):
        chunk_end = min(chunk_start + MEGA_BATCH_SIZE, total_sequences)
        chunk_limit = chunk_end - chunk_start
        print(f'\n--- Processing chunk {chunk_start}-{chunk_end} ({chunk_limit} sequences) ---')

        # Load this chunk
        sequences, headers, data, n_loaded = load_data_chunk(
            input_fn, chunk_start, MEGA_BATCH_SIZE, seq_limit_for_debugging)

        if n_loaded == 0:
            break

        # Initialize (orient + roll) sequences for this chunk
        sequence_array, directions, best_rolls = initialize_sequences(
            sequences, barcode_template, data,
            synthetic_data_available, chunk_limit, batch_size)

        # Score and decode
        scores = []
        for i in range(0, sequence_array.shape[0], batch_size):
            batch_seq = sequence_array[i:i + batch_size, :reference_array.shape[2]]
            batch_ref = reference_array.copy()
            score = score_sequences_simple(batch_seq, batch_ref)

            score = (score > 0).astype(np.int32).transpose((2, 1, 0, 3))
            scores.append(score.sum(axis=-1))
        scores = np.concatenate(scores, axis=0)

        threshold = 0
        passing_idxs = np.where(scores > threshold)[0]
        length = reference_array.shape[-2]
        template_allowed = [set(nuc_dict[base]) for base in barcode_template]
        for i in passing_idxs:
            oriented_seq = sequences[i] if directions[i] == 0 else reverse_complement(sequences[i])
            coarse_window = _extract_coarse_window(oriented_seq, best_rolls[i], length, extra)
            decoded_seq, _ = _normalize_barcode_window(
                coarse_window,
                barcode_template,
                extra,
                template_allowed)
            # Use global index so write_full_fastq can find the right record
            all_passed_seqs.append((int(i + global_offset), decoded_seq))

        all_directions.append(directions)
        global_offset += n_loaded

        print(f'Chunk done: {len(passing_idxs)} sequences passed (cumulative: {len(all_passed_seqs)})')

        # Free memory
        del sequences, sequence_array, scores
        if data is not None and not synthetic_data_available:
            del data

    # Concatenate all directions
    all_directions = np.concatenate(all_directions, axis=0)

    # Save all alignments
    print(f'\nWriting {len(all_passed_seqs)} total passing sequences...')
    write_full_fastq(all_passed_seqs, all_directions, output_fn, input_fn, reoriented_fn)


def cli():
    global MEGA_BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. If not set, derived from input filename.')
    parser.add_argument('--barcode_template', type=str, required=True,
                        help='Reference degenerate barcode to align sequences to')

    # Set debugging/optimization parameters
    parser.add_argument('--seq_limit_for_debugging', type=int, default=None,
                        help='Filter dataset to subset for debugging')
    parser.add_argument('--synthetic_data_available', default=False, action='store_true',
                        help='Compare alignments to synthetic data or true values')

    # Set alignment parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--mega_batch_size', type=int, default=100_000,
                        help='Max sequences to load into memory at once (default: 100000)')
    parser.add_argument('--extra', type=int, default=10,
                        help='Number of bases of context to keep on each side of the '
                             'aligned barcode for clustering (default: 10)')

    all_args = parser.parse_known_args()
    args = all_args[0]

    # Allow overriding the mega batch size via CLI
    MEGA_BATCH_SIZE = args.mega_batch_size

    # Log processing speed metrics for optimization
    if args.synthetic_data_available:
        import cProfile, pstats, io
        from pstats import SortKey

        pr = cProfile.Profile()
        pr.enable()

    # Derive output_dir if not explicitly provided
    if args.output_dir is None:
        args.barcode_directory = args.input_fn.split('/barcode')[-1].split('/')[0].split('_')[0]
        args.barcode_directory = 'sample' if args.barcode_directory == '' else args.barcode_directory
        args.output_dir = f'{args.barcode_directory}/'

    # Ensure trailing slash for consistency
    if not args.output_dir.endswith('/'):
        args.output_dir += '/'

    args.cluster_dir = args.output_dir + 'clusters/'
    args.consensus_dir = args.output_dir + 'consensus/'
    args.aligned_dir = args.output_dir + 'aligned/'
    # Strip all common sequence extensions, then add the output suffix
    input_basename = path.basename(args.input_fn)
    for ext in ['.fastq.gz', '.fastq', '.pkl']:
        if input_basename.endswith(ext):
            input_stem = input_basename[:-len(ext)]
            break
    else:
        input_stem = input_basename

    args.output_fn = args.aligned_dir + input_stem + '_barcodes.fasta'
    args.reoriented_fn = args.aligned_dir + input_stem + '_reoriented.fastq'

    print(f'Input: {args.input_fn}')
    print(f'Output dir: {args.output_dir}')
    print(f'Barcode output: {args.output_fn}')

    makedirs(args.cluster_dir + 'barcodes/', exist_ok=True)
    makedirs(args.cluster_dir + 'full_seqs/', exist_ok=True)
    makedirs(args.aligned_dir, exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)
    makedirs('DNABARMAP_outputs', exist_ok=True)

    # Run alignment
    align(**vars(args))

    if args.synthetic_data_available:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(50)
        print(s.getvalue())

if __name__ == '__main__':
    cli()
