import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
from pathlib import Path

from dnabarmap.align_actions import *
from dnabarmap.utils import read_fastq, read_fastqgz, write_full_fastq, degenerate_nucleotide_mapping, reverse_complement

MEGA_BATCH_SIZE = 100_000  # Max sequences to load/process at a time

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
    best_rolls = np.zeros(seq_stacked.shape[1])
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

    return best_sequences, directions


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


def count_total_sequences(input_fq):
    """Count total sequences in a file without loading them all into memory."""
    if input_fq.endswith('.pkl'):
        data = pd.read_pickle(input_fq)
        return len(data.synthetic_sequence)
    elif input_fq.endswith('.fastq'):
        count = 0
        with open(input_fq, 'r') as f:
            for line in f:
                count += 1
        return count // 4
    elif input_fq.endswith('.fastq.gz'):
        import gzip
        count = 0
        with gzip.open(input_fq, 'rt') as f:
            for line in f:
                count += 1
        return count // 4
    else:
        raise ValueError('Input file must be either a .pkl, .fastq, or .fastq.gz file')


def load_data_chunk(input_fq, start, chunk_size, seq_limit_for_debugging):
    """Load a chunk of sequences from the input file.

    Returns sequences, headers, data, and the actual number of sequences loaded.
    """
    end = start + chunk_size

    if input_fq.endswith('.pkl'):
        data = pd.read_pickle(input_fq)
        total = len(data.synthetic_sequence)
        if seq_limit_for_debugging is not None:
            total = min(total, seq_limit_for_debugging)
        end = min(end, total)
        if start >= total:
            return None, None, None, 0
        sequences = data.synthetic_sequence.to_list()[start:end]
        headers = None
        return sequences, headers, data, len(sequences)

    elif input_fq.endswith('.fastq'):
        sequences = []
        headers = []
        with open(input_fq, 'r') as f:
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

    elif input_fq.endswith('.fastq.gz'):
        import gzip
        sequences = []
        headers = []
        with gzip.open(input_fq, 'rt') as f:
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


def load_data(input_fq, seq_limit_for_debugging, batch_size):
    """Load all data at once (kept for backward compatibility)."""
    if input_fq.endswith('.fastq'):
        sequences, headers = read_fastq(input_fq, seq_limit_for_debugging)
        data = None
    elif input_fq.endswith('.fastq.gz'):
        sequences, headers = read_fastqgz(input_fq, seq_limit_for_debugging)
        data = None
    elif input_fq.endswith('.pkl'):
        data = pd.read_pickle(input_fq)
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


def align(input_fq, output_fn, reoriented_fn, seq_limit_for_debugging, batch_size, barcode_template,
          synthetic_data_available, extra,
          **kwargs):

    assert os.path.exists(input_fq)
    if synthetic_data_available:
        assert input_fq.endswith('.pkl')

    # Count total sequences to process
    total_sequences = count_total_sequences(input_fq)
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
            input_fq, chunk_start, MEGA_BATCH_SIZE, seq_limit_for_debugging)

        if n_loaded == 0:
            break

        # Initialize (orient + roll) sequences for this chunk
        sequence_array, directions = initialize_sequences(
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
        for i in passing_idxs:
            length = reference_array.shape[-2]
            final_seq = np.concatenate((sequence_array[i, -extra:], sequence_array[i, :length+extra]))
            decoded_seq = decode_alignment(final_seq)[0]
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
    write_full_fastq(all_passed_seqs, all_directions, output_fn, input_fq, reoriented_fn)


def cli():
    global MEGA_BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fq', type=str, default=None, required=True)
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


    args.barcode_directory = 'barcode_' + args.input_fq.split('/barcode')[-1].split('/')[0].split('_')[0]
    args.barcode_directory = 'sample' if args.barcode_directory == '' else args.barcode_directory
    args.output_dir = f'temp/{args.barcode_directory}/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'
    args.aligned_dir = args.output_dir + '/aligned/'
    args.extra = 3

    args.output_fn = args.aligned_dir + path.basename(Path(args.input_fq)).replace('.pkl', '_barcodes.fasta').replace('.fastq', '_barcodes.fasta')
    args.reoriented_fn = args.aligned_dir + path.basename(Path(args.input_fq)).replace('.fastq', '_reoriented.fastq')

    makedirs(args.cluster_dir+'/barcodes/', exist_ok=True)
    makedirs(args.cluster_dir+'/full_seqs/', exist_ok=True)
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