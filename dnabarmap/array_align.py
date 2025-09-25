import argparse
from collections import deque
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from dnabarmap.align_actions import *
from dnabarmap.utils import read_fastq, write_full_fastq, degenerate_nucleotide_mapping, reverse_complement



def decode_alignment(sequence, reference=None, reduce=False, extra=5):
    """Convert one-hot encoded sequence array or alignment back to nucleotide sequence."""
    sequences = []
    if reference is not None and reduce:
        adj_ref = reference.copy()
        adj_ref[adj_ref == 6] = np.nan  # mark gaps as NaN
        mask = (~np.all(np.isnan(adj_ref), axis=-1))

        valid = np.where(mask)[0]
        first_idx = max(0, valid[0] - extra)
        last_idx = min(mask.shape[0], valid[-1] + extra + 1)
        reference[first_idx:valid[0]] = 1
        reference[valid[-1] + 1:last_idx] = 1
        reference = reference[first_idx:last_idx]
        sequence = sequence[first_idx:last_idx]
        sequences.append(reference)
    elif reference is not None:
        sequences.append(reference)

    sequences.append(sequence)
    decoded_sequences = []
    for input in sequences:
        result = []
        for row in input:
            try:
                key = tuple((row.astype(int).get()))
            except:
                key = tuple((row).astype(int))
            nucleotide = degenerate_nucleotide_mapping.get(key, '-')
            result.append(nucleotide)
        decoded_sequences.append(''.join(result))

    if reference is not None and reduce:
        nonred_ref = decoded_sequences[0][:]
        decoded_sequences[0] = ''.join([val for i,val in enumerate(nonred_ref) if val != '-'])
        decoded_sequences[1] = ''.join([val for i,val in enumerate(decoded_sequences[1]) if nonred_ref[i] != '-'])
        decoded_sequences[0] = decoded_sequences[0].replace('-', 'N')
        decoded_sequences[1] = decoded_sequences[1].replace('-', 'N')
        # decoded_sequences[1] = ''.join([v for i,v in enumerate(decoded_sequences[1]) if decoded_sequences[0][i] not in ['A', 'T', 'C', 'G']])
        # decoded_sequences[0] = ''.join([i for i in decoded_sequences[0] if i not in ['A', 'T', 'C', 'G']])

    return decoded_sequences

def initialize_sequences(sequences, barcode_template, data,
                         synthetic_data_available, seq_limit_for_debugging, buffer, batch_size, **kwargs):
    # Convert sequences to arrays and do initial approximate alignment
    buffer = max(buffer - 30, 0)
    length_mult = 3
    template_len = len(barcode_template)
    template_len = int(template_len * length_mult)
    max_len = int(3.5 * len(barcode_template)) # slightly larger so there are sufficient nans

    # Initialize top and bottom seq arrays and top reference array
    sequences_A = [i[buffer:buffer + template_len] for i in sequences]
    sequences_B = [reverse_complement(i)[buffer:buffer + template_len] for i in sequences]

    reference_array = reference_to_array(barcode_template, max_len)
    seq_A_array = sequences_to_array(sequences_A, reference_array.shape[-1])
    seq_B_array = sequences_to_array(sequences_B, reference_array.shape[-1])
    sequence_array = np.stack([seq_A_array, seq_B_array], axis=0)
    reference_array = np.repeat(reference_array[np.newaxis], len(sequence_array[1]), axis=0)
    reference_array = np.transpose(reference_array[:, :, 0], (0, 2, 1))

    seq_stacked = np.stack([seq_A_array, seq_B_array], axis=0)
    ref_stacked = np.stack([reference_array, reference_array], axis=0)

    # Score top and bottom strand alignments to orient and approximately position sequences
    score_array = np.zeros((2, sequence_array.shape[1]))
    directions = np.zeros(sequence_array.shape[1], dtype=np.int32)
    best_rolls = np.zeros(sequence_array.shape[1])
    for batch_idx in range(0, seq_stacked.shape[1], batch_size):
        batch_end = min(batch_idx + batch_size, seq_stacked.shape[1])

        # Find best roll and score for both strands simultaneously
        rolls, sub_directions = find_best_rolls_batch(
            seq_stacked[:, batch_idx:batch_end],
            ref_stacked[:, batch_idx:batch_end])

        directions[batch_idx:batch_end] = sub_directions
        best_rolls[batch_idx:batch_end] = rolls

    # Gather sequences corresponding to best strand
    batch_idxs = np.arange(score_array.shape[1])
    best_sequences = sequence_array[directions.astype(np.int32), batch_idxs]
    best_sequences = roll_batch(best_sequences, best_rolls.astype(np.int32))  # reroll best sequences

    if synthetic_data_available:
        print(f'If using synthetic data, number of incorrectly oriented sequences: {directions.astype(np.int32).sum()}')
        report_alignment_result(best_sequences, reference_array, data, seq_limit_for_debugging,
                                range(best_sequences.shape[0]))

    return best_sequences, directions


def report_alignment_result(best_sequences, reference_array, data, seq_limit_for_debugging, indices, plot=False, extra=5):
    # Print alignment to true barcode and barcode reference
    results = []
    decoded_sequences = []
    for test_idx in indices:
        decoded_reference, decoded_sequence = decode_alignment(best_sequences[test_idx], reference_array[test_idx], reduce=True, extra=extra)
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

def load_data(input_fn, seq_limit_for_debugging, batch_size):
    # Load data according to the filetype provided
    if input_fn.endswith('.fastq'):
        sequences, headers = read_fastq(input_fn, seq_limit_for_debugging)
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
          synthetic_data_available, buffer, extra,
          **kwargs):

    # Load dataset
    assert os.path.exists(input_fn)
    if synthetic_data_available:
        assert input_fn.endswith('.pkl')
    sequences, headers, data, seq_limit_for_debugging = load_data(input_fn, seq_limit_for_debugging, batch_size)

    # Initialize sequence, reference, and patience arrays
    sequence_array, directions = initialize_sequences(sequences, barcode_template, data,
                                          synthetic_data_available, seq_limit_for_debugging, buffer, batch_size)
    reference_array = reference_to_array(barcode_template, sequence_array.shape[1])
    reference_array = np.repeat(reference_array[np.newaxis], len(sequence_array), axis=0)
    reference_array = np.transpose(reference_array[:, :, 0], (0, 2, 1))

    # Convert arrays into nucleotide sequences for downstream processing
    scores = []
    for i in range(0, sequence_array.shape[0], batch_size):
        batch_seq = sequence_array[i:i + batch_size]
        batch_ref = reference_array[i:i + batch_size]
        score = score_sequences_simple(batch_seq, batch_ref)

        score = (score > 0).astype(np.int32)
        scores.append(score.sum(axis=-1))  # sum over sequence length
    scores = np.concatenate(scores, axis=0)

    threshold = 0
    passing_idxs = np.where(scores > threshold)[0]

    passed_seqs = []
    for i in passing_idxs:
        decoded_reference, decoded_seq = decode_alignment(sequence_array[i], reference_array[i], reduce=True, extra=extra)
        passed_seqs.append((i, decoded_seq))

    if synthetic_data_available:
        print('\nFinal alignment results:')
        report_alignment_result(sequence_array, reference_array, data, seq_limit_for_debugging, range(sequence_array.shape[0]), plot=True)

    # Save alignments
    write_full_fastq(passed_seqs, directions, output_fn, input_fn, reoriented_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set debugging/optimization parameters
    parser.add_argument('--seq_limit_for_debugging', type=int, default=None,
                        help='Filter dataset to subset for debugging')
    parser.add_argument('--synthetic_data_available', default=True, action='store_true',
                        help='Compare alignments to synthetic data or true values')

    # Set alignment parameters
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=0,
                        help='How many times to try next best suggestion before giving up')
    parser.add_argument('--buffer', type=int, default=40,
                        help='Expected constant region on the DNA fragment before the barcode')
    parser.add_argument('--barcode_template', type=str,
                        default='YHWSBYRVWBYMDSKWWVSBWSSWDRKMDSYMWYSKRWYDRYSKMSYDYSWVYRYKRYVR', # TATGAYHWSBYRVWBYMDSKWWVSBWSSWDRKMDSYMWYSKRWYDRYSKMSYDYSWVYRYKRYVRCGATC
                                           help='Reference degenerate barcode to align sequences to')
    parser.add_argument('--input_fn', type=str, default='./syndata/syndataA.pkl')

    args = parser.parse_args()

    # Log processing speed metrics for optimization
    if args.synthetic_data_available:
        import cProfile, pstats, io
        from pstats import SortKey

        pr = cProfile.Profile()
        pr.enable()


    args.output_fn = args.input_fn.replace('.pkl', '_barcodes.fasta').replace('.fastq', '_barcodes.fasta')
    args.filtered_fn = args.output_fn.replace('barcodes.fasta', 'filtered.fastq')

    # Run alignment
    align(**vars(args))

    if args.synthetic_data_available:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(50)
        print(s.getvalue())
