import numpy as np
import pandas as pd
import random
import argparse
from itertools import product
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from os import makedirs

from dnabarmap.generate import generate_sequence, generate_random_barcodes
from dnabarmap.utils import degenerate_map, int_to_degenerate, generate_random_mut, write_synthetic_fastq
import dnabarmap.simulate as sim

seed = 100
np.random.seed(seed)


IUPAC = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
    'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T']
}


def expanded_motif_repetition_penalty(seq, k=3):
    # 1) Build a list of sets: all possible k‑mers at each window
    obs = []
    for i in range(len(seq) - k + 1):
        codes = seq[i:i + k]
        pools = [IUPAC[c] for c in codes]
        # cartesian product → all concrete k‑mers here
        poss = [''.join(p) for p in product(*pools)]
        obs.append(poss)

    # 2) Count in how many windows each concrete motif appears
    obs = sum(obs, [])
    motif_to_windows = Counter(obs)
    total = sum([np.log(v ** 2) / k ** 2 for key, v in motif_to_windows.items()])
    return total


# Example usage:
def score_template(barcode):
    _2 = expanded_motif_repetition_penalty(barcode, k=2)
    _3 = expanded_motif_repetition_penalty(barcode, k=3)
    _4 = expanded_motif_repetition_penalty(barcode, k=4)
    _5 = expanded_motif_repetition_penalty(barcode, k=5)
    score = np.mean([_2, _3, _4, _5])

    return score


def could_form_homopolymer(seq, max_len):
    for i in range(len(seq) - max_len + 1):
        window = seq[i:i+max_len]
        sets = [degenerate_map.get(base, {base}) for base in window]
        intersection = set.intersection(*sets)
        if intersection:
            return True  # This window could form a homopolymer
    return False

def generate_barcode_template(barcode_len, motif, max_homopolymer_len=3, attempt_n_barcodes=10, **kwargs):
    initial_template = motif * np.ceil(barcode_len / len(motif)).astype(int)
    initial_template = initial_template[:barcode_len]

    observations = []
    scores = []
    for idx in range(attempt_n_barcodes):
        processing = True
        while processing:
            # Add bases to minimize homopolymer introduction
            barcode = []
            for _, c in enumerate(initial_template):
                invalid_nucleotides = set()
                if len(barcode) > 0:
                    if barcode[-1] == 'N':
                        invalid_nucleotides.add('N')

                if len(barcode) >= max_homopolymer_len - 1:
                    # restrict option set for next time to reduce homopolymers
                    recent = barcode[-(max_homopolymer_len - 1):]

                    expanded_sets = [degenerate_map[base] for base in recent]
                    possible_homopolymer_bases = set.intersection(*expanded_sets)
                    if possible_homopolymer_bases:
                        invalid_nucleotides.update(
                            n for n, expansion in degenerate_map.items() if possible_homopolymer_bases & expansion)

                # Filter probabilities to exclude invalid nucleotides
                valid_nucleotides = [n for n in int_to_degenerate[int(c)] if n not in invalid_nucleotides]

                # Sample the next nucleotide
                if len(valid_nucleotides) == 0:
                    continue
                    # raise Exception('Parameters (x and max homopolymer length) too stringent for template construction')
                else:
                    next_base = random.choices(valid_nucleotides, k=1)[0]
                    barcode.append(next_base)

            barcode = ''.join(barcode)
            if barcode_len != len(barcode):
                continue

            processing = False
            observations.append(barcode)
            score = score_template(barcode)
            scores.append(score)

    sns.histplot(scores)
    plt.show()
    barcode = observations[np.argmax(scores)]

    return barcode


def simulate_barcoded_data(variant, barcode_template, duplication_rate, left_coding_flank, right_coding_flank):
    # Define the degenerate nucleotide set
    degenerate_nucleotides = set("RYSWBKMDHVN")

    true_barcodes = generate_random_barcodes(barcode_template, 1) * duplication_rate
    start_idx = next(i for i, nt in enumerate(barcode_template) if nt in degenerate_nucleotides)
    end_idx = next(
        i for i in range(len(barcode_template) - 1, -1, -1) if barcode_template[i] in degenerate_nucleotides) + 1

    true_barcodes_no_constant = [seq[start_idx:end_idx] for seq in true_barcodes]

    pre_post_flank_template = '4' * 60
    post_flank = generate_random_barcodes(generate_sequence(pre_post_flank_template), 1)[0]
    pre_flank = generate_random_barcodes(generate_sequence(pre_post_flank_template), 1)[0]
    buffer = generate_random_barcodes(generate_sequence(pre_post_flank_template), 1)[0]

    # noise sequence with nanopore simulator
    true_reference = barcode_template
    generated_sequences_init = [pre_flank + b + buffer + left_coding_flank + variant + right_coding_flank + post_flank for b in true_barcodes]
    generated_sequences, generated_qualities = sim.simulate_many(generated_sequences_init)

    return true_reference, generated_sequences, true_barcodes, true_barcodes_no_constant, generated_qualities

def main(barcode_template, coding_sequence, left_coding_flank, right_coding_flank,
         duplication_rate, barcodes_per_variant, num_variants, fn, **kwargs):
    fastq_fn = fn + '.fastq'
    fasta_fn = fn + '.fasta'
    mapping_fn = fn + '_mapping.tsv'

    data = []
    qualities = []
    for m in range(num_variants):
        variant = generate_random_mut(coding_sequence, num_muts=3)
        for b in range(barcodes_per_variant):
            out = simulate_barcoded_data(variant, barcode_template, duplication_rate, left_coding_flank, right_coding_flank)
            data += [[out[0], v, out[2][i], out[3][i], variant] for i, v in enumerate(out[1])]
            qualities += out[-1]

    data = pd.DataFrame(data, columns=['reference', 'synthetic_sequence', 'barcode_with_flanks', 'true_barcode', 'variant'])
    write_synthetic_fastq(data.synthetic_sequence.to_list(), fastq_fn, qualities=qualities)

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_pickle(fasta_fn.replace('.fasta', '.pkl'))

    print(f"Generated {len(data)} synthetic sequences using barcode template:\n{data.reference[0]}")

    mapping = data[['true_barcode','variant']].drop_duplicates().reset_index(drop=True)
    mapping.to_csv(mapping_fn.replace('.tsv', '_synthetic.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=60,
                        help='Length of barcode when generating')
    parser.add_argument('--max_homopolymer_len', type=int, default=3,
                        help='Do not allow sequences with possible homopolymers longer than this value')
    parser.add_argument('--attempt_n_barcodes', type=int, default=1000,
                        help='Number of attempts to generate barcode template')
    parser.add_argument('--motif', type=str, default=None,
                        help='Sequence of 1,2,3,4 integers to repeat until barcode_len is met for degenerate sampling')

    # Parameters defining what syndata to generate
    parser.add_argument('--duplication_rate', type=float, default=50,
                        help='Analogous to sequencing depth')
    parser.add_argument('--barcodes_per_variant', type=float, default=10)
    parser.add_argument('--num_variants', type=float, default=10)

    # Barcode and coding parameters
    parser.add_argument('--barcode_template', type=str,
                        default='ATGCAGRMBRWYRWHBMRDBHRVBWBRNMKHVWSWHVBWBSHDVKMBWBVSWVNKMDSWSDNWSVHGCATC',
                        help='Reference degenerate barcode to align sequences to')
    parser.add_argument('--coding_sequence', type=str,
                        default='ATGGAAAACAATCTGGAAAACCTGACCATCGGCGTGTTTGCGAAGGCTGCGGGCGTAAACGTGGAAACGATTCGTTTCTATCA'
                       'GCGTAAAGGGCTGCTGCGCGAACCTGACAAACCATACGGCTCAATTCGGCGTTATGGTGAGGCCGATGTCGTGCGCGTAAAATT'
                       'TGTGAAAAGTGCTCAACGCCTGGGGTTCTCCTTGGATGAGATCGCTGAACTTCTGCGTCTGGATGATGGAACTCACTGCGAAGAA'
                       'GCGAGTTCGCTCGCAGAACATAAACTCAAAGACGTTCGCGAGAAAATGGCCGACCTTGCACGTATGGAAACCGTCTTATCTGAACT'
                       'GGTTTGCGCGTGTCATGCGCGCAAGGGTAATGTTAGCTGTCCGCTGATTGCGAGCTTGCAGGGTGAGGCCGGCTTAGCCCGGAGCGCAATGCCGTAA',
                        help='Sequence that barcodes will be mapped to')
    parser.add_argument('--left_coding_flank', type=str, default='CTGCTATCGT',
                        help='Sequence just left of coding sequence to be used for extraction of mapping after clustering')
    parser.add_argument('--right_coding_flank', type=str, default='ATCTAGCATC',
                        help='Sequence just right of coding sequence to be used for extraction of mapping after clustering')
    parser.add_argument('--fn', type=str, default='syndata/syndataF')

    args = parser.parse_args()

    makedirs('/'.join(args.fn.split('/')[:-1]), exist_ok=True)
    if args.barcode_template is None:
        # Generating new barcode based on motif
        assert args.motif is not None
        args.barcode_template = generate_barcode_template(**vars(args))

    main(**vars(args))