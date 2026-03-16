import numpy as np
import pandas as pd
import random
import argparse
from itertools import product
from collections import Counter
from os import makedirs

from dnabarmap.generate import generate_sequence, generate_random_barcodes
from dnabarmap.utils import degenerate_map, int_to_degenerate, generate_random_mut, write_synthetic_fastq
import dnabarmap.simulate as sim

DEFAULT_SEED = 100
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)


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

    barcode = observations[np.argmax(scores)]

    return barcode


def simulate_barcoded_data(variant, barcode_template, duplication_rate, left_coding_flank,
                           right_coding_flank, quality_preset='cleaner',
                           qshmm_model='QSHMM-ONT-HQ.model', accuracy_mean=None,
                           difference_ratio=None, hp_del_bias=None,
                           pass_num=None, length_mean=None, length_sd=None,
                           seed=None):
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
    generated_sequences, generated_qualities = sim.simulate_many(
        generated_sequences_init,
        quality_preset=quality_preset,
        qshmm_model=qshmm_model,
        accuracy_mean=accuracy_mean,
        difference_ratio=difference_ratio,
        hp_del_bias=hp_del_bias,
        pass_num=pass_num,
        length_mean=length_mean,
        length_sd=length_sd,
        seed=seed,
    )

    return true_reference, generated_sequences, true_barcodes, true_barcodes_no_constant, generated_qualities

def main(barcode_template, coding_sequence, left_coding_flank, right_coding_flank,
         duplication_rate, barcodes_per_variant, num_variants, fn, **kwargs):
    fastq_fn = fn + '.fastq'
    fasta_fn = fn + '.fasta'
    mapping_fn = fn + '_mapping.tsv'

    data = []
    pbsim_seed_base = kwargs.get('pbsim_seed')
    if pbsim_seed_base is None:
        pbsim_seed_base = kwargs.get('seed')
    for m in range(num_variants):
        variant = generate_random_mut(coding_sequence, num_muts=3)
        for b in range(barcodes_per_variant):
            call_seed = None
            if pbsim_seed_base is not None:
                call_seed = int(pbsim_seed_base) + (m * barcodes_per_variant) + b
            out = simulate_barcoded_data(
                variant,
                barcode_template,
                duplication_rate,
                left_coding_flank,
                right_coding_flank,
                quality_preset=kwargs.get('quality_preset', 'cleaner'),
                qshmm_model=kwargs.get('qshmm_model', 'QSHMM-ONT-HQ.model'),
                accuracy_mean=kwargs.get('accuracy_mean'),
                difference_ratio=kwargs.get('difference_ratio'),
                hp_del_bias=kwargs.get('hp_del_bias'),
                pass_num=kwargs.get('pass_num'),
                length_mean=kwargs.get('length_mean'),
                length_sd=kwargs.get('length_sd'),
                seed=call_seed,
            )
            data += [
                [out[0], seq, out[2][i], out[3][i], variant, out[-1][i]]
                for i, seq in enumerate(out[1])
            ]

    data = pd.DataFrame(
        data,
        columns=['reference', 'synthetic_sequence', 'barcode_with_flanks',
                 'true_barcode', 'variant', 'quality'],
    )
    data = data.sample(frac=1, random_state=kwargs.get('seed')).reset_index(drop=True)

    write_synthetic_fastq(
        data.synthetic_sequence.to_list(),
        fastq_fn,
        qualities=data.quality.to_list(),
    )

    data.to_pickle(fasta_fn.replace('.fasta', '.pkl'))

    print(f"Generated {len(data)} synthetic sequences using barcode template:\n{data.reference[0]}")
    print(f"Saved sequences to {fasta_fn.replace('.fasta', '.pkl')}")

    mapping = data[['true_barcode','variant']].drop_duplicates().reset_index(drop=True)
    mapping.to_csv(mapping_fn.replace('.tsv', '_synthetic.tsv'), sep='\t', index=False)

def cli():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DNABARMAP reads plus a truth table."
    )

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=60,
                        help='Barcode length to generate when '
                             '--barcode_template is not supplied.')
    parser.add_argument('--max_homopolymer_len', type=int, default=4,
                        help='Reject generated templates with possible '
                             'homopolymers longer than this.')
    parser.add_argument('--attempt_n_barcodes', type=int, default=200,
                        help='Number of template candidates to try when '
                             'generating a barcode template automatically.')
    parser.add_argument('--motif', type=str, default=None,
                        help='Optional repeating motif of degeneracy classes '
                             '(1,2,3,4) used when generating a template.')

    # Parameters defining what syndata to generate
    parser.add_argument('--duplication_rate', type=int, default=50,
                        help='Approximate reads per construct before PBSIM '
                             'simulation.')
    parser.add_argument('--barcodes_per_variant', type=int, default=10,
                        help='Distinct barcodes to generate for each variant.')
    parser.add_argument('--num_variants', type=int, default=10,
                        help='Number of distinct coding variants to simulate.')

    # Barcode and coding parameters
    parser.add_argument('--barcode_template', type=str,
                        default='VHKNSHDKSYRRSHDVYHDVBMKHDMBKVWBKMBNDMKKMKVVHKMKVHBKMNKHVDBMKVYBKMBSWBVHKMKWN',
                        help='Degenerate barcode template to sample from. If '
                             'omitted, one is generated automatically.')
    parser.add_argument('--coding_sequence', type=str,
                        default='ATGGAAAACAATCTGGAAAACCTGACCATCGGCGTGTTTGCGAAGGCTGCGGGCGTAAACGTGGAAACGATTCGTTTCTATCA'
                       'GCGTAAAGGGCTGCTGCGCGAACCTGACAAACCATACGGCTCAATTCGGCGTTATGGTGAGGCCGATGTCGTGCGCGTAAAATT'
                       'TGTGAAAAGTGCTCAACGCCTGGGGTTCTCCTTGGATGAGATCGCTGAACTTCTGCGTCTGGATGATGGAACTCACTGCGAAGAA'
                       'GCGAGTTCGCTCGCAGAACATAAACTCAAAGACGTTCGCGAGAAAATGGCCGACCTTGCACGTATGGAAACCGTCTTATCTGAACT'
                       'GGTTTGCGCGTGTCATGCGCGCAAGGGTAATGTTAGCTGTCCGCTGATTGCGAGCTTGCAGGGTGAGGCCGGCTTAGCCCGGAGCGCAATGCCGTAA',
                        help='Base coding sequence used to generate variants.')
    parser.add_argument('--left_coding_flank', type=str, default='CCCACTG',
                        help='Constant sequence immediately left of the coding '
                             'region in the simulated construct.')
    parser.add_argument('--right_coding_flank', type=str, default='ATGCGTA',
                        help='Constant sequence immediately right of the coding '
                             'region in the simulated construct.')
    parser.add_argument('--fn', type=str, default='syndata/syndata',
                        help='Output stem for the generated .pkl and .fastq.')
    parser.add_argument('--quality_preset', choices=('cleaner', 'default', 'harsh'),
                        default='cleaner',
                        help="PBSIM quality preset. 'cleaner' is the recommended "
                             "default and produces easier reads, 'harsh' produces "
                             "noisier reads, and 'default' preserves the baseline "
                             "simulator behavior.")
    parser.add_argument('--qshmm_model', type=str, default='QSHMM-ONT-HQ.model',
                        help='PBSIM QSHMM model path, or a model filename '
                             'inside pbsim3_models/.')
    parser.add_argument('--accuracy_mean', type=float, default=None,
                        help='Optional PBSIM override for mean read accuracy.')
    parser.add_argument('--difference_ratio', type=str, default=None,
                        help='Optional PBSIM override for substitution:insertion:deletion ratio, e.g. 39:24:36.')
    parser.add_argument('--hp_del_bias', type=float, default=None,
                        help='Optional PBSIM override for homopolymer deletion bias.')
    parser.add_argument('--pass_num', type=int, default=None,
                        help='Optional PBSIM override for sequencing passes.')
    parser.add_argument('--length_mean', type=float, default=None,
                        help='Optional PBSIM override for mean read length.')
    parser.add_argument('--length_sd', type=float, default=None,
                        help='Optional PBSIM override for read length standard deviation.')
    parser.add_argument('--pbsim_seed', type=int, default=None,
                        help='Optional PBSIM random seed for reproducible simulations.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Master random seed for NumPy, Python random, and '
                             'default PBSIM seeding (default: 100).')

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.pbsim_seed is None:
        args.pbsim_seed = args.seed

    dirs = '/'.join(args.fn.split('/')[:-1]) if '/'.join(args.fn.split('/')[:-1]) != '' else './'
    makedirs(dirs, exist_ok=True)
    if args.barcode_template is None:
        # Generating new barcode based on motif
        assert args.motif is not None
        args.barcode_template = generate_barcode_template(**vars(args))

    main(**vars(args))

if __name__ == '__main__':
    cli()
