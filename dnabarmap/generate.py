import random
import pandas as pd
from dnabarmap.utils import convert_AA_to_nucleotide, degenerate_map

nuc_dict = {'A': ['A'],
            'T': ['T'],
            'C': ['C'],
            'G': ['G'],
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            'M': ['A', 'C'],
            'K': ['G', 'T'],
            'S': ['C', 'G'],
            'W': ['A', 'T'],
            'H': ['A', 'C', 'T'],
            'B': ['C', 'G', 'T'],
            'V': ['A', 'C', 'G'],
            'D': ['A', 'G', 'T'],
            'N': ['A', 'T', 'C', 'G'],
            }

def generate_random_barcodes(template_barcode, num_barcodes):
    barcodes = []
    for idx in range(num_barcodes):
        barcodes.append(''.join([random.choice(nuc_dict[n]) for n in template_barcode]))

    return barcodes

def generate_sequence(template):
    barcode_sequence = list(str(template))
    for i, v in enumerate(barcode_sequence):
        if '1' == v:
            barcode_sequence[i] = random.choice(['A', 'T', 'C', 'G'])
        elif '2' == v:
            barcode_sequence[i] = random.choice(['R', 'Y', 'M', 'K', 'S', 'W'])
        elif '3' == v:
            barcode_sequence[i] = random.choice(['H', 'B', 'D', 'V'])
        elif '4' == v:
            barcode_sequence[i] = 'N'

    return ''.join(barcode_sequence)

def adjust_probabilites(probabilites):
    adj_probabilites = {}
    total = 0
    # provide probabiliites for four classes of distance for nucleotides or semi degenerate nucs
    for i, v in enumerate(probabilites):
        if i == 0:
            options = ['A', 'T', 'C', 'G']
        elif i == 1:
            options = ['R', 'Y', 'M', 'K', 'S', 'W']
        elif i == 2:
            options = ['H', 'B', 'D', 'V']
        elif i == 3:
            options = ['N']
        else:
            raise Exception('Invalid input probabilities')

        for o in options:
            adj_probabilites[o] = probabilites[i] / len(options)
            total += probabilites[i] / len(options)

    return adj_probabilites



def generate_protein_variants(WT, num_seqs, num_muts):
    mutant_seqs = []
    for i in range(num_seqs):
        indices = []
        for m in range(num_muts):
            indices.append(random.randint(0, len(WT)-1))

        mutant_seq = list(WT)
        for index in indices:
            mutant_options = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            mutant_options.remove(mutant_seq[index])
            mutant_seq[index] = random.choice(mutant_options)
        mutant_seqs.append(''.join(mutant_seq))

    variants_df = pd.DataFrame(mutant_seqs, columns=['sequence'])
    variants_df.loc[:, 'dna_sequence'] = convert_AA_to_nucleotide(variants_df.sequence)

    return variants_df


def make_barcode_template(frac_1, frac_2, frac_3, frac_4, max_homopolymer_len=3, barcode_len=None):
    # Initialize barcode
    total = frac_1 + frac_2 + frac_3 + frac_4
    probs = [frac_1 / total, frac_2 / total, frac_3 / total, frac_4 / total]
    adj_probs = adjust_probabilites(probs)
    barcode = []
    nucleotides = list(adj_probs.keys())

    if barcode_len is None:
        barcode_len = random.randint(40, 80)

    # Add bases to minimize homopolymer introduction
    for i in range(barcode_len):
        invalid_nucleotides = set()
        if len(barcode) > 0:
            if barcode[-1] == 'N':
                invalid_nucleotides.add('N')

        if len(barcode) >= max_homopolymer_len - 1:
            # Get the last `max_homopolymer_len - 1` nucleotides
            recent = barcode[-(max_homopolymer_len - 1):]

            # Expand the recent sequence using the degenerate map
            expanded_sets = [degenerate_map[base] for base in recent]
            possible_homopolymer_bases = set.intersection(*expanded_sets)
            if possible_homopolymer_bases:
                invalid_nucleotides.update(
                    n for n, expansion in degenerate_map.items() if possible_homopolymer_bases & expansion)

        # Filter probabilities to exclude invalid nucleotides
        valid_nucleotides = [n for n in nucleotides if n not in invalid_nucleotides]
        valid_probs = [adj_probs[n] for n in valid_nucleotides]
        valid_probs = [p / sum(valid_probs) for p in valid_probs]  # Renormalize

        # Sample the next nucleotide
        next_base = random.choices(valid_nucleotides, weights=valid_probs, k=1)[0]
        barcode.append(next_base)

    barcode_template = ''.join(barcode)

    return barcode_template
