from glob import glob
import regex
from Bio import SeqIO
from os.path import isdir
import argparse
import time

from dnabarmap.utils import nuc_dict


def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R',  # R=A/G, Y=C/T
        'S': 'S', 'W': 'W',  # S=G/C, W=A/T (self-complementary)
        'K': 'M', 'M': 'K',  # K=G/T, M=A/C
        'B': 'V', 'V': 'B',  # B=C/G/T (not A), V=A/C/G (not T)
        'D': 'H', 'H': 'D',  # D=A/G/T (not C), H=A/C/T (not G)
        'N': 'N'
    }
    return ''.join(complement[b] for b in reversed(seq.upper()))


def make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                              left_fuzz, right_fuzz, bar_fuzz):
    barcode_regex = build_degenerate_regex(barcode_template)
    barcode_regex_rc = build_degenerate_regex(reverse_complement(barcode_template))

    left_flank_rc = reverse_complement(left_coding_flank)
    right_flank_rc = reverse_complement(right_coding_flank)

    # Forward orientation: left_flank(coding)right_flank...barcode
    regex_A = (
        fr"(?:{left_coding_flank}){{e<={left_fuzz}}}(?P<coding>[ATCGN]*)(?:{right_coding_flank}){{e<={right_fuzz}}}"
        fr"[ATCGN]*(?P<barcode>{barcode_regex}){{e<={bar_fuzz}}}"
    )

    # Reverse orientation: barcode_rc...right_flank_rc(coding)left_flank_rc
    regex_B = (
        fr"(?P<barcode>{barcode_regex_rc}){{e<={bar_fuzz}}}[ATCGN]*"
        fr"(?:{right_flank_rc}){{e<={right_fuzz}}}(?P<coding>[ATCGN]*)(?:{left_flank_rc}){{e<={left_fuzz}}}"
    )

    # Return: (pattern, barcode_group, coding_group, name)
    return [(regex_A, 'barcode', 'coding', 'A'), (regex_B, 'barcode', 'coding', 'B')]


def match_with_orientation(seq, matchers, orientation_counts):
    sorted_matchers = sorted(matchers, key=lambda m: orientation_counts.get(m[3], 0), reverse=True)
    other_key = {'A': 'B', 'B': 'A'}

    for compiled_regex, barcode_group, coding_group, name in sorted_matchers:
        match = regex.search(compiled_regex, seq, regex.BESTMATCH)
        if match:
            orientation_counts[name] = orientation_counts.get(name, 0) + 1
            return match.group(barcode_group), match.group(coding_group), orientation_counts, name
        elif sum(orientation_counts.values()) > 100:
            if orientation_counts.get(name, 0) > (orientation_counts.get(other_key[name], 0) + 1) * 100:
                return None, None, orientation_counts, None

    return None, None, orientation_counts, None

def build_degenerate_regex(template):
    pattern = ''
    for base in template:
        allowed = nuc_dict[base]
        if len(allowed) == 1:
            pattern += allowed[0]
        else:
            pattern += f"[{''.join(allowed)}]"
    return pattern


def consensus_mapping(consensus_dir, barcode_template, left_coding_flank, right_coding_flank, mapping_fn, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.2))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.2))
    bar_fuzz = max(1, int(len(barcode_template) * 0.01))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 consensus_dir, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn)


def direct_mapping(fn, barcode_template, left_coding_flank, right_coding_flank, mapping_fn, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.2))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.2))
    bar_fuzz = max(1, int(len(barcode_template) * 0.025))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 fn, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn)


def map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 input_files, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn, **kwargs):
    if isdir(input_files):
        consensus_files = glob(f"{input_files}/*/cluster_*_consensus.fasta")  # if running map
        print(f"{input_files}/*/cluster_*_consensus.fasta")
        print(f"Determining mapping for {len(consensus_files)} consensus sequences")

        if len(consensus_files) == 0:
            consensus_files = glob(f"{input_files}/consensus/*/cluster_*_consensus.fasta")  # if running dnabarmap
            if len(consensus_files) == 0:
                raise Exception(
                    f"No consensus sequences found in {input_files}. Consider altering hyperparameters or doing deeper sequencing.")
    else:
        consensus_files = [input_files]

    matchers = make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                                         left_fuzz, right_fuzz, bar_fuzz)
    orientation_counts = {}
    no_match_count = 0
    observations = 0
    mapping_fn = '.'.join(mapping_fn.split('.')[:-1]) + f'.tsv'
    with open(mapping_fn, "w") as out:
        out.write("filename\tbarcode\tcoding_region\n")
        for file in sorted(consensus_files):
            for record in SeqIO.parse(file, file.split('.')[-1]):
                seq = str(record.seq).upper()
                barcode, coding_region, orientation_counts, orientation = match_with_orientation(seq, matchers,
                                                                                                 orientation_counts)
                if barcode:
                    if orientation == 'B':
                        barcode = reverse_complement(barcode)
                        coding_region = reverse_complement(coding_region)
                    out.write(f"{file}\t{barcode}\t{coding_region}\n")
                else:
                    no_match_count += 1
                observations += 1

    print(f"Found a match for {observations - no_match_count}/{observations} sequences")


def main():
    direct = False

    if direct:
        parser = argparse.ArgumentParser()
        # Set alignment pa
        # rameters
        parser.add_argument('--barcode_template', type=str,
                            default=None,
                            help='Reference degenerate barcode to align sequences to')
        parser.add_argument('--fn', type=str, default=None)
        parser.add_argument('--left_coding_flank', type=str, default=None)
        parser.add_argument('--right_coding_flank', type=str, default=None)
        parser.add_argument('--mapping_fn', type=str, default=None)

        all_args = parser.parse_known_args()
        args = all_args[0]

        direct_mapping(**vars(args))

    else:
        parser = argparse.ArgumentParser()

        # Directories and filenaemes
        parser.add_argument('--consensus_dir', type=str, default=None, required=not direct,
                            help='Combined input fasta file')
        parser.add_argument("--mapping_fn", default=None, required=True,
                            help="Final mapping output filename")

        # Define barcode and sequence parameters
        parser.add_argument('--barcode_template', type=str, required=True,
                            default=None,
                            help='Degenerate reference for conducting approximate alignment of sequences')
        parser.add_argument("--left_coding_flank", default=None, required=True,
                            help="Left constant sequence of coding region")
        parser.add_argument("--right_coding_flank", default=None, required=True,
                            help="Right constant sequence of coding region")

        all_args = parser.parse_known_args()
        args = all_args[0]

        # Set up directories and filenames

        args.barcode_directory = args.consensus_dir.split('/consensus')[-1].split('/')[-1]
        args.output_dir = f'temp/{args.barcode_directory}/'
        args.cluster_dir = args.output_dir + '/clusters/'

        # Use regular expressions to map barcodes to coding sequences for consensus sequences
        print('Mapping barcodes to coding sequences...')
        mapping_start_time = time.time()
        consensus_mapping(**vars(args))
        mapping_time = time.time() - mapping_start_time
        print(f'Finished mapping barcodes in {round(mapping_time / 60, 1)} minutes\n')


def cli():
    main()


if __name__ == "__main__":
    cli()
