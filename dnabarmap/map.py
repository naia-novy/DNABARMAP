from glob import glob
import regex
from Bio import SeqIO
from os.path import isdir

from dnabarmap.utils import nuc_dict
def make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                          left_fuzz, right_fuzz, bar_fuzz):
    barcode_regex = build_degenerate_regex(barcode_template)

    # regex_A = (
    #     fr"({barcode_regex}){{e<={bar_fuzz},s<={bar_fuzz * sub_mult}}}[ATCGN]*"
    #     fr"{left_coding_flank}{{e<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{e<={right_fuzz}}}"
    # )
    #
    # regex_B = (
    #     fr"{left_coding_flank}{{e<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{e<={right_fuzz}}}"
    #     fr"[ATCGN]*({barcode_regex}){{e<={bar_fuzz},s<={bar_fuzz * sub_mult}}}"
    # )

    regex_A = (
        fr"({barcode_regex}){{e<={bar_fuzz}}}[ATCGN]*"
        fr"{left_coding_flank}{{e<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{e<={right_fuzz}}}"
    )

    regex_B = (
        fr"{left_coding_flank}{{e<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{e<={right_fuzz}}}"
        fr"[ATCGN]*({barcode_regex}){{e<={bar_fuzz}}}"
    )

    return [(regex_A, 1, 2, 'A'),  (regex_B, 2, 1, 'B')]

def match_with_orientation(seq, matchers, orientation_counts):
    """
    Try to match sequence against both orientations, starting with the more successful one.
    Updates orientation_counts in place and returns (barcode, coding_region) or (None, None).
    """
    # Sort matchers by success count (descending) to try the dominant one first
    sorted_matchers = sorted(matchers, key=lambda m: orientation_counts.get(m[3], 0), reverse=True)
    other_key = {'A': 'B', 'B': 'A'}

    for compiled_regex, barcode_pos, coding_pos, name in sorted_matchers:
        match = regex.search(compiled_regex, seq, regex.BESTMATCH)
        if match:
            orientation_counts[name] = orientation_counts.get(name, 0) + 1
            return match.group(barcode_pos), match.group(coding_pos), orientation_counts
        elif sum(orientation_counts.values()) > 100: # This is needed to prevent segsevg for some reason
            if orientation_counts.get(name, 0) > (orientation_counts.get(other_key[name], 0) + 1) * 100:
                # confident in orientation, instead of retesting just continue
                return None, None, orientation_counts

    return None, None, orientation_counts


def build_degenerate_regex(template):
    pattern = ''
    for base in template:
        allowed = nuc_dict[base]
        if len(allowed) == 1:
            pattern += allowed[0]
        else:
            pattern += f"[{''.join(allowed)}]"
    return pattern

def consensus_mapping(consensus_dir, barcode_template, left_coding_flank, right_coding_flank, output_mapping_fn, barcode_directory, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.2))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.2))
    bar_fuzz = max(1, int(len(barcode_template) * 0.01))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
        consensus_dir, barcode_template, left_coding_flank,
        right_coding_flank, output_mapping_fn, barcode_directory)

def direct_mapping(fn, barcode_template, left_coding_flank, right_coding_flank, output_mapping_fn, barcode_directory, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.2))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.2))
    bar_fuzz = max(1, int(len(barcode_template) * 0.025))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
        fn, barcode_template, left_coding_flank,
        right_coding_flank, output_mapping_fn, barcode_directory)


def map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
        input_data, barcode_template, left_coding_flank,
        right_coding_flank, output_mapping_fn, barcode_directory, **kwargs):

    if isdir(input_data):
        consensus_files = glob(f"{input_data}consensus_*/cluster_*_consensus.fasta")
        print(f"Determining mapping for {len(consensus_files)} consensus sequences")

        if len(consensus_files) == 0:
            raise Exception(f"No consensus sequences found in {input_data}/{barcode_directory}/consensus. Consider altering hyperparameters or doing deeper sequencing.")
    else:
        consensus_files = [input_data]

    matchers = make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                                         left_fuzz, right_fuzz, bar_fuzz)
    orientation_counts = {}
    no_match_count = 0
    observations = 0
    output_mapping_fn = '.'.join(output_mapping_fn.split('.')[:-1]) + f'.tsv'
    with open(output_mapping_fn, "w") as out:
        out.write("filename\tbarcode\tcoding_region\n")
        for file in sorted(consensus_files):
            for record in SeqIO.parse(file, file.split('.')[-1]):
                seq = str(record.seq).upper()
                barcode, coding_region, orientation_counts = match_with_orientation(seq, matchers, orientation_counts)

                if barcode:
                    out.write(f"{file}\t{barcode}\t{coding_region}\n")
                else:
                    no_match_count += 1
                observations += 1

    print(f"Found a match for {observations-no_match_count}/{observations} sequences")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Set alignment parameters
    parser.add_argument('--barcode_template', type=str,
                        default=None,
                        help='Reference degenerate barcode to align sequences to')
    parser.add_argument('--fn', type=str, default=None)
    parser.add_argument('--left_coding_flank', type=str, default=None)
    parser.add_argument('--right_coding_flank', type=str, default=None)
    parser.add_argument('--output_mapping_fn', type=str, default=None)
    parser.add_argument('--barcode_directory', type=str, default=None)

    args = parser.parse_args()

    direct_mapping(**vars(args))
