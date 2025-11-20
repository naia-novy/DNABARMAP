from glob import glob
import regex
from Bio import SeqIO

from .utils import nuc_dict

def determine_mapping(consensus_dir, barcode_template, left_coding_flank, right_coding_flank, output_mapping_fn, **kwargs):
    # Use regular expressions to extract the barcode and variant mappings from the consensus sequence
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    if not consensus_dir.endswith('/'):
        consensus_dir += '/'

    consensus_files = glob(f"temp/consensus/consensus_*/cluster_*_consensus.fasta")
    print(f"Determining mapping for {len(consensus_files)} consensus sequences")

    if len(consensus_files) == 0:
        raise Exception("No consensus sequences found. Consider altering hyperparameters or doing deeper sequencing.")

    left_fuzz, right_fuzz = max(1, int(len(left_coding_flank)*0.1)), max(1, int(len(right_coding_flank)*0.1))
    bar_fuzz = max(1, int(len(barcode_template)*0.1))
    barcode_regex = build_degenerate_regex(barcode_template)

    # try two possible orientations
    # combined_regex = regex.compile(fr"({barcode_regex}){{s<={bar_fuzz}}}[ATCGN]*{left_coding_flank}{{s<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{s<={right_fuzz}}}", flags=regex.BESTMATCH)
    # barcode_pos = 1
    # coding_pos = 2
    combined_regex = regex.compile(fr"{left_coding_flank}{{s<={left_fuzz}}}([ATCGN]*){right_coding_flank}{{s<={right_fuzz}}}[ATCGN]*({barcode_regex}){{s<={bar_fuzz}}}", flags=regex.BESTMATCH)
    barcode_pos = 2
    coding_pos = 1

    no_match_count = 0
    with open(output_mapping_fn, "w") as out:
        out.write("filename\tbarcode\tcoding_region\n")
        for file in sorted(consensus_files):
            for record in SeqIO.parse(file, "fasta"):
                seq = str(record.seq).upper()
                match = combined_regex.search(seq)
                if match:
                    barcode = match.group(barcode_pos)
                    coding_region = match.group(coding_pos)
                    out.write(f"{file}\t{barcode}\t{coding_region}\n")
                else:
                    no_match_count += 1
                break # only do first record

    print(f"Did not find a match for {no_match_count}/{len(consensus_files)} sequences")

def build_degenerate_regex(template):
    pattern = ''
    for base in template:
        allowed = nuc_dict[base]
        if len(allowed) == 1:
            pattern += allowed[0]
        else:
            pattern += f"[{''.join(allowed)}]"
    return pattern

