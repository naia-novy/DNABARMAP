from glob import glob
import regex
from Bio import SeqIO

from .utils import nuc_dict

def determine_mapping(consensus_dir, barcode_template, left_coding_flank, right_coding_flank, output_mapping_fn, **kwargs):
    # Use regular expressions to extract the barcode and variant mappings from the consensus sequence
    if not consensus_dir.endswith('/'):
        consensus_dir += '/'

    consensus_files = glob(f"tmp/consensus/cluster_*_consensus.fasta")
    print(f"Determining mapping for {len(consensus_files)} consensus sequences")

    barcode_len = len(barcode_template)
    if len(consensus_files) == 0:
        raise Exception("No consensus sequences found. Consider altering hyperparameters or doing deeper sequencing.")

    barcode_regex = build_degenerate_regex(barcode_template)

    # combined_regex = regex.compile(
    #     fr"({barcode_regex}){{s<=5}}[ATCGN]*{left_coding_flank}{{e<=2}}([ATCGN]*){right_coding_flank}{{e<=2}}"
    # )
    combined_regex = regex.compile(
        fr"{left_coding_flank}{{e<=2}}([ATCGN]*){right_coding_flank}{{e<=2}}"
    )

    no_match_count = 0
    with open(output_mapping_fn, "w") as out:
        out.write("filename\tbarcode\tcoding_region\n")
        for file in sorted(consensus_files):
            for record in SeqIO.parse(file, "fasta"):
                seq = str(record.seq).upper()
                match = combined_regex.search(seq)
                if match:
                    barcode = match.group(1)
                    coding_region = match.group(2)
                    out.write(f"{file}\t{barcode}\t{coding_region}\n")
                else:
                    no_match_count += 1

                break # only do first record

    print(f"Did not find matches for {round(no_match_count/len(consensus_files)*100, 1)}% of sequences")

def build_degenerate_regex(template):
    pattern = ''
    for base in template:
        allowed = nuc_dict[base]
        if len(allowed) == 1:
            pattern += allowed[0]
        else:
            pattern += f"[{''.join(allowed)}]"
    return pattern

