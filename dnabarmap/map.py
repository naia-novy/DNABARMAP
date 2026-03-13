from glob import glob
import regex
import re
import csv
from Bio import SeqIO
from os.path import isdir
from pathlib import Path
from collections import defaultdict
import argparse
import time

from dnabarmap.utils import nuc_dict


# ═════════════════════════════════════════════════════════════════
# HOMOPOLYMER-AWARE REFERENCE MATCHING
# ═════════════════════════════════════════════════════════════════

def hp_compress_simple(seq):
    """Collapse homopolymer runs to single bases: 'AAACGT' → 'ACGT'"""
    return re.sub(r'(.)\1+', r'\1', seq)


def _edit_distance_banded(s1, s2, max_dist=None):
    """Levenshtein distance with early exit if distance exceeds max_dist."""
    n, m = len(s1), len(s2)
    if max_dist is not None and abs(n - m) > max_dist:
        return max_dist + 1
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
        if max_dist is not None and min(prev) > max_dist:
            return max_dist + 1
    return prev[n]


def build_reference_index(references):
    """
    Build a homopolymer-compressed index from a dict of {name: sequence}.
    Maps compressed_seq → [(name, original_seq), ...]
    """
    hp_index = defaultdict(list)
    for name, seq in references.items():
        compressed = hp_compress_simple(seq.upper())
        hp_index[compressed].append((name, seq))

    n_refs = len(references)
    n_compressed = len(hp_index)
    print(f"  Reference index: {n_refs} sequences → {n_compressed} unique HP-compressed entries")
    if n_compressed < n_refs:
        n_ambiguous = sum(1 for v in hp_index.values() if len(v) > 1)
        print(f"  ⚠ {n_ambiguous} compressed entries map to multiple references "
              f"(differ only in homopolymer lengths)")
    return hp_index


def snap_to_reference(coding_seq, hp_index, max_edits_compressed=3, max_edits_full=5):
    """
    Snap an extracted coding region to the nearest reference sequence.

    Uses HP-compressed edit distance as a fast pre-filter, then full
    edit distance for final selection.

    Returns:
        (ref_name, ref_seq) if a match is found within thresholds,
        (None, None) otherwise — caller should use the original sequence.
    """
    if not coding_seq:
        return None, None

    query_upper = coding_seq.upper()
    query_compressed = hp_compress_simple(query_upper)

    # Fast path: exact compressed match
    if query_compressed in hp_index:
        candidates = hp_index[query_compressed]
        if len(candidates) == 1:
            return candidates[0]
        best, best_dist = None, float('inf')
        for name, seq in candidates:
            d = _edit_distance_banded(query_upper, seq.upper())
            if d < best_dist:
                best_dist, best = d, (name, seq)
        return best

    # Slow path: nearest compressed match within threshold
    best_name = best_seq = None
    best_compressed_dist = best_full_dist = float('inf')

    for ref_compressed, candidates in hp_index.items():
        if abs(len(ref_compressed) - len(query_compressed)) > max_edits_compressed:
            continue
        d_comp = _edit_distance_banded(query_compressed, ref_compressed, max_dist=max_edits_compressed)
        if d_comp > max_edits_compressed:
            continue
        for name, seq in candidates:
            d_full = _edit_distance_banded(query_upper, seq.upper(), max_dist=max_edits_full)
            if d_full > max_edits_full:
                continue
            if (d_comp < best_compressed_dist or
                    (d_comp == best_compressed_dist and d_full < best_full_dist)):
                best_compressed_dist = d_comp
                best_full_dist = d_full
                best_name, best_seq = name, seq

    if best_name is not None:
        return best_name, best_seq

    return None, None


def load_references_from_file(ref_path, seq_col=None, name_col=None):
    """
    Load reference sequences from FASTA, CSV, or TSV.
    Returns dict of {name: sequence}
    """
    ext = Path(ref_path).suffix.lower()

    if ext in ('.fa', '.fasta', '.fna'):
        return {r.id: str(r.seq).upper() for r in SeqIO.parse(ref_path, "fasta")}

    if ext == '.csv':
        delimiter = ','
    elif ext in ('.tsv', '.tab'):
        delimiter = '\t'
    else:
        with open(ref_path) as f:
            sample = f.read(4096)
        try:
            delimiter = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            delimiter = '\t'

    import pandas as pd
    df = pd.read_csv(ref_path, sep=delimiter)

    if seq_col and seq_col in df.columns:
        s_col = seq_col
    else:
        dna_pattern = re.compile(r'^[ACGTacgtNn]+$')
        candidates = []
        for col in df.columns:
            sample_vals = df[col].dropna().head(20).astype(str)
            if sample_vals.apply(lambda x: bool(dna_pattern.match(x))).mean() > 0.8:
                candidates.append((col, sample_vals.str.len().mean()))
        if not candidates:
            raise ValueError(
                f"Could not auto-detect DNA sequence column in {ref_path}. "
                f"Columns: {list(df.columns)}. Use --ref_seq_col to specify."
            )
        s_col = max(candidates, key=lambda x: x[1])[0]
        print(f"  Auto-detected sequence column: '{s_col}'")

    n_col = name_col if (name_col and name_col in df.columns) else next(
        (c for c in df.columns if c != s_col), None
    )

    refs = {}
    for i, row in df.iterrows():
        seq = str(row[s_col]).upper().strip()
        if not seq or seq == 'NAN':
            continue
        name = str(row[n_col]) if n_col else f"ref_{i}"
        if name in refs:
            name = f"{name}_{i}"
        refs[name] = seq

    return refs


# ═════════════════════════════════════════════════════════════════
# BARCODE / CODING REGION EXTRACTION
# ═════════════════════════════════════════════════════════════════

def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R',
        'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K',
        'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D',
        'N': 'N'
    }
    return ''.join(complement[b] for b in reversed(seq.upper()))

def make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                              left_fuzz, right_fuzz, bar_fuzz):
    barcode_regex = build_degenerate_regex(barcode_template)
    barcode_regex_rc = build_degenerate_regex(reverse_complement(barcode_template))

    left_flank_rc = reverse_complement(left_coding_flank)
    right_flank_rc = reverse_complement(right_coding_flank)

    # --- Barcode RIGHT of coding region ---
    # Forward: left_flank(coding)right_flank...barcode
    regex_A = (
        fr"(?:{left_coding_flank}){{e<={left_fuzz}}}(?P<coding>[ATCGN]*)(?:{right_coding_flank}){{e<={right_fuzz}}}"
        fr"[ATCGN]*(?P<barcode>{barcode_regex}){{s<={bar_fuzz}}}"
    )
    # Reverse complement of A: barcode_rc...right_flank_rc(coding)left_flank_rc
    regex_B = (
        fr"(?P<barcode>{barcode_regex_rc}){{s<={bar_fuzz}}}[ATCGN]*"
        fr"(?:{right_flank_rc}){{e<={right_fuzz}}}(?P<coding>[ATCGN]*)(?:{left_flank_rc}){{e<={left_fuzz}}}"
    )

    # --- Barcode LEFT of coding region ---
    # Forward: barcode...left_flank(coding)right_flank
    regex_C = (
        fr"(?P<barcode>{barcode_regex}){{s<={bar_fuzz}}}"
        fr"[ATCGN]*(?:{left_coding_flank}){{e<={left_fuzz}}}(?P<coding>[ATCGN]*)(?:{right_coding_flank}){{e<={right_fuzz}}}"
    )
    # Reverse complement of C: left_flank_rc(coding)right_flank_rc...barcode_rc
    regex_D = (
        fr"(?:{left_flank_rc}){{e<={left_fuzz}}}(?P<coding>[ATCGN]*)(?:{right_flank_rc}){{e<={right_fuzz}}}"
        fr"[ATCGN]*(?P<barcode>{barcode_regex_rc}){{s<={bar_fuzz}}}"
    )

    return [
        (regex_A, 'barcode', 'coding', 'A'),
        (regex_B, 'barcode', 'coding', 'B'),
        (regex_C, 'barcode', 'coding', 'A'),
        (regex_D, 'barcode', 'coding', 'B'),
    ]
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


# ═════════════════════════════════════════════════════════════════
# MAPPING
# ═════════════════════════════════════════════════════════════════

def consensus_mapping(consensus_dir, barcode_template, left_coding_flank, right_coding_flank,
                      mapping_fn, reference_seqs=None, ref_seq_col=None, ref_name_col=None,
                      max_edits_compressed=3, max_edits_full=5, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.1))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.1))
    bar_fuzz = max(1, int(len(barcode_template) * 0.005))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 consensus_dir, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn,
                 reference_seqs=reference_seqs, ref_seq_col=ref_seq_col,
                 ref_name_col=ref_name_col, max_edits_compressed=max_edits_compressed,
                 max_edits_full=max_edits_full)


def direct_mapping(fn, barcode_template, left_coding_flank, right_coding_flank,
                   mapping_fn, reference_seqs=None, ref_seq_col=None, ref_name_col=None,
                   max_edits_compressed=3, max_edits_full=5, **kwargs):
    left_coding_flank = left_coding_flank.upper()
    right_coding_flank = right_coding_flank.upper()
    barcode_template = barcode_template.upper()

    left_fuzz = max(1, int(len(left_coding_flank) * 0.1))
    right_fuzz = max(1, int(len(right_coding_flank) * 0.1))
    bar_fuzz = max(1, int(len(barcode_template) * 0.01))

    map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 fn, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn,
                 reference_seqs=reference_seqs, ref_seq_col=ref_seq_col,
                 ref_name_col=ref_name_col, max_edits_compressed=max_edits_compressed,
                 max_edits_full=max_edits_full)


def map_barcodes(left_fuzz, right_fuzz, bar_fuzz,
                 input_files, barcode_template, left_coding_flank,
                 right_coding_flank, mapping_fn,
                 reference_seqs=None, ref_seq_col=None, ref_name_col=None,
                 max_edits_compressed=3, max_edits_full=5, **kwargs):

    # ── Load reference index if provided ─────────────────────────
    hp_index = None
    if reference_seqs:
        print(f"Loading reference sequences from {reference_seqs}...")
        refs = load_references_from_file(reference_seqs, seq_col=ref_seq_col, name_col=ref_name_col)
        print(f"  Loaded {len(refs)} reference sequences")
        hp_index = build_reference_index(refs)
        print(f"  Reference snapping: ON "
              f"(max_edits_compressed={max_edits_compressed}, max_edits_full={max_edits_full})")
    else:
        print("  Reference snapping: OFF (no --reference_seqs provided)")

    if isdir(input_files):
        consensus_files = glob(f"{input_files}/*/cluster_*_consensus.fasta")
        print(f"{input_files}/*/cluster_*_consensus.fasta")
        print(f"Determining mapping for {len(consensus_files)} consensus sequences")

        if len(consensus_files) == 0:
            consensus_files = glob(f"{input_files}/consensus/*/cluster_*_consensus.fasta")
            if len(consensus_files) == 0:
                raise Exception(
                    f"No consensus sequences found in {input_files}. "
                    f"Consider altering hyperparameters or doing deeper sequencing.")
    else:
        consensus_files = [input_files]

    matchers = make_orientation_matchers(barcode_template, left_coding_flank, right_coding_flank,
                                         left_fuzz, right_fuzz, bar_fuzz)
    orientation_counts = {}
    no_match_count = 0
    snapped_count = 0
    no_snap_count = 0
    observations = 0
    mapping_fn = '.'.join(mapping_fn.split('.')[:-1]) + '.tsv'

    with open(mapping_fn, "w") as out:
        out.write("filename\tbarcode\tcoding_region\n")
        for file in sorted(consensus_files):
            for record in SeqIO.parse(file, file.split('.')[-1]):
                seq = str(record.seq).upper()
                barcode, coding_region, orientation_counts, orientation = match_with_orientation(
                    seq, matchers, orientation_counts
                )
                if barcode:
                    if orientation == 'B':
                        barcode = reverse_complement(barcode)
                        coding_region = reverse_complement(coding_region)

                    # ── Snap coding region to nearest reference ───
                    if hp_index is not None and coding_region:
                        ref_name, ref_seq = snap_to_reference(
                            coding_region, hp_index,
                            max_edits_compressed=max_edits_compressed,
                            max_edits_full=max_edits_full,
                        )
                        if ref_seq is not None:
                            coding_region = ref_seq
                            snapped_count += 1
                        else:
                            no_snap_count += 1

                    out.write(f"{file}\t{barcode}\t{coding_region}\n")
                else:
                    no_match_count += 1
                observations += 1

    print(f"Found a match for {observations - no_match_count}/{observations} sequences")
    if hp_index is not None:
        total_mapped = observations - no_match_count
        print(f"  Snapped to reference:    {snapped_count}/{total_mapped} "
              f"({100 * snapped_count / max(total_mapped, 1):.1f}%)")
        print(f"  No reference snap:       {no_snap_count}/{total_mapped} "
              f"({100 * no_snap_count / max(total_mapped, 1):.1f}%)")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main():
    direct = False

    if direct:
        parser = argparse.ArgumentParser()
        parser.add_argument('--barcode_template', type=str, default=None)
        parser.add_argument('--fn', type=str, default=None)
        parser.add_argument('--left_coding_flank', type=str, default=None)
        parser.add_argument('--right_coding_flank', type=str, default=None)
        parser.add_argument('--mapping_fn', type=str, default=None)
        parser.add_argument('--reference_seqs', type=str, default=None)
        parser.add_argument('--ref_seq_col', type=str, default=None)
        parser.add_argument('--ref_name_col', type=str, default=None)
        parser.add_argument('--max_edits_compressed', type=int, default=3)
        parser.add_argument('--max_edits_full', type=int, default=5)

        all_args = parser.parse_known_args()
        args = all_args[0]
        direct_mapping(**vars(args))

    else:
        parser = argparse.ArgumentParser()

        parser.add_argument('--consensus_dir', type=str, default=None, required=not direct)
        parser.add_argument("--mapping_fn", default=None, required=True)
        parser.add_argument('--barcode_template', type=str, required=True, default=None)
        parser.add_argument("--left_coding_flank", default=None, required=True)
        parser.add_argument("--right_coding_flank", default=None, required=True)

        # Reference snapping args
        parser.add_argument("--reference_seqs", type=str, default=None,
                            help="Path to reference sequences (FASTA, CSV, or TSV) for coding "
                                 "region snapping. If provided, extracted coding regions will be "
                                 "replaced with the nearest reference sequence within edit distance "
                                 "thresholds.")
        parser.add_argument("--ref_seq_col", type=str, default=None,
                            help="Column name for sequences in reference CSV/TSV")
        parser.add_argument("--ref_name_col", type=str, default=None,
                            help="Column name for sequence names in reference CSV/TSV")
        parser.add_argument("--max_edits_compressed", type=int, default=3,
                            help="Max edit distance on HP-compressed sequences for reference "
                                 "snapping pre-filter (default: 3)")
        parser.add_argument("--max_edits_full", type=int, default=5,
                            help="Max full edit distance to accept a reference snap (default: 5)")

        all_args = parser.parse_known_args()
        args = all_args[0]

        args.barcode_directory = args.consensus_dir.split('/consensus')[-1].split('/')[-1]
        args.output_dir = f'temp/{args.barcode_directory}/'
        args.cluster_dir = args.output_dir + '/clusters/'

        print('Mapping barcodes to coding sequences...')
        mapping_start_time = time.time()
        consensus_mapping(**vars(args))
        mapping_time = time.time() - mapping_start_time
        print(f'Finished mapping barcodes in {round(mapping_time / 60, 1)} minutes\n')


def cli():
    main()


if __name__ == "__main__":
    cli()