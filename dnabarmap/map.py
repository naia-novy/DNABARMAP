from glob import glob
import regex
import re
import csv
from Bio import SeqIO
from os.path import isdir
from pathlib import Path
from collections import Counter, defaultdict
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
    import pandas as pd
    if ext == '.pkl':
        df = pd.read_pickle(ref_path)
    else:
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
    seen_sequences = {}
    for i, row in df.iterrows():
        seq = str(row[s_col]).upper().strip()
        if not seq or seq == 'NAN':
            continue
        if seq in seen_sequences:
            continue
        name = str(row[n_col]) if n_col else f"ref_{i}"
        if name in refs:
            name = f"{name}_{i}"
        refs[name] = seq
        seen_sequences[seq] = name

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


def _template_match_fraction(subseq, template):
    if len(subseq) != len(template) or not template:
        return 0.0
    matches = 0
    for base, code in zip(subseq.upper(), template.upper()):
        if base in nuc_dict[code]:
            matches += 1
    return matches / len(template)


def _find_best_barcode_window(seq, barcode_template):
    template = barcode_template.upper()
    best = None

    for orientation, work_seq in (('fwd', seq.upper()), ('rc', reverse_complement(seq))):
        if len(work_seq) < len(template):
            continue
        for start in range(len(work_seq) - len(template) + 1):
            end = start + len(template)
            score = _template_match_fraction(work_seq[start:end], template)
            if best is None or score > best['score']:
                best = {
                    'orientation': orientation,
                    'work_seq': work_seq,
                    'start': start,
                    'end': end,
                    'score': score,
                }

    return best


def _find_exact_positions(seq, motif):
    positions = []
    start = 0
    while True:
        pos = seq.find(motif, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _collect_flank_candidates(seq, motif, start_min=0, start_max=None,
                              max_edits=2, preferred_start=None, limit=5):
    motif = motif.upper()
    seq = seq.upper()
    motif_len = len(motif)
    if motif_len == 0 or len(seq) < motif_len:
        return []

    start_min = max(0, start_min)
    max_valid_start = len(seq) - motif_len
    if start_max is None:
        start_max = max_valid_start
    else:
        start_max = min(start_max, max_valid_start)
    if start_min > start_max:
        return []

    candidates = []
    for pos in _find_exact_positions(seq, motif):
        if start_min <= pos <= start_max:
            candidates.append((pos, 0))

    if candidates:
        return sorted(
            candidates,
            key=lambda item: (item[1], abs(item[0] - (preferred_start if preferred_start is not None else item[0])))
        )[:limit]

    for pos in range(start_min, start_max + 1):
        window = seq[pos:pos + motif_len]
        edits = _edit_distance_banded(window, motif, max_dist=max_edits)
        if edits <= max_edits:
            candidates.append((pos, edits))

    candidates.sort(
        key=lambda item: (item[1], abs(item[0] - (preferred_start if preferred_start is not None else item[0])))
    )
    return candidates[:limit]


def _best_reference_match(query_seq, refs_by_len, max_dist=60):
    best = None
    second_best = None
    query_len = len(query_seq)

    for ref_len, ref_group in refs_by_len.items():
        if abs(ref_len - query_len) > max_dist:
            continue
        for name, ref_seq in ref_group:
            dist = _edit_distance_banded(query_seq, ref_seq, max_dist=max_dist)
            if dist > max_dist:
                continue
            candidate = (dist, name, ref_seq)
            if best is None or dist < best[0]:
                second_best = best
                best = candidate
            elif second_best is None or dist < second_best[0]:
                second_best = candidate

    if best is None:
        return None

    return {
        'dist': best[0],
        'name': best[1],
        'seq': best[2],
        'second_dist': second_best[0] if second_best is not None else None,
    }


def _extract_with_reference_guidance(seq, barcode_template, left_coding_flank,
                                     right_coding_flank, refs_by_len,
                                     max_edits_full=5):
    barcode_hit = _find_best_barcode_window(seq, barcode_template)
    if barcode_hit is None or barcode_hit['score'] < 0.72:
        return None

    work_seq = barcode_hit['work_seq']
    barcode_start = barcode_hit['start']
    barcode_end = barcode_hit['end']
    barcode = work_seq[barcode_start:barcode_end]

    left_len = len(left_coding_flank)
    right_len = len(right_coding_flank)
    max_barcode_gap = 260
    max_shift = 30
    max_flank_edits = max(3, max(len(left_coding_flank), len(right_coding_flank)) // 3)
    max_ref_dist = max(max_edits_full, 25)
    best = None

    def consider_candidate(coding_start, coding_end, left_edits, right_edits,
                           barcode_gap, delta):
        nonlocal best
        if coding_start < 0 or coding_end <= coding_start or coding_end > len(work_seq):
            return
        coding_region = work_seq[coding_start:coding_end]
        ref_match = _best_reference_match(coding_region, refs_by_len, max_dist=max_ref_dist)
        if ref_match is None:
            return

        score = (
            ref_match['dist'],
            left_edits + right_edits,
            abs(delta),
            barcode_gap,
        )
        if best is None or score < best['score']:
            best = {
                'barcode': barcode,
                'coding_region': coding_region,
                'ref_match': ref_match,
                'score': score,
            }

    # Barcode ... left_flank [coding] right_flank
    left_candidates = _collect_flank_candidates(
        work_seq,
        left_coding_flank,
        start_min=barcode_end,
        start_max=min(len(work_seq) - left_len, barcode_end + max_barcode_gap),
        max_edits=max_flank_edits,
        preferred_start=barcode_end,
        limit=5,
    )
    for left_start, left_edits in left_candidates:
        coding_start = left_start + left_len
        for ref_len in sorted(refs_by_len):
            for delta in range(-max_shift, max_shift + 1):
                coding_end = coding_start + ref_len + delta
                if coding_end + right_len > len(work_seq):
                    continue
                right_window = work_seq[coding_end:coding_end + right_len]
                right_edits = _edit_distance_banded(
                    right_window,
                    right_coding_flank,
                    max_dist=max_flank_edits,
                )
                if right_edits <= max_flank_edits:
                    consider_candidate(
                        coding_start,
                        coding_end,
                        left_edits,
                        right_edits,
                        left_start - barcode_end,
                        delta,
                    )

    # left_flank [coding] right_flank ... barcode
    right_candidates = _collect_flank_candidates(
        work_seq,
        right_coding_flank,
        start_min=max(0, barcode_start - max_barcode_gap),
        start_max=barcode_start,
        max_edits=max_flank_edits,
        preferred_start=barcode_start,
        limit=5,
    )
    for right_start, right_edits in right_candidates:
        for ref_len in sorted(refs_by_len):
            for delta in range(-max_shift, max_shift + 1):
                coding_end = right_start
                coding_start = coding_end - (ref_len + delta)
                left_start = coding_start - left_len
                if left_start < 0:
                    continue
                left_window = work_seq[left_start:left_start + left_len]
                left_edits = _edit_distance_banded(
                    left_window,
                    left_coding_flank,
                    max_dist=max_flank_edits,
                )
                if left_edits <= max_flank_edits:
                    consider_candidate(
                        coding_start,
                        coding_end,
                        left_edits,
                        right_edits,
                        barcode_start - right_start,
                        delta,
                    )

    if best is None:
        return None

    ref_match = best['ref_match']
    second_dist = ref_match.get('second_dist')
    if ref_match['dist'] > max_ref_dist:
        return None
    if second_dist is not None and second_dist - ref_match['dist'] < 2:
        return None

    return {
        'barcode': best['barcode'],
        'coding_region': ref_match['seq'],
        'orientation': 'A',
        'ref_name': ref_match['name'],
        'snapped': True,
    }


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


def _find_motif_candidates(seq, motif, max_mismatches, start=0, end=None):
    seq = seq.upper()
    motif = motif.upper()
    motif_len = len(motif)
    if motif_len == 0 or len(seq) < motif_len:
        return []

    stop = len(seq) if end is None else min(len(seq), end)
    candidates = []
    for pos in range(max(0, start), max(0, stop - motif_len + 1)):
        window = seq[pos:pos + motif_len]
        mismatches = sum(a != b for a, b in zip(window, motif))
        if mismatches <= max_mismatches:
            candidates.append((pos, mismatches))
    return candidates


def _infer_cluster_fastq(consensus_file):
    consensus_path = Path(consensus_file)
    full_seq_parts = []
    replaced = False
    for part in consensus_path.parts:
        if part == 'consensus' and not replaced:
            full_seq_parts.extend(['clusters', 'full_seqs'])
            replaced = True
        else:
            full_seq_parts.append(part)

    if not replaced:
        return None

    cluster_fastq = Path(*full_seq_parts)
    cluster_fastq = cluster_fastq.with_name(cluster_fastq.name.replace('_consensus.fasta', '.fastq'))
    return cluster_fastq if cluster_fastq.exists() else None


def _infer_reference_from_cluster_reads(consensus_file, left_coding_flank,
                                        right_coding_flank, refs_by_len,
                                        max_flank_mismatches=3,
                                        max_ref_dist=160):
    cluster_fastq = _infer_cluster_fastq(consensus_file)
    if cluster_fastq is None:
        return None

    votes = Counter()
    informative = 0
    left_len = len(left_coding_flank)
    right_len = len(right_coding_flank)

    for record in SeqIO.parse(str(cluster_fastq), "fastq"):
        seq = str(record.seq).upper()
        left_candidates = _find_motif_candidates(seq, left_coding_flank, max_flank_mismatches)
        right_candidates = _find_motif_candidates(seq, right_coding_flank, max_flank_mismatches)
        if not left_candidates or not right_candidates:
            continue

        best_pair = None
        for left_pos, left_mm in left_candidates:
            for right_pos, right_mm in right_candidates:
                coding_len = right_pos - (left_pos + left_len)
                if coding_len < 200:
                    continue
                tail_len = len(seq) - (right_pos + right_len)
                pair_score = (
                    left_mm + right_mm,
                    abs(tail_len),
                    -coding_len,
                )
                if best_pair is None or pair_score < best_pair[0]:
                    best_pair = (pair_score, left_pos, right_pos)

        if best_pair is None:
            continue

        _, left_pos, right_pos = best_pair
        coding_region = seq[left_pos + left_len:right_pos]
        ref_match = _best_reference_match(coding_region, refs_by_len, max_dist=max_ref_dist)
        if ref_match is None:
            continue

        informative += 1
        votes[ref_match['seq']] += 1

    if informative == 0 or not votes:
        return None

    ranked = votes.most_common(2)
    top_seq, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0

    if top_count < max(3, second_count + 2):
        return None

    return top_seq


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
    refs_by_len = None
    if reference_seqs:
        print(f"Loading reference sequences from {reference_seqs}...")
        refs = load_references_from_file(reference_seqs, seq_col=ref_seq_col, name_col=ref_name_col)
        print(f"  Loaded {len(refs)} reference sequences")
        hp_index = build_reference_index(refs)
        refs_by_len = defaultdict(list)
        for name, ref_seq in refs.items():
            refs_by_len[len(ref_seq)].append((name, ref_seq))
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
                guided = None
                if refs_by_len is not None:
                    guided = _extract_with_reference_guidance(
                        seq,
                        barcode_template,
                        left_coding_flank,
                        right_coding_flank,
                        refs_by_len,
                        max_edits_full=max_edits_full,
                    )

                if guided is not None:
                    barcode = guided['barcode']
                    coding_region = guided['coding_region']
                    orientation = guided['orientation']
                    snapped_count += 1
                else:
                    barcode, coding_region, orientation_counts, orientation = match_with_orientation(
                        seq, matchers, orientation_counts
                    )
                if barcode:
                    if orientation == 'B':
                        barcode = reverse_complement(barcode)
                        coding_region = reverse_complement(coding_region)

                    # ── Snap coding region to nearest reference ───
                    if hp_index is not None and coding_region and guided is None:
                        ref_name, ref_seq = snap_to_reference(
                            coding_region, hp_index,
                            max_edits_compressed=max_edits_compressed,
                            max_edits_full=max_edits_full,
                        )
                        if ref_seq is not None:
                            coding_region = ref_seq
                            snapped_count += 1
                        else:
                            voted_ref = None
                            if refs_by_len is not None:
                                voted_ref = _infer_reference_from_cluster_reads(
                                    file,
                                    left_coding_flank,
                                    right_coding_flank,
                                    refs_by_len,
                                )
                            if voted_ref is not None:
                                coding_region = voted_ref
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
        parser = argparse.ArgumentParser(
            description="Map a single consensus FASTA directly to barcode and "
                        "coding-region calls."
        )
        parser.add_argument('--barcode_template', type=str, default=None,
                            help="Degenerate barcode template used to locate "
                                 "the barcode interval.")
        parser.add_argument('--fn', type=str, default=None,
                            help="Consensus FASTA file to map.")
        parser.add_argument('--left_coding_flank', type=str, default=None,
                            help="Constant sequence immediately left of the "
                                 "coding region.")
        parser.add_argument('--right_coding_flank', type=str, default=None,
                            help="Constant sequence immediately right of the "
                                 "coding region.")
        parser.add_argument('--mapping_fn', type=str, default=None,
                            help="Output TSV filename for mapping results.")
        parser.add_argument('--reference_seqs', type=str, default=None,
                            help="Optional reference sequences for coding-region snapping.")
        parser.add_argument('--ref_seq_col', type=str, default=None,
                            help="Sequence column to use when --reference_seqs "
                                 "points to a table or pickle.")
        parser.add_argument('--ref_name_col', type=str, default=None,
                            help="Name column to use when --reference_seqs "
                                 "points to a table or pickle.")
        parser.add_argument('--max_edits_compressed', type=int, default=3,
                            help="Maximum edit distance in homopolymer-compressed "
                                 "space for reference snapping prefilter.")
        parser.add_argument('--max_edits_full', type=int, default=5,
                            help="Maximum full-length edit distance to accept a "
                                 "reference snap.")

        all_args = parser.parse_known_args()
        args = all_args[0]
        direct_mapping(**vars(args))

    else:
        parser = argparse.ArgumentParser(
            description="Map consensus sequences to barcode and coding-region "
                        "calls, with optional snapping to a reference set."
        )

        parser.add_argument('--consensus_dir', type=str, default=None, required=not direct,
                            help="Directory containing cluster_*_consensus.fasta "
                                 "files to map.")
        parser.add_argument("--mapping_fn", default=None, required=True,
                            help="Output TSV filename for the final mapping results.")
        parser.add_argument('--barcode_template', type=str, required=True, default=None,
                            help="Degenerate barcode template used to locate the "
                                 "barcode interval in each consensus.")
        parser.add_argument("--left_coding_flank", default=None, required=True,
                            help="Constant sequence immediately left of the "
                                 "coding region.")
        parser.add_argument("--right_coding_flank", default=None, required=True,
                            help="Constant sequence immediately right of the "
                                 "coding region.")

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
