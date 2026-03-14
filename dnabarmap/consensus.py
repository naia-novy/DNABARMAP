#!/usr/bin/env python3
"""
Generate consensus sequences from nanopore cluster FASTQs.

Pipeline per cluster:
  1. minimap2 self-map (all-vs-all)
  2. Intra-cluster purity filter: extract dominant sub-cluster of
     mutually similar reads, discard chimeras/contaminants
  3. Pick best draft read from filtered set
  4. Optional Medaka polishing
  5. Optional barcode interval calling for diagnostics/flank anchoring

Clusters are processed in parallel via ProcessPoolExecutor.
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from os import remove, makedirs, path
from pathlib import Path
from glob import glob
import subprocess
import argparse
import time
import shutil
import re
import csv
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import statistics
import sys

TIMEOUT = 3600  # 1 hour timeout for all subprocess calls
DEFAULT_MIN_OVERLAP_FRACTION = 0.6
DEFAULT_MIN_ALIGNED_BASES = 100


# ═════════════════════════════════════════════════════════════════
# IUPAC DEGENERATE BASE LOOKUP
# ═════════════════════════════════════════════════════════════════

IUPAC = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'},
    'W': {'A', 'T'}, 'K': {'G', 'T'}, 'M': {'A', 'C'},
    'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
    'N': {'A', 'C', 'G', 'T'},
}


# ═════════════════════════════════════════════════════════════════
# INTRA-CLUSTER PURITY FILTERING
# ═════════════════════════════════════════════════════════════════

def _parse_paf_overlaps(paf_file, min_identity=0.85,
                        min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                        min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES):
    """
    Parse a PAF file and return high-confidence read overlaps.

    Returns:
        edges: dict of {(read_a, read_b): metrics} for best overlap per pair
    """
    edges = {}

    with open(paf_file) as f:
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 12:
                continue

            qname = cols[0]
            qlen = int(cols[1])
            qstart = int(cols[2])
            qend = int(cols[3])
            tname = cols[5]
            tlen = int(cols[6])
            tstart = int(cols[7])
            tend = int(cols[8])

            # Skip self-hits
            if qname == tname:
                continue

            residue_matches = int(cols[9])
            aln_block_len = int(cols[10])
            if aln_block_len == 0:
                continue
            identity = residue_matches / aln_block_len
            if identity < min_identity:
                continue

            shorter_read_len = min(qlen, tlen)
            if shorter_read_len == 0:
                continue

            q_span = max(0, qend - qstart)
            t_span = max(0, tend - tstart)
            overlap_fraction = min(q_span, t_span) / shorter_read_len

            if overlap_fraction < min_overlap_fraction:
                continue
            if aln_block_len < min_aligned_bases:
                continue

            score = identity * aln_block_len * overlap_fraction
            pair = tuple(sorted((qname, tname)))
            metrics = {
                'identity': identity,
                'aligned_bases': aln_block_len,
                'overlap_fraction': overlap_fraction,
                'score': score,
            }

            # Keep the strongest overlap for each unordered pair.
            if pair not in edges or score > edges[pair]['score']:
                edges[pair] = metrics

    return edges


def _find_dominant_subcluster(read_ids, edges, min_identity=0.85,
                              min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION):
    """
    Given all read IDs in a cluster and their pairwise identities from
    the PAF, find the largest group of reads connected at >= min_identity.

    Uses connected-component extraction via BFS:
      1. Build adjacency list of reads connected above threshold
      2. Find all connected components
      3. Return the largest component as the dominant sub-cluster

    This is not full clique-finding (NP-hard) but connected-component
    extraction, which is fast and sufficient — if two chimeric reads
    happen to both align well to a legitimate read, they'll be included,
    but the barcode snap step downstream catches remaining errors.
    """
    if len(read_ids) <= 2:
        return set(read_ids)

    # Build adjacency at threshold
    adj = defaultdict(set)
    read_set = set(read_ids)

    for (q, t), metrics in edges.items():
        if (q in read_set and t in read_set and
                metrics['identity'] >= min_identity and
                metrics['overlap_fraction'] >= min_overlap_fraction):
            adj[q].add(t)
            adj[t].add(q)

    if not adj:
        # No overlap graph means the cluster is incoherent.
        return set()

    # Find all connected components via BFS
    visited = set()
    components = []

    for seed in read_ids:
        if seed in visited:
            continue
        component = set()
        queue = [seed]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj.get(node, set()) - visited)
        components.append(component)

    # Return the largest component
    components.sort(key=len, reverse=True)
    return components[0]


def _compute_draft_support(read_ids, edges):
    """
    Score each kept read by the amount of overlap support it receives from
    other kept reads. This keeps contaminants from influencing draft choice.
    """
    support = {}
    for read_id in read_ids:
        support[read_id] = 0.0

    read_set = set(read_ids)
    for (q, t), metrics in edges.items():
        if q in read_set and t in read_set:
            support[q] += metrics['score']
            support[t] += metrics['score']

    return support


def filter_cluster_reads(input_fq, paf_file, min_identity=0.85,
                         min_dominant_fraction=0.5,
                         min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                         min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES):
    """
    Parse PAF, find dominant sub-cluster, write filtered FASTQ.

    Returns:
        filtered_fq: path to filtered FASTQ (or None if cluster fails purity)
        draft_support: per-read support within the dominant sub-cluster
        filter_stats: dict with filtering statistics
    """
    # Get all read IDs from the FASTQ
    all_read_ids = []
    for record in SeqIO.parse(input_fq, "fastq"):
        all_read_ids.append(record.id)

    n_total = len(all_read_ids)

    if n_total <= 2:
        draft_support = {read_id: 0.0 for read_id in all_read_ids}
        return input_fq, draft_support, {
            'n_total': n_total, 'n_kept': n_total,
            'fraction_kept': 1.0, 'filtered': False,
            'n_overlap_edges': 0,
        }

    edges = _parse_paf_overlaps(
        paf_file,
        min_identity=min_identity,
        min_overlap_fraction=min_overlap_fraction,
        min_aligned_bases=min_aligned_bases,
    )

    dominant = _find_dominant_subcluster(all_read_ids, edges,
                                         min_identity=min_identity,
                                         min_overlap_fraction=min_overlap_fraction)
    fraction = len(dominant) / n_total

    stats = {
        'n_total': n_total,
        'n_kept': len(dominant),
        'fraction_kept': fraction,
        'filtered': True,
        'n_overlap_edges': len(edges),
    }

    if not dominant:
        stats['rejected'] = True
        stats['reason'] = ("no usable overlaps passed the identity/coverage "
                           "filters")
        return None, {}, stats

    # Check purity threshold
    if fraction < min_dominant_fraction:
        stats['rejected'] = True
        stats['reason'] = (f"dominant sub-cluster too small: "
                           f"{len(dominant)}/{n_total} = {fraction:.2f} "
                           f"< {min_dominant_fraction}")
        return None, {}, stats

    stats['rejected'] = False
    draft_support = _compute_draft_support(dominant, edges)

    # If all reads are in the dominant cluster, no need to write a new file
    if len(dominant) == n_total:
        stats['filtered'] = False
        return input_fq, draft_support, stats

    # Write filtered FASTQ
    filtered_fq = input_fq.replace('.fastq', '_filtered.fastq')
    kept = 0
    with open(filtered_fq, 'w') as out_handle:
        for record in SeqIO.parse(input_fq, "fastq"):
            if record.id in dominant:
                SeqIO.write(record, out_handle, "fastq")
                kept += 1

    return filtered_fq, draft_support, stats


# ═════════════════════════════════════════════════════════════════
# BARCODE TEMPLATE SNAPPING (NO FLANKS REQUIRED)
# ═════════════════════════════════════════════════════════════════

def _template_match_score(seq, template):
    if len(seq) != len(template):
        return 0.0
    matches = sum(1 for s, t in zip(seq.upper(), template.upper())
                  if s in IUPAC.get(t, {t}))
    return matches / len(template)


def _reverse_complement_template(template):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K', 'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D', 'N': 'N',
    }
    return ''.join(complement.get(b, 'N') for b in reversed(template.upper()))


def _find_barcode_region(consensus_seq, template, search_margin=3,
                         allow_reverse_complement=False):
    template_len = len(template)
    seq_upper = consensus_seq.upper()
    seq_len = len(seq_upper)

    best_start = best_end = -1
    best_score = -1.0
    best_orient = None
    best_tmpl = template

    orientations = [(template, 'fwd')]
    if allow_reverse_complement:
        orientations.append((_reverse_complement_template(template), 'rc'))

    for tmpl, orient in orientations:
        # ── Try exact-length windows first (fast path) ───────────
        for start in range(seq_len - template_len + 1):
            subseq = seq_upper[start:start + template_len]
            score = _template_match_score(subseq, tmpl)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = start + template_len
                best_orient = orient
                best_tmpl = tmpl

        # ── Only try variable-length windows if exact-length
        #    didn't find a good match ─────────────────────────────
        if best_score >= 0.85:
            continue

        for window_len in range(template_len - search_margin,
                                template_len + search_margin + 1):
            if window_len < 1 or window_len > seq_len or window_len == template_len:
                continue
            for start in range(seq_len - window_len + 1):
                subseq = seq_upper[start:start + window_len]
                _, _, aln_score = _nw_align_to_template(subseq, tmpl)
                norm_score = aln_score / (template_len * 2)
                if norm_score > best_score:
                    best_score = norm_score
                    best_start = start
                    best_end = start + window_len
                    best_orient = orient
                    best_tmpl = tmpl

    return best_start, best_end, best_orient, best_score, best_tmpl


def _nw_align_to_template(query, template, match_score=2, mismatch_penalty=-1,
                           gap_penalty=-2):
    n = len(query)
    m = len(template)

    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
        trace[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            qbase = query[i - 1].upper()
            tcode = template[j - 1].upper()
            allowed = IUPAC.get(tcode, {tcode})

            s = match_score if qbase in allowed else mismatch_penalty

            diag = score[i - 1][j - 1] + s
            up   = score[i - 1][j] + gap_penalty
            left = score[i][j - 1] + gap_penalty

            best = max(diag, up, left)
            score[i][j] = best

            if best == diag:
                trace[i][j] = 0
            elif best == up:
                trace[i][j] = 1
            else:
                trace[i][j] = 2

    aligned_q, aligned_t = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            aligned_q.append(query[i - 1])
            aligned_t.append(template[j - 1])
            i -= 1; j -= 1
        elif i > 0 and trace[i][j] == 1:
            aligned_q.append(query[i - 1])
            aligned_t.append('-')
            i -= 1
        else:
            aligned_q.append('-')
            aligned_t.append(template[j - 1])
            j -= 1

    return ''.join(reversed(aligned_q)), ''.join(reversed(aligned_t)), score[n][m]


def _pick_best_allowed(observed, allowed):
    if observed in allowed:
        return observed
    transitions = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
    partner = transitions.get(observed)
    if partner and partner in allowed:
        return partner
    priority = {'T': 0, 'A': 1, 'G': 2, 'C': 3}
    return min(allowed, key=lambda b: priority.get(b, 99))


def snap_barcode_to_template(barcode, template, max_mismatches=5, max_indels=3):
    barcode = barcode.upper().strip()
    template = template.upper().strip()

    if not barcode:
        return None, {'rejected': True, 'reason': 'empty'}

    if abs(len(barcode) - len(template)) > max_indels:
        return None, {
            'rejected': True, 'reason': 'length',
            'barcode_len': len(barcode), 'template_len': len(template),
        }

    aligned_q, aligned_t, aln_score = _nw_align_to_template(barcode, template)

    corrected = []
    n_mismatches = n_insertions = n_deletions = n_corrected = 0

    for qbase, tcode in zip(aligned_q, aligned_t):
        if tcode == '-':
            n_insertions += 1
            continue

        allowed = IUPAC.get(tcode, {tcode})

        if qbase == '-':
            n_deletions += 1
            corrected.append(_pick_best_allowed('N', allowed))
            continue

        if qbase in allowed:
            corrected.append(qbase)
        else:
            n_mismatches += 1
            corrected.append(_pick_best_allowed(qbase, allowed))
            n_corrected += 1

    stats = {
        'rejected': False,
        'mismatches': n_mismatches,
        'insertions': n_insertions,
        'deletions': n_deletions,
        'total_edits': n_mismatches + n_insertions + n_deletions,
        'corrected_bases': n_corrected,
        'alignment_score': aln_score,
    }

    if n_mismatches > max_mismatches:
        stats['rejected'] = True
        stats['reason'] = 'too_many_mismatches'
        return None, stats

    if (n_insertions + n_deletions) > max_indels:
        stats['rejected'] = True
        stats['reason'] = 'too_many_indels'
        return None, stats

    result = ''.join(corrected)
    if len(result) != len(template):
        stats['rejected'] = True
        stats['reason'] = f'length_mismatch ({len(result)} vs {len(template)})'
        return None, stats

    return result, stats


def snap_barcode_in_consensus(consensus_seq, barcode_template,
                               max_mismatches=5, max_indels=3,
                               min_window_score=0.7,
                               allow_reverse_complement=False):
    consensus_upper = consensus_seq.upper()
    template_upper = barcode_template.upper()

    start, end, orient, window_score, effective_template = _find_barcode_region(
        consensus_upper, template_upper,
        search_margin=max_indels,
        allow_reverse_complement=allow_reverse_complement,
    )

    if start < 0 or window_score < min_window_score:
        return None, {
            'rejected': True,
            'reason': f'no_match (best_score={window_score:.3f}, '
                      f'threshold={min_window_score})',
        }

    raw_barcode = consensus_upper[start:end]

    corrected_bc, stats = snap_barcode_to_template(
        raw_barcode, effective_template,
        max_mismatches=max_mismatches,
        max_indels=max_indels,
    )

    if corrected_bc is None:
        return None, stats

    stats['orientation'] = orient
    stats['window_score'] = window_score
    stats['barcode_start'] = start
    stats['barcode_end'] = end

    corrected_full = consensus_upper[:start] + corrected_bc + consensus_upper[end:]
    return corrected_full, stats


def _normalize_barcode_to_template(barcode, template):
    """
    Align a raw barcode window to the template length while preserving the
    observed read bases. Insertions are dropped and deletions become 'N'.
    """
    aligned_q, aligned_t, aln_score = _nw_align_to_template(barcode, template)

    normalized = []
    n_insertions = 0
    n_deletions = 0

    for qbase, tcode in zip(aligned_q, aligned_t):
        if tcode == '-':
            n_insertions += 1
            continue
        if qbase == '-':
            n_deletions += 1
            normalized.append('N')
        else:
            normalized.append(qbase.upper())

    return ''.join(normalized), {
        'insertions': n_insertions,
        'deletions': n_deletions,
        'alignment_score': aln_score,
    }


def _find_barcode_region_fast(consensus_seq, template,
                              allow_reverse_complement=False):
    template_len = len(template)
    seq_upper = consensus_seq.upper()
    seq_len = len(seq_upper)

    if seq_len < template_len:
        return -1, -1, None, -1.0, template

    best_start = best_end = -1
    best_score = -1.0
    best_orient = None
    best_tmpl = template

    orientations = [(template, 'fwd')]
    if allow_reverse_complement:
        orientations.append((_reverse_complement_template(template), 'rc'))

    for tmpl, orient in orientations:
        for start in range(seq_len - template_len + 1):
            subseq = seq_upper[start:start + template_len]
            score = _template_match_score(subseq, tmpl)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = start + template_len
                best_orient = orient
                best_tmpl = tmpl

    return best_start, best_end, best_orient, best_score, best_tmpl


def _extract_read_barcode_signature(seq, barcode_template,
                                    min_window_score=0.55,
                                    search_margin=6):
    seq_upper = seq.upper()
    template_upper = barcode_template.upper()

    start, end, orient, window_score, effective_template = _find_barcode_region_fast(
        seq_upper,
        template_upper,
        allow_reverse_complement=False,
    )

    if start < 0 or window_score < min_window_score:
        return None

    raw_barcode = seq_upper[start:end]
    normalized, norm_stats = _normalize_barcode_to_template(
        raw_barcode,
        effective_template,
    )

    return {
        'barcode_start': start,
        'barcode_end': end,
        'window_score': window_score,
        'orientation': orient,
        'raw_barcode': raw_barcode,
        'normalized_barcode': normalized,
        'template': effective_template,
        **norm_stats,
    }


def _build_signature_barcode_consensus(input_fq, barcode_template,
                                       min_window_score=0.55,
                                       max_mismatches=5, max_indels=3):
    stats = {
        'n_used': 0,
        'rejected': False,
    }

    if not input_fq or not path.exists(input_fq):
        stats['rejected'] = True
        stats['reason'] = 'missing_cluster_reads'
        return None, stats

    normalized_barcodes = []
    relaxed_window = min(min_window_score, 0.55)
    search_margin = max(max_indels * 2, 6)

    for record in SeqIO.parse(input_fq, "fastq"):
        signature = _extract_read_barcode_signature(
            str(record.seq),
            barcode_template,
            min_window_score=relaxed_window,
            search_margin=search_margin,
        )
        if signature is None:
            continue

        corrected_bc, _ = snap_barcode_to_template(
            signature['raw_barcode'],
            signature['template'],
            max_mismatches=max(max_mismatches * 2, 10),
            max_indels=max(max_indels * 2, 8),
        )
        normalized_barcodes.append(
            corrected_bc if corrected_bc is not None else signature['normalized_barcode']
        )

    stats['n_used'] = len(normalized_barcodes)
    if not normalized_barcodes:
        stats['rejected'] = True
        stats['reason'] = 'no_signature_barcodes'
        return None, stats

    top_counts = []
    coverages = []
    consensus = []
    width = len(normalized_barcodes[0])
    for idx in range(width):
        counts = Counter(base for base in (bc[idx] for bc in normalized_barcodes) if base != 'N')
        if counts:
            top_base, top_count = counts.most_common(1)[0]
            consensus.append(top_base)
            top_counts.append(top_count)
            coverages.append(sum(counts.values()))
        else:
            consensus.append('N')
            top_counts.append(0)
            coverages.append(0)

    consensus_barcode = ''.join(consensus)
    corrected_consensus, snap_stats = snap_barcode_to_template(
        consensus_barcode,
        barcode_template,
        max_mismatches=max(max_mismatches * 2, 10),
        max_indels=max(max_indels * 2, 8),
    )
    if corrected_consensus is not None:
        consensus_barcode = corrected_consensus
        stats['total_edits'] = snap_stats.get('total_edits', 0)

    stats['top_fractions'] = [
        top / max(1, cov)
        for top, cov in zip(top_counts, coverages)
    ]
    return consensus_barcode, stats


def _apply_signature_edge_correction(barcode_seq, signature_barcode,
                                     signature_barcode_stats,
                                     min_tail_fraction=0.75):
    if (not barcode_seq or not signature_barcode or
            not signature_barcode_stats or
            signature_barcode_stats.get('rejected') or
            signature_barcode == barcode_seq):
        return barcode_seq, None

    diffs = [
        idx for idx, (a, b) in enumerate(zip(barcode_seq, signature_barcode))
        if a != b
    ]
    if len(diffs) != 1:
        return barcode_seq, None

    diff_idx = diffs[0]
    top_fractions = signature_barcode_stats.get('top_fractions', [])
    diff_frac = top_fractions[diff_idx] if diff_idx < len(top_fractions) else 0.0

    if diff_idx == len(barcode_seq) - 1 and diff_frac >= min_tail_fraction:
        return barcode_seq[:-1] + signature_barcode[-1], {
            'kind': 'tail_vote',
            'frac': diff_frac,
        }

    return barcode_seq, None


def _barcode_distance(barcode_a, barcode_b, min_comparable=50):
    mismatches = 0
    comparable = 0

    for base_a, base_b in zip(barcode_a, barcode_b):
        if base_a == 'N' or base_b == 'N':
            continue
        comparable += 1
        if base_a != base_b:
            mismatches += 1

    if comparable < min_comparable:
        return None, comparable

    return mismatches, comparable


def _barcode_consensus_signature(barcodes):
    if not barcodes:
        return None

    consensus = []
    width = len(barcodes[0])
    for i in range(width):
        counts = Counter(base for base in (bc[i] for bc in barcodes) if base != 'N')
        if counts:
            consensus.append(counts.most_common(1)[0][0])
        else:
            consensus.append('N')

    return ''.join(consensus)


def _iter_cigar_ops(cigar):
    for length, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar):
        yield int(length), op


def _pileup_barcode_columns(query_seq, qstart, tstart, cigar,
                            interval_start, interval_end, columns):
    qpos = qstart
    tpos = tstart

    for length, op in _iter_cigar_ops(cigar):
        if op in 'M=X':
            overlap_start = max(tpos, interval_start)
            overlap_end = min(tpos + length, interval_end)
            if overlap_start < overlap_end:
                for target_pos in range(overlap_start, overlap_end):
                    columns[target_pos - interval_start].append(
                        query_seq[qpos + (target_pos - tpos)]
                    )
            qpos += length
            tpos += length
        elif op in 'IS':
            qpos += length
        elif op in 'DN':
            tpos += length


def _call_barcode_from_columns(columns, draft_barcode, barcode_template,
                               max_mismatches=5, max_indels=3):
    relaxed_mismatches = max(max_mismatches * 2, 10)
    relaxed_indels = max(max_indels * 2, 8)

    top_counts = []
    coverages = []
    barcode = []

    for idx, bases in enumerate(columns):
        counts = Counter(base for base in bases if base != 'N')
        if counts:
            top_base, top_count = counts.most_common(1)[0]
            barcode.append(top_base)
            top_counts.append(top_count)
            coverages.append(sum(counts.values()))
        else:
            barcode.append(draft_barcode[idx])
            top_counts.append(0)
            coverages.append(0)

    raw_barcode = ''.join(barcode)
    corrected_barcode, snap_stats = snap_barcode_to_template(
        raw_barcode,
        barcode_template,
        max_mismatches=relaxed_mismatches,
        max_indels=relaxed_indels,
    )
    if corrected_barcode is None:
        corrected_barcode, norm_stats = _normalize_barcode_to_template(
            raw_barcode,
            barcode_template,
        )
        snap_stats = {
            **norm_stats,
            'rejected': False,
            'used_normalization': True,
            'total_edits': norm_stats.get('insertions', 0) + norm_stats.get('deletions', 0),
        }
    else:
        snap_stats = {
            **snap_stats,
            'used_normalization': False,
        }

    return corrected_barcode, {
        'raw_barcode': raw_barcode,
        'top_counts': top_counts,
        'coverages': coverages,
        'support_total': sum(top_counts),
        'head_ratios': [
            top / max(1, cov)
            for top, cov in zip(top_counts[:5], coverages[:5])
        ],
        'tail_ratios': [
            top / max(1, cov)
            for top, cov in zip(top_counts[-5:], coverages[-5:])
        ],
        'snap_stats': snap_stats,
    }


def _project_interval_to_read_barcode(alignment_rows, reads, interval_start, interval_end,
                                      barcode_template, max_mismatches=5,
                                      max_indels=3, extra_right=1):
    projected_barcodes = []
    used_ids = set()

    for row in alignment_rows:
        if row['tend'] <= interval_start or row['tstart'] >= interval_end:
            continue

        qseq = reads.get(row['qname'])
        if not qseq:
            continue

        qpos = row['qstart']
        tpos = row['tstart']
        read_start = None
        read_end = None

        for length, op in _iter_cigar_ops(row['cigar']):
            if op in 'M=X':
                if read_start is None and tpos <= interval_start < tpos + length:
                    read_start = qpos + (interval_start - tpos)
                if read_end is None and tpos <= interval_end - 1 < tpos + length:
                    read_end = qpos + (interval_end - 1 - tpos) + 1
                qpos += length
                tpos += length
            elif op in 'IS':
                qpos += length
            elif op in 'DN':
                tpos += length

        if read_start is None or read_end is None:
            continue

        raw_barcode = qseq[read_start:min(len(qseq), read_end + extra_right)]
        if not raw_barcode:
            continue

        corrected_barcode, snap_stats = snap_barcode_to_template(
            raw_barcode,
            barcode_template,
            max_mismatches=max(max_mismatches * 2, 10),
            max_indels=max(max_indels * 2, 8),
        )
        if corrected_barcode is None:
            corrected_barcode, norm_stats = _normalize_barcode_to_template(
                raw_barcode,
                barcode_template,
            )
            snap_stats = {
                **norm_stats,
                'used_normalization': True,
            }

        projected_barcodes.append(corrected_barcode)
        used_ids.add(row['qname'])

    if not projected_barcodes:
        return None

    consensus_barcode = _barcode_consensus_signature(projected_barcodes)
    corrected_consensus, snap_stats = snap_barcode_to_template(
        consensus_barcode,
        barcode_template,
        max_mismatches=max(max_mismatches * 2, 10),
        max_indels=max(max_indels * 2, 8),
    )
    if corrected_consensus is not None:
        consensus_barcode = corrected_consensus
    else:
        snap_stats = {'used_normalization': True}

    return {
        'barcode': consensus_barcode,
        'interval_start': interval_start,
        'interval_end': interval_end,
        'n_used': len(used_ids),
        'support_total': len(projected_barcodes),
        'tail_ratios': [],
        'snap_stats': snap_stats,
    }


def _build_fullread_barcode_consensus(input_fq, consensus_seq, barcode_template,
                                      max_mismatches=5, max_indels=3,
                                      tail_instability_threshold=0.65,
                                      head_instability_threshold=0.8):
    stats = {
        'n_total': 0,
        'n_used': 0,
        'rejected': False,
    }

    if not input_fq or not path.exists(input_fq):
        stats['rejected'] = True
        stats['reason'] = 'missing_cluster_reads'
        return None, stats

    consensus_upper = consensus_seq.upper()
    interval_start, interval_end, _, window_score, _ = _find_barcode_region_fast(
        consensus_upper,
        barcode_template.upper(),
        allow_reverse_complement=False,
    )
    if interval_start < 0:
        stats['rejected'] = True
        stats['reason'] = 'barcode_interval_not_found'
        return None, stats

    reads = {
        record.id: str(record.seq).upper()
        for record in SeqIO.parse(input_fq, "fastq")
    }
    stats['n_total'] = len(reads)
    if not reads:
        stats['rejected'] = True
        stats['reason'] = 'no_cluster_reads'
        return None, stats

    with tempfile.TemporaryDirectory(prefix='dnabarmap_barcode_') as temp_dir:
        consensus_fasta = path.join(temp_dir, "consensus.fasta")
        SeqIO.write(
            [SeqRecord(Seq(consensus_upper), id="consensus", description="")],
            consensus_fasta,
            "fasta",
        )

        result = subprocess.run(
            ["minimap2", "-x", "map-ont", "-c", consensus_fasta, input_fq],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT,
        )
        if result.returncode != 0:
            stats['rejected'] = True
            stats['reason'] = 'barcode_pileup_map_failed'
            stats['stderr_tail'] = result.stderr[-500:]
            return None, stats

    alignment_rows = []
    for line in result.stdout.splitlines():
        if not line or line.startswith('['):
            continue
        cols = line.split('\t')
        cigar = next((c[5:] for c in cols[12:] if c.startswith('cg:Z:')), None)
        if cigar is None or cols[0] not in reads:
            continue
        alignment_rows.append({
            'qname': cols[0],
            'qstart': int(cols[2]),
            'tstart': int(cols[7]),
            'tend': int(cols[8]),
            'cigar': cigar,
        })

    def call_interval(iv_start, iv_end):
        if iv_end <= iv_start:
            return None

        columns = [[] for _ in range(iv_end - iv_start)]
        used_ids = set()
        for row in alignment_rows:
            if row['tend'] <= iv_start or row['tstart'] >= iv_end:
                continue
            _pileup_barcode_columns(
                reads[row['qname']],
                row['qstart'],
                row['tstart'],
                row['cigar'],
                iv_start,
                iv_end,
                columns,
            )
            used_ids.add(row['qname'])

        if not used_ids:
            return None

        called_barcode, call_stats = _call_barcode_from_columns(
            columns,
            consensus_upper[iv_start:iv_end],
            barcode_template=barcode_template,
            max_mismatches=max_mismatches,
            max_indels=max_indels,
        )
        return {
            'barcode': called_barcode,
            'interval_start': iv_start,
            'interval_end': iv_end,
            'n_used': len(used_ids),
            **call_stats,
        }

    best_call = call_interval(interval_start, interval_end)
    if best_call is None:
        stats['rejected'] = True
        stats['reason'] = 'no_overlapping_barcode_alignments'
        return None, stats

    stats['left_shift_retry'] = 0
    while best_call['interval_start'] > 0:
        head_ratios = best_call.get('head_ratios', [])
        first_ratio = head_ratios[0] if head_ratios else 1.0
        if first_ratio >= head_instability_threshold:
            break

        shifted_call = call_interval(
            best_call['interval_start'] - 1,
            best_call['interval_end'],
        )
        if shifted_call is None:
            break

        shifted_head_ratios = shifted_call.get('head_ratios', [])
        shifted_first_ratio = shifted_head_ratios[0] if shifted_head_ratios else 0.0
        if shifted_first_ratio <= first_ratio:
            break

        best_call = shifted_call
        stats['left_shift_retry'] += 1
        if stats['left_shift_retry'] >= 2:
            break

    tail_ratios = best_call.get('tail_ratios', [])
    penultimate_ratio = tail_ratios[-2] if len(tail_ratios) >= 2 else 1.0
    stats['tail_penultimate_ratio'] = round(penultimate_ratio, 3)
    head_ratios = best_call.get('head_ratios', [])
    first_ratio = head_ratios[0] if head_ratios else 1.0
    stats['head_first_ratio'] = round(first_ratio, 3)

    if (penultimate_ratio < tail_instability_threshold and
            best_call['interval_end'] < len(consensus_upper)):
        shifted_call = call_interval(
            best_call['interval_start'],
            best_call['interval_end'] + 1,
        )
        if shifted_call is not None:
            best_call = shifted_call
            stats['right_shift_retry'] = True
    else:
        stats['right_shift_retry'] = False

    projected_call = _project_interval_to_read_barcode(
        alignment_rows,
        reads,
        best_call['interval_start'],
        best_call['interval_end'],
        barcode_template,
        max_mismatches=max_mismatches,
        max_indels=max_indels,
        extra_right=1,
    )
    if projected_call is not None:
        direct_barcode = best_call['barcode']
        projected_barcode = projected_call['barcode']
        n_barcode_diffs = sum(
            base_a != base_b
            for base_a, base_b in zip(direct_barcode, projected_barcode)
        )
        min_tail_ratio = min(best_call.get('tail_ratios') or [1.0])
        if n_barcode_diffs >= 3 and min_tail_ratio < 0.8:
            best_call = projected_call
            stats['projected_local_retry'] = True
        else:
            stats['projected_local_retry'] = False
    else:
        stats['projected_local_retry'] = False

    stats['n_used'] = best_call.get('n_used', 0)
    stats['support_total'] = best_call.get('support_total', 0)
    stats['window_score'] = round(window_score, 3)
    stats['interval'] = (best_call['interval_start'], best_call['interval_end'])
    stats['head_ratios'] = [round(r, 3) for r in best_call.get('head_ratios', [])]
    stats['tail_ratios'] = [round(r, 3) for r in best_call.get('tail_ratios', [])]
    stats['snap_used_normalization'] = best_call.get('snap_stats', {}).get('used_normalization', False)
    stats['total_edits'] = best_call.get('snap_stats', {}).get('total_edits', 0)

    return best_call['barcode'], stats


def _inject_barcode_into_consensus(consensus_seq, barcode_seq, barcode_template,
                                   search_margin=6):
    if not consensus_seq or not barcode_seq:
        return None, {'rejected': True, 'reason': 'missing_sequence'}

    start, end, orient, window_score, effective_template = _find_barcode_region(
        consensus_seq.upper(),
        barcode_template.upper(),
        search_margin=search_margin,
        allow_reverse_complement=False,
    )

    if start < 0:
        return None, {'rejected': True, 'reason': 'barcode_region_not_found'}

    corrected_full = consensus_seq.upper()[:start] + barcode_seq.upper() + consensus_seq.upper()[end:]
    return corrected_full, {
        'rejected': False,
        'barcode_start': start,
        'barcode_end': end,
        'window_score': window_score,
        'orientation': orient,
        'template': effective_template,
    }


def _inject_barcode_at_interval(consensus_seq, barcode_seq, interval_start, interval_end):
    if not consensus_seq or not barcode_seq:
        return None, {'rejected': True, 'reason': 'missing_sequence'}

    if interval_start < 0 or interval_end <= interval_start or interval_end > len(consensus_seq):
        return None, {'rejected': True, 'reason': 'invalid_interval'}

    corrected_full = (
        consensus_seq.upper()[:interval_start]
        + barcode_seq.upper()
        + consensus_seq.upper()[interval_end:]
    )
    return corrected_full, {
        'rejected': False,
        'barcode_start': interval_start,
        'barcode_end': interval_end,
    }


def _find_best_motif_window(seq, motif, start=0, end=None):
    motif = motif.upper()
    seq_upper = seq.upper()
    motif_len = len(motif)

    if motif_len == 0 or len(seq_upper) < motif_len:
        return -1, motif_len

    stop = len(seq_upper) if end is None else min(len(seq_upper), end)
    best_pos = -1
    best_mismatches = motif_len + 1

    for pos in range(max(0, start), max(0, stop - motif_len + 1)):
        window = seq_upper[pos:pos + motif_len]
        mismatches = sum(a != b for a, b in zip(window, motif))
        if mismatches < best_mismatches:
            best_pos = pos
            best_mismatches = mismatches
            if mismatches == 0:
                break

    return best_pos, best_mismatches


def _find_motif_candidates(seq, motif, max_mismatches, start=0, end=None):
    motif = motif.upper()
    seq_upper = seq.upper()
    motif_len = len(motif)

    if motif_len == 0 or len(seq_upper) < motif_len:
        return []

    stop = len(seq_upper) if end is None else min(len(seq_upper), end)
    candidates = []
    for pos in range(max(0, start), max(0, stop - motif_len + 1)):
        window = seq_upper[pos:pos + motif_len]
        mismatches = sum(a != b for a, b in zip(window, motif))
        if mismatches <= max_mismatches:
            candidates.append((pos, mismatches))

    return candidates


def _find_exact_subseq_positions(seq, subseq):
    seq_upper = seq.upper()
    subseq_upper = subseq.upper()
    positions = []
    start = 0
    while subseq_upper:
        pos = seq_upper.find(subseq_upper, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _find_barcode_anchored_flank_pair(seq, left_flank, right_flank, barcode_seq,
                                      max_mismatches=2,
                                      preferred_coding_len=None):
    seq_upper = seq.upper()
    left_flank = left_flank.upper()
    right_flank = right_flank.upper()
    barcode_seq = barcode_seq.upper()

    barcode_positions = _find_exact_subseq_positions(seq_upper, barcode_seq)
    if not barcode_positions:
        return None

    relaxed_flank_mismatches = max(max_mismatches, 4)
    min_coding_len = max(50, int(0.2 * len(seq_upper)))
    candidates = []

    for barcode_start in barcode_positions:
        barcode_end = barcode_start + len(barcode_seq)

        left_candidates = _find_motif_candidates(
            seq_upper,
            left_flank,
            max_mismatches=relaxed_flank_mismatches,
            start=barcode_end,
        )
        for left_pos, left_mm in left_candidates:
            if left_pos <= barcode_end:
                continue
            right_candidates = _find_motif_candidates(
                seq_upper,
                right_flank,
                max_mismatches=relaxed_flank_mismatches,
                start=left_pos + len(left_flank) + min_coding_len,
            )
            for right_pos, right_mm in right_candidates:
                if right_pos <= left_pos + len(left_flank):
                    continue
                coding_len = right_pos - (left_pos + len(left_flank))
                tail_len = len(seq_upper) - (right_pos + len(right_flank))
                gap = left_pos - barcode_end
                length_penalty = 0
                if preferred_coding_len is not None:
                    length_penalty = 4 * abs(coding_len - preferred_coding_len)
                score = (40 * (left_mm + right_mm)) + length_penalty + tail_len + gap
                candidates.append({
                    'orientation': 'barcode_before_coding',
                    'left_pos': left_pos,
                    'right_pos': right_pos,
                    'left_mismatches': left_mm,
                    'right_mismatches': right_mm,
                    'coding_len': coding_len,
                    'barcode_start': barcode_start,
                    'barcode_end': barcode_end,
                    'score': score,
                })

        right_candidates = _find_motif_candidates(
            seq_upper,
            right_flank,
            max_mismatches=relaxed_flank_mismatches,
            end=barcode_start - min_coding_len,
        )
        for right_pos, right_mm in right_candidates:
            left_candidates = _find_motif_candidates(
                seq_upper,
                left_flank,
                max_mismatches=relaxed_flank_mismatches,
                end=right_pos - len(left_flank),
            )
            for left_pos, left_mm in left_candidates:
                if right_pos <= left_pos + len(left_flank):
                    continue
                coding_len = right_pos - (left_pos + len(left_flank))
                head_len = left_pos
                gap = barcode_start - (right_pos + len(right_flank))
                length_penalty = 0
                if preferred_coding_len is not None:
                    length_penalty = 4 * abs(coding_len - preferred_coding_len)
                score = (40 * (left_mm + right_mm)) + length_penalty + head_len + gap
                candidates.append({
                    'orientation': 'barcode_after_coding',
                    'left_pos': left_pos,
                    'right_pos': right_pos,
                    'left_mismatches': left_mm,
                    'right_mismatches': right_mm,
                    'coding_len': coding_len,
                    'barcode_start': barcode_start,
                    'barcode_end': barcode_end,
                    'score': score,
                })

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda cand: (
            cand['score'],
            abs(cand['coding_len'] - preferred_coding_len)
            if preferred_coding_len is not None else 0,
            cand['left_mismatches'] + cand['right_mismatches'],
            -cand['coding_len'],
        ),
    )


def _estimate_coding_length_from_reads(input_fq, left_flank, right_flank,
                                       max_mismatches=3):
    if not input_fq or not path.exists(input_fq):
        return None

    left_flank = (left_flank or '').upper()
    right_flank = (right_flank or '').upper()
    if not left_flank or not right_flank:
        return None

    best_lengths = []
    min_coding_len = max(150, len(left_flank) + len(right_flank))

    for record in SeqIO.parse(input_fq, "fastq"):
        seq = str(record.seq).upper()
        left_candidates = _find_motif_candidates(seq, left_flank, max_mismatches)
        right_candidates = _find_motif_candidates(seq, right_flank, max_mismatches)
        if not left_candidates or not right_candidates:
            continue

        best_pair = None
        for left_pos, left_mm in left_candidates:
            for right_pos, right_mm in right_candidates:
                coding_len = right_pos - (left_pos + len(left_flank))
                if coding_len < min_coding_len:
                    continue
                tail_len = len(seq) - (right_pos + len(right_flank))
                pair_score = (
                    left_mm + right_mm,
                    abs(tail_len),
                    -coding_len,
                )
                if best_pair is None or pair_score < best_pair[0]:
                    best_pair = (pair_score, coding_len)

        if best_pair is not None:
            best_lengths.append(best_pair[1])

    if not best_lengths:
        return None

    median_len = int(round(statistics.median(best_lengths)))
    refined = [
        length for length in best_lengths
        if abs(length - median_len) <= 40
    ]
    if refined:
        median_len = int(round(statistics.median(refined)))

    return median_len


def snap_coding_flanks_in_consensus(consensus_seq, left_coding_flank,
                                    right_coding_flank, max_mismatches=2,
                                    barcode_seq=None,
                                    preferred_coding_len=None):
    if not left_coding_flank or not right_coding_flank:
        return consensus_seq, {'rejected': True, 'reason': 'missing_flanks'}

    seq_upper = consensus_seq.upper()
    left_flank = left_coding_flank.upper()
    right_flank = right_coding_flank.upper()

    if barcode_seq:
        pair = _find_barcode_anchored_flank_pair(
            seq_upper,
            left_flank,
            right_flank,
            barcode_seq=barcode_seq,
            max_mismatches=max_mismatches,
            preferred_coding_len=preferred_coding_len,
        )
        if pair is not None:
            corrected = list(seq_upper)
            left_pos = pair['left_pos']
            right_pos = pair['right_pos']
            corrected[left_pos:left_pos + len(left_flank)] = left_flank
            corrected[right_pos:right_pos + len(right_flank)] = right_flank
            return ''.join(corrected), {
                'rejected': False,
                'left_pos': left_pos,
                'right_pos': right_pos,
                'left_mismatches': pair['left_mismatches'],
                'right_mismatches': pair['right_mismatches'],
                'total_edits': pair['left_mismatches'] + pair['right_mismatches'],
                'coding_len': pair['coding_len'],
                'orientation': pair['orientation'],
                'anchored_to_barcode': True,
                'preferred_coding_len': preferred_coding_len,
            }

    left_pos, left_mismatches = _find_best_motif_window(seq_upper, left_flank)
    if left_pos < 0 or left_mismatches > max_mismatches:
        return consensus_seq, {
            'rejected': True,
            'reason': f'left_flank_not_found ({left_mismatches})',
        }

    right_pos, right_mismatches = _find_best_motif_window(
        seq_upper,
        right_flank,
        start=left_pos + len(left_flank),
    )
    if right_pos < 0 or right_mismatches > max_mismatches:
        return consensus_seq, {
            'rejected': True,
            'reason': f'right_flank_not_found ({right_mismatches})',
        }

    corrected = list(seq_upper)
    corrected[left_pos:left_pos + len(left_flank)] = left_flank
    corrected[right_pos:right_pos + len(right_flank)] = right_flank

    return ''.join(corrected), {
        'rejected': False,
        'left_pos': left_pos,
        'right_pos': right_pos,
        'left_mismatches': left_mismatches,
        'right_mismatches': right_mismatches,
        'total_edits': left_mismatches + right_mismatches,
    }


def filter_cluster_reads_by_barcode(input_fq, barcode_template,
                                    max_mismatches=5, max_indels=3,
                                    min_window_score=0.7,
                                    min_group_size=3):
    """
    Refine a cluster using only the barcode region.

    The upstream clustering can still mix multiple constructs because their
    full-length reads share a long, highly similar coding region. This step
    picks the densest neighborhood in barcode space instead of relying on a
    whole-read connected component.
    """
    records = list(SeqIO.parse(input_fq, "fastq"))
    n_total = len(records)

    stats = {
        'n_total': n_total,
        'n_barcode_candidates': 0,
        'n_kept': 0,
        'filtered': False,
        'rejected': False,
    }

    if n_total <= min_group_size:
        kept_ids = {record.id for record in records}
        stats['n_barcode_candidates'] = n_total
        stats['n_kept'] = n_total
        return input_fq, kept_ids, stats

    relaxed_window = min(min_window_score, 0.55)
    relaxed_mismatches = max(max_mismatches * 2, 10)
    relaxed_indels = max(max_indels * 2, 6)
    search_margin = max(max_indels * 2, 6)

    candidates = []
    for record in records:
        signature = _extract_read_barcode_signature(
            str(record.seq),
            barcode_template,
            min_window_score=relaxed_window,
            search_margin=search_margin,
        )
        if signature is None:
            continue

        corrected_bc, snap_stats = snap_barcode_to_template(
            signature['raw_barcode'],
            signature['template'],
            max_mismatches=relaxed_mismatches,
            max_indels=relaxed_indels,
        )

        cluster_barcode = corrected_bc if corrected_bc is not None else signature['normalized_barcode']
        candidates.append({
            'record_id': record.id,
            'cluster_barcode': cluster_barcode,
            'window_score': signature['window_score'],
            'template_alignment_score': signature['alignment_score'],
            'snap_stats': snap_stats,
        })

    stats['n_barcode_candidates'] = len(candidates)
    if len(candidates) < min_group_size:
        stats['rejected'] = True
        stats['reason'] = 'too_few_barcode_candidates'
        return None, set(), stats

    barcode_len = len(barcode_template)
    min_comparable = max(45, int(barcode_len * 0.7))
    max_distance = max(8, int(barcode_len * 0.16))

    best_indices = []
    best_tiebreak = (-1, -1.0, -1.0)

    for i, seed in enumerate(candidates):
        neighbors = [i]
        similarity_sum = 0.0

        for j, other in enumerate(candidates):
            if i == j:
                continue

            distance, comparable = _barcode_distance(
                seed['cluster_barcode'],
                other['cluster_barcode'],
                min_comparable=min_comparable,
            )
            if distance is None or distance > max_distance:
                continue

            neighbors.append(j)
            similarity_sum += comparable - distance

        tiebreak = (
            len(neighbors),
            similarity_sum,
            seed['window_score'],
        )
        if tiebreak > best_tiebreak:
            best_tiebreak = tiebreak
            best_indices = neighbors

    if len(best_indices) < min_group_size:
        stats['rejected'] = True
        stats['reason'] = (
            f'barcode_group_too_small ({len(best_indices)} < {min_group_size})'
        )
        return None, set(), stats

    refined_signature = _barcode_consensus_signature(
        [candidates[i]['cluster_barcode'] for i in best_indices]
    )

    refined_indices = []
    if refined_signature is not None:
        for i, candidate in enumerate(candidates):
            distance, comparable = _barcode_distance(
                candidate['cluster_barcode'],
                refined_signature,
                min_comparable=min_comparable,
            )
            if distance is not None and distance <= max_distance:
                refined_indices.append(i)

    if len(refined_indices) >= len(best_indices):
        best_indices = refined_indices

    kept_ids = {candidates[i]['record_id'] for i in best_indices}
    stats['n_kept'] = len(kept_ids)
    stats['fraction_kept'] = len(kept_ids) / n_total
    stats['seed_group_size'] = len(best_indices)
    stats['min_comparable'] = min_comparable
    stats['max_barcode_distance'] = max_distance

    if len(kept_ids) == n_total:
        return input_fq, kept_ids, stats

    filtered_fq = input_fq.replace('.fastq', '_barcode_filtered.fastq')
    with open(filtered_fq, 'w') as out_handle:
        for record in records:
            if record.id in kept_ids:
                SeqIO.write(record, out_handle, "fastq")

    stats['filtered'] = True
    return filtered_fq, kept_ids, stats


def _choose_draft_record(records, draft_support=None):
    if not records:
        return None

    if draft_support is None:
        draft_support = {}

    lengths = sorted(len(record.seq) for record in records)
    median_len = lengths[len(lengths) // 2]

    return max(
        records,
        key=lambda record: (
            draft_support.get(record.id, 0.0),
            -abs(len(record.seq) - median_len),
            len(record.seq),
        ),
    )


def _select_polishing_records(records, draft_support=None, max_reads=24):
    """Keep a representative high-support subset for polishing."""
    if max_reads is None:
        return records

    if len(records) <= max_reads:
        return records

    if draft_support is None:
        draft_support = {}

    lengths = sorted(len(record.seq) for record in records)
    median_len = lengths[len(lengths) // 2]

    ranked = sorted(
        records,
        key=lambda record: (
            draft_support.get(record.id, 0.0),
            -abs(len(record.seq) - median_len),
            len(record.seq),
            record.id,
        ),
        reverse=True,
    )
    return ranked[:max_reads]
# ═════════════════════════════════════════════════════════════════
# MEDAKA DETECTION
# ═════════════════════════════════════════════════════════════════

def _check_medaka_available():
    """Check once at startup; returns True/False."""
    return shutil.which("medaka_consensus") is not None


# ═════════════════════════════════════════════════════════════════
# MEDAKA POLISHING
# ═════════════════════════════════════════════════════════════════

def _resolve_medaka_model(model):
    """
    Pass the model name through to medaka_consensus as-is.

    The medaka_consensus bash script handles model resolution internally,
    including auto-detection from input BAM files. Do NOT append
    ':consensus' — that suffix is only for `medaka inference` direct calls.
    """
    return model


def run_medaka(input_reads, draft_fasta, output_dir, model, threads):
    """
    Run Medaka polishing via `medaka_consensus`.

    `medaka_consensus` is a bash script that internally pipes between
    minimap2, medaka inference, and medaka sequence. We redirect output
    to a log file and verify the result is actually polished.
    """
    makedirs(output_dir, exist_ok=True)

    cmd = ["medaka_consensus",
           "-i", input_reads,
           "-d", draft_fasta,
           "-o", output_dir,
           "-t", str(threads),
           "-m", model,
           "-f"]  # force overwrite

    if not shutil.which("medaka_consensus"):
        raise RuntimeError(
            "medaka_consensus not found on PATH. "
            "Install with: pip install medaka"
        )

    log_file = path.join(output_dir, "medaka.log")

    with open(log_file, "w") as log_handle:
        result = subprocess.run(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT,
        )

    # Read log to check for internal failures
    log_content = ""
    if path.exists(log_file):
        with open(log_file) as f:
            log_content = f.read()

    # medaka_consensus bash script doesn't always propagate errors
    # via exit code, so check the log for failure signatures
    failure_signatures = [
        "Failed to run alignment",
        "Failed to run inference",
        "Failed to run sequence",
        "Error:",
        "Traceback (most recent call last)",
        "FileNotFoundError",
        "No such file or directory",
        "is not a recognised basecaller model",
    ]
    for sig in failure_signatures:
        if sig in log_content:
            raise RuntimeError(
                f"medaka_consensus failed internally ({sig})\n"
                f"CMD: {' '.join(cmd)}\n"
                f"LOG (tail):\n{log_content[-2000:]}"
            )

    if result.returncode != 0:
        raise RuntimeError(
            f"medaka_consensus failed (exit {result.returncode})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"LOG (tail):\n{log_content[-2000:]}"
        )

    # medaka_consensus outputs to consensus.fasta in the output dir
    consensus_out = path.join(output_dir, "consensus.fasta")

    if not path.exists(consensus_out):
        fastas = glob(path.join(output_dir, "*.fasta"))
        fastas = [f for f in fastas
                  if 'draft' not in path.basename(f).lower()
                  and path.basename(f) != path.basename(draft_fasta)]
        if fastas:
            consensus_out = fastas[0]
        else:
            raise RuntimeError(
                f"medaka_consensus produced no FASTA output in {output_dir}\n"
                f"LOG (tail):\n{log_content[-2000:]}"
            )

    return consensus_out


# ═════════════════════════════════════════════════════════════════
# RACON CONSENSUS (lightweight, no Medaka)
# ═════════════════════════════════════════════════════════════════

def run_racon(input_reads, draft_fasta, output_dir, threads, rounds=1):
    """
    Polish a draft sequence using Racon (partial-order alignment consensus).

    Racon handles indels properly — unlike simple pileup majority vote —
    and is the standard lightweight polisher for nanopore data. One round
    typically gets ~97-99% accuracy from raw nanopore reads.

    Multiple rounds can be run iteratively (each round uses the previous
    round's output as the new draft), though for 20-100 read clusters
    a single round is usually sufficient.
    """
    makedirs(output_dir, exist_ok=True)

    current_draft = draft_fasta

    for rnd in range(rounds):
        paf = path.join(output_dir, f"overlaps_r{rnd}.paf")
        out = path.join(output_dir, f"racon_r{rnd}.fasta")

        # Align reads to current draft
        with open(paf, "w") as paf_handle:
            result = subprocess.run(
                ["minimap2", "-x", "map-ont", "-t", str(threads),
                 current_draft, input_reads],
                stdout=paf_handle,
                stderr=subprocess.PIPE,
                timeout=TIMEOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Racon alignment failed (round {rnd})\n"
                f"STDERR:\n{result.stderr.decode()}"
            )

        # Run Racon
        with open(out, "w") as out_handle:
            result = subprocess.run(
                ["racon", "-t", str(threads),
                 input_reads, paf, current_draft],
                stdout=out_handle,
                stderr=subprocess.PIPE,
                timeout=TIMEOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Racon failed (round {rnd})\n"
                f"STDERR:\n{result.stderr.decode()}"
            )

        # Check that racon produced output
        if not path.exists(out) or path.getsize(out) == 0:
            raise RuntimeError(
                f"Racon produced empty output (round {rnd})"
            )

        # Clean up intermediate PAF
        if path.exists(paf):
            remove(paf)

        # Clean up previous round draft (but not the original)
        if rnd > 0 and current_draft != draft_fasta and path.exists(current_draft):
            remove(current_draft)

        current_draft = out

    return current_draft


# ═════════════════════════════════════════════════════════════════
# SINGLE-CLUSTER CONSENSUS (worker function)
# ═════════════════════════════════════════════════════════════════

def _process_one_cluster(cluster_fq, threads, medaka_model,
                         barcode_template, max_mismatches, max_indels,
                         min_window_score, min_identity, min_dominant_fraction,
                         min_overlap_fraction, min_aligned_bases,
                         timeout, left_coding_flank=None,
                         right_coding_flank=None, racon_rounds=2,
                         max_polish_reads=24,
                         keep_intermediates=False):
    """
    Process a single cluster FASTQ -> consensus FASTA.

    This is the worker function called by the process pool.
    Returns a result dict for aggregation by the parent.
    """
    global TIMEOUT
    TIMEOUT = timeout

    cluster_id = Path(cluster_fq).stem
    consensus_dir = cluster_fq.split('clusters')[0] + '/consensus/'
    sub_dir = consensus_dir + cluster_id[-2:]
    medaka_dir = f"{sub_dir}/{cluster_id}_medaka"

    makedirs(sub_dir, exist_ok=True)

    final_consensus = f"{sub_dir}/{cluster_id}_consensus.fasta"
    draft_fasta = f"{sub_dir}/{cluster_id}_draft.fasta"
    paf_file = f"{sub_dir}/{cluster_id}.paf"

    result = {'cluster_id': cluster_id, 'status': 'ok'}

    # ── Step 1: minimap2 self-map ────────────────────────────────
    try:
        with open(paf_file, "w") as paf_handle:
            subprocess.run(
                ["minimap2", "-x", "ava-ont", "-c", "-t", str(threads),
                 cluster_fq, cluster_fq],
                stdout=paf_handle,
                check=True,
                timeout=TIMEOUT
            )
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout_selfmap'
        if path.exists(paf_file):
            remove(paf_file)
        return result
    except Exception as e:
        result['status'] = f'error_selfmap: {e}'
        return result

    # ── Step 2: Purity filter ────────────────────────────────────
    filtered_fq, draft_support, filter_stats = filter_cluster_reads(
        cluster_fq, paf_file,
        min_identity=min_identity,
        min_dominant_fraction=min_dominant_fraction,
        min_overlap_fraction=min_overlap_fraction,
        min_aligned_bases=min_aligned_bases,
    )

    if path.exists(paf_file):
        remove(paf_file)

    # Always propagate read counts
    result['n_total'] = filter_stats.get('n_total', 0)
    result['n_kept'] = filter_stats.get('n_kept', 0)
    result['n_filtered_out'] = result['n_total'] - result['n_kept']
    result['n_overlap_edges'] = filter_stats.get('n_overlap_edges', 0)

    barcode_filtered_fq = None
    barcode_keep_ids = set()
    barcode_filter_stats = None

    if barcode_template:
        barcode_filter_input = filtered_fq if filtered_fq is not None else cluster_fq
        barcode_filtered_fq, barcode_keep_ids, barcode_filter_stats = filter_cluster_reads_by_barcode(
            barcode_filter_input,
            barcode_template=barcode_template,
            max_mismatches=max_mismatches,
            max_indels=max_indels,
            min_window_score=min_window_score,
        )
        if barcode_filter_stats is not None:
            result['n_barcode_candidates'] = barcode_filter_stats.get('n_barcode_candidates', 0)

    if filtered_fq is None and barcode_filtered_fq is None:
        result['status'] = 'purity_rejected'
        result['filter_stats'] = filter_stats
        if barcode_filter_stats is not None:
            result['barcode_filter_stats'] = barcode_filter_stats
        return result

    use_barcode_filter = False
    if barcode_filtered_fq is not None:
        barcode_n_kept = barcode_filter_stats.get('n_kept', result['n_total'])
        fullread_n_kept = filter_stats.get('n_kept', result['n_total'])
        use_barcode_filter = (
            filtered_fq is None or
            barcode_n_kept >= max(3, int(fullread_n_kept * 0.6))
        )

    if use_barcode_filter:
        if filtered_fq and filtered_fq != cluster_fq and filtered_fq != barcode_filtered_fq:
            _cleanup_filtered(filtered_fq, cluster_fq)
        filtered_fq = barcode_filtered_fq
        result['n_kept'] = barcode_filter_stats.get('n_kept', result['n_kept'])
        result['n_filtered_out'] = result['n_total'] - result['n_kept']
        result['barcode_group_size'] = barcode_filter_stats.get('n_kept', 0)
        if barcode_filter_stats.get('filtered'):
            result['barcode_filter'] = (
                f"{barcode_filter_stats.get('n_kept', 0)}/"
                f"{barcode_filter_stats.get('n_total', 0)} kept"
            )
    elif barcode_filtered_fq and barcode_filtered_fq != cluster_fq:
        _cleanup_filtered(barcode_filtered_fq, cluster_fq)

    # ── Step 3: Pick best draft read ─────────────────────────────
    filtered_records = list(SeqIO.parse(filtered_fq, "fastq"))
    if barcode_keep_ids:
        draft_support = {rid: score for rid, score in draft_support.items()
                         if rid in barcode_keep_ids}

    if not filtered_records:
        result['status'] = 'no_draft'
        _cleanup_filtered(filtered_fq, cluster_fq)
        return result

    if not draft_support and barcode_keep_ids:
        draft_support = {record.id: 0.0 for record in filtered_records}

    draft_record = _choose_draft_record(filtered_records, draft_support=draft_support)
    if draft_record is None:
        result['status'] = 'no_draft'
        _cleanup_filtered(filtered_fq, cluster_fq)
        return result

    SeqIO.write(draft_record, draft_fasta, "fasta")

    polish_records = _select_polishing_records(
        filtered_records,
        draft_support=draft_support,
        max_reads=max_polish_reads,
    )
    polish_fq = filtered_fq
    if len(polish_records) < len(filtered_records):
        polish_fq = filtered_fq.replace('.fastq', '_polish.fastq')
        with open(polish_fq, 'w') as out_handle:
            SeqIO.write(polish_records, out_handle, "fastq")
        result['n_polish_reads'] = len(polish_records)
    else:
        result['n_polish_reads'] = len(filtered_records)

    # ── Step 4: Polishing ────────────────────────────────────────
    pileup_dir = f"{sub_dir}/{cluster_id}_pileup"
    racon_dir = f"{sub_dir}/{cluster_id}_racon"

    if medaka_model and medaka_model.lower() != "none":
        # Full Medaka polishing
        try:
            polished = run_medaka(
                input_reads=polish_fq,
                draft_fasta=draft_fasta,
                output_dir=medaka_dir,
                model=medaka_model,
                threads=threads,
            )
            result['consensus_method'] = 'medaka'
        except subprocess.TimeoutExpired:
            # Medaka timed out — fall back to Racon
            result['medaka_warning'] = 'timeout, falling back to racon'
            try:
                polished = run_racon(
                    input_reads=polish_fq,
                    draft_fasta=draft_fasta,
                    output_dir=racon_dir,
                    threads=threads,
                    rounds=racon_rounds,
                )
                result['consensus_method'] = 'racon (medaka fallback)'
            except Exception:
                polished = draft_fasta
                result['consensus_method'] = 'draft (all polishing failed)'
        except Exception as e:
            # Medaka errored — fall back to Racon
            result['medaka_warning'] = f'{e}, falling back to racon'
            try:
                polished = run_racon(
                    input_reads=polish_fq,
                    draft_fasta=draft_fasta,
                    output_dir=racon_dir,
                    threads=threads,
                    rounds=racon_rounds,
                )
                result['consensus_method'] = 'racon (medaka fallback)'
            except Exception:
                polished = draft_fasta
                result['consensus_method'] = 'draft (all polishing failed)'
    else:
        # Lightweight Racon consensus (partial-order alignment)
        try:
            polished = run_racon(
                input_reads=polish_fq,
                draft_fasta=draft_fasta,
                output_dir=racon_dir,
                threads=threads,
                rounds=racon_rounds,
            )
            result['consensus_method'] = 'racon'
        except subprocess.TimeoutExpired:
            polished = draft_fasta
            result['racon_warning'] = 'timeout, fell back to draft'
            result['consensus_method'] = 'draft (racon timeout)'
        except Exception as e:
            polished = draft_fasta
            result['racon_warning'] = f'{e}, fell back to draft'
            result['consensus_method'] = 'draft (racon failed)'

    records = list(SeqIO.parse(polished, "fasta"))
    if not records:
        result['status'] = 'no_consensus'
        _cleanup_filtered(filtered_fq, cluster_fq)
        return result

    consensus_seq = str(records[0].seq)

    # ── Diagnostic: check if polishing actually changed the draft ─
    if path.exists(draft_fasta):
        draft_records = list(SeqIO.parse(draft_fasta, "fasta"))
        if draft_records:
            draft_seq = str(draft_records[0].seq)
            if consensus_seq == draft_seq:
                result['polish_warning'] = 'consensus identical to draft (polishing had no effect)'
            else:
                # Count differences
                if len(consensus_seq) == len(draft_seq):
                    n_diff = sum(1 for a, b in zip(consensus_seq, draft_seq) if a != b)
                    result['polish_changes'] = n_diff
                else:
                    result['polish_changes'] = f'len_change:{len(draft_seq)}->{len(consensus_seq)}'

    # ── Step 5: Barcode diagnostics (no barcode reinjection) ─────
    authoritative_barcode = None
    if barcode_template:
        authoritative_barcode, authoritative_barcode_stats = _build_fullread_barcode_consensus(
            cluster_fq,
            consensus_seq=consensus_seq,
            barcode_template=barcode_template,
            max_mismatches=max_mismatches,
            max_indels=max_indels,
        )
        if authoritative_barcode_stats is not None:
            result['n_barcode_consensus_reads'] = authoritative_barcode_stats.get('n_used', 0)
            if authoritative_barcode_stats.get('right_shift_retry'):
                result['barcode_interval_retry'] = 'end+1'
            elif authoritative_barcode_stats.get('left_shift_retry'):
                result['barcode_interval_retry'] = f"start-{authoritative_barcode_stats['left_shift_retry']}"
            elif authoritative_barcode_stats.get('projected_local_retry'):
                result['barcode_interval_retry'] = 'projected_local'
            tail_ratio = authoritative_barcode_stats.get('tail_penultimate_ratio')
            if tail_ratio is not None:
                result['barcode_tail_penult'] = tail_ratio
            head_ratio = authoritative_barcode_stats.get('head_first_ratio')
            if head_ratio is not None:
                result['barcode_head_first'] = head_ratio
            if authoritative_barcode_stats.get('total_edits', 0) > 0:
                result['barcode_consensus_edits'] = authoritative_barcode_stats['total_edits']

    if left_coding_flank and right_coding_flank:
        preferred_coding_len = _estimate_coding_length_from_reads(
            filtered_fq,
            left_coding_flank,
            right_coding_flank,
        )
        if preferred_coding_len is not None:
            result['preferred_coding_len'] = preferred_coding_len
        corrected_seq, flank_stats = snap_coding_flanks_in_consensus(
            consensus_seq,
            left_coding_flank=left_coding_flank,
            right_coding_flank=right_coding_flank,
            barcode_seq=authoritative_barcode,
            preferred_coding_len=preferred_coding_len,
        )
        if not flank_stats.get('rejected'):
            consensus_seq = corrected_seq
            if flank_stats.get('total_edits', 0) > 0:
                result['flank_edits'] = flank_stats['total_edits']
        else:
            result['flank_warning'] = flank_stats.get('reason', 'unknown')

    # ── Write final consensus ────────────────────────────────────
    record = SeqRecord(Seq(consensus_seq), id=cluster_id, description="")
    SeqIO.write(record, final_consensus, "fasta")

    # ── Cleanup ──────────────────────────────────────────────────
    if not keep_intermediates:
        if path.exists(draft_fasta):
            remove(draft_fasta)
        if path.isdir(medaka_dir):
            shutil.rmtree(medaka_dir, ignore_errors=True)
        if path.isdir(racon_dir):
            shutil.rmtree(racon_dir, ignore_errors=True)
        if path.isdir(pileup_dir):
            shutil.rmtree(pileup_dir, ignore_errors=True)
        _cleanup_filtered(filtered_fq, cluster_fq)
        _cleanup_filtered(polish_fq, filtered_fq)

    return result


def _cleanup_filtered(filtered_fq, original_fq):
    """Remove the filtered FASTQ if it's a separate file."""
    if filtered_fq and filtered_fq != original_fq and path.exists(filtered_fq):
        remove(filtered_fq)


# ═════════════════════════════════════════════════════════════════
# BATCH CONSENSUS (parallel)
# ═════════════════════════════════════════════════════════════════

def determine_consensus_parallel(output_dir, total_threads=8,
                                  medaka_model='none',
                                  barcode_template=None,
                                  left_coding_flank=None,
                                  right_coding_flank=None,
                                  max_mismatches=5, max_indels=3,
                                  min_window_score=0.7,
                                  min_identity=0.85,
                                  min_dominant_fraction=0.5,
                                  min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                                  min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES,
                                  n_workers=None,
                                  timeout=3600,
                                  racon_rounds=2,
                                  max_polish_reads=24,
                                  keep_intermediates=False,
                                  **kwargs):
    """
    Process all cluster FASTQs in parallel using ProcessPoolExecutor.

    Thread budget is split: total_threads / n_workers = threads per worker.
    """
    output_dir = output_dir.rstrip('/')

    full_seqs_dir = f"{output_dir}/clusters/full_seqs"
    cluster_files = sorted(glob(f"{full_seqs_dir}/**/*.fastq", recursive=True))

    if not cluster_files:
        cluster_files = sorted(glob(f"{full_seqs_dir}/*/*.fastq"))

    cluster_files = [
        f for f in cluster_files
        if not path.basename(f).endswith(('_filtered.fastq', '_barcode_filtered.fastq'))
    ]

    if not cluster_files:
        print(f"  WARNING: No cluster FASTQ files found in {full_seqs_dir}/")
        return

    print(f"  Found {len(cluster_files)} clusters to process")

    # ── Determine parallelism ────────────────────────────────────
    n_cpus = multiprocessing.cpu_count()

    if n_workers is None:
        n_workers = min(len(cluster_files), max(1, n_cpus // 2))

    threads_per_worker = max(1, total_threads // n_workers)

    print(f"  Parallelism: {n_workers} workers x {threads_per_worker} threads "
          f"= {n_workers * threads_per_worker} threads "
          f"({n_cpus} CPUs available)")

    # ── Validate medaka if needed ────────────────────────────────
    if medaka_model and medaka_model.lower() != "none":
        if not _check_medaka_available():
            raise RuntimeError(
                "Medaka not found. Install with: pip install medaka\n"
                "Or disable with --medaka_model none"
            )
        print(f"  Medaka model: {medaka_model}")

    # ── Launch workers ───────────────────────────────────────────
    succeeded = 0
    failed = 0
    purity_rejected = 0
    total_filtered_out = 0
    total_reads_kept = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for cluster_fq in cluster_files:
            fut = executor.submit(
                _process_one_cluster,
                cluster_fq=cluster_fq,
                threads=threads_per_worker,
                medaka_model=medaka_model,
                barcode_template=barcode_template,
                max_mismatches=max_mismatches,
                max_indels=max_indels,
                min_window_score=min_window_score,
                min_identity=min_identity,
                min_dominant_fraction=min_dominant_fraction,
                min_overlap_fraction=min_overlap_fraction,
                min_aligned_bases=min_aligned_bases,
                timeout=timeout,
                left_coding_flank=left_coding_flank,
                right_coding_flank=right_coding_flank,
                racon_rounds=racon_rounds,
                max_polish_reads=max_polish_reads,
                keep_intermediates=keep_intermediates,
            )
            futures[fut] = cluster_fq

        done_count = 0
        for fut in as_completed(futures):
            done_count += 1

            if done_count % 100 == 0 or done_count == len(cluster_files):
                print(f"  Progress: {done_count}/{len(cluster_files)} "
                      f"(ok={succeeded}, rejected={purity_rejected}, "
                      f"err={failed})")

            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                cid = Path(futures[fut]).stem
                print(f"  ERROR {cid}: {e}")
                continue

            status = res.get('status', 'unknown')
            cid = res.get('cluster_id', '?')

            n_total = res.get('n_total', 0)
            n_kept = res.get('n_kept', 0)
            n_out = res.get('n_filtered_out', 0)

            if status == 'ok':
                succeeded += 1
                total_filtered_out += n_out
                total_reads_kept += n_kept
                method = res.get('consensus_method', 'unknown')
                parts = [f"reads: {n_kept}/{n_total} kept", f"method: {method}"]
                if n_out > 0:
                    parts.append(f"{n_out} filtered out")
                if res.get('n_polish_reads') is not None and res['n_polish_reads'] != n_kept:
                    parts.append(f"polish_reads={res['n_polish_reads']}")
                if res.get('snap_edits', 0) > 0:
                    parts.append(f"snap_edits={res['snap_edits']}")
                if res.get('flank_edits', 0) > 0:
                    parts.append(f"flank_edits={res['flank_edits']}")
                if res.get('barcode_filter'):
                    parts.append(f"barcode_filter={res['barcode_filter']}")
                if res.get('barcode_interval_retry'):
                    parts.append(f"barcode_retry={res['barcode_interval_retry']}")
                if res.get('snap_warning'):
                    parts.append(res['snap_warning'])
                if res.get('flank_warning'):
                    parts.append(res['flank_warning'])
                if res.get('medaka_warning'):
                    parts.append(f"medaka: {res['medaka_warning']}")
                if res.get('racon_warning'):
                    parts.append(f"racon: {res['racon_warning']}")
                if res.get('polish_warning'):
                    parts.append(f"WARNING: {res['polish_warning']}")
                if res.get('polish_changes') is not None:
                    parts.append(f"polish_changes={res['polish_changes']}")
                print(f"  {cid}: OK — {', '.join(parts)}")
            elif status == 'purity_rejected':
                purity_rejected += 1
                total_filtered_out += n_total  # entire cluster rejected
                fs = res.get('filter_stats', {})
                print(f"  {cid}: REJECTED — reads: {n_kept}/{n_total} in dominant cluster — "
                      f"{fs.get('reason', '?')}")
            else:
                failed += 1
                print(f"  {cid}: {status}")

    print(f"\n  === Summary ===")
    print(f"  Clusters succeeded:  {succeeded}")
    print(f"  Clusters rejected:   {purity_rejected}")
    print(f"  Clusters failed:     {failed}")
    print(f"  Reads kept:          {total_reads_kept}")
    print(f"  Reads filtered out:  {total_filtered_out}")


# ═════════════════════════════════════════════════════════════════
# SINGLE-CLUSTER ENTRY POINT (backward compat)
# ═════════════════════════════════════════════════════════════════

def determine_consensus(threads, input_fn, medaka_model,
                        barcode_template=None, left_coding_flank=None,
                        right_coding_flank=None, max_mismatches=5, max_indels=3,
                        min_window_score=0.7, min_identity=0.85,
                        min_dominant_fraction=0.5,
                        min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                        min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES,
                        racon_rounds=2,
                        max_polish_reads=24,
                        keep_intermediates=False):
    """Single-cluster entry point. Calls the worker function directly."""
    result = _process_one_cluster(
        cluster_fq=input_fn,
        threads=threads,
        medaka_model=medaka_model,
        barcode_template=barcode_template,
        max_mismatches=max_mismatches,
        max_indels=max_indels,
        min_window_score=min_window_score,
        min_identity=min_identity,
        min_dominant_fraction=min_dominant_fraction,
        min_overlap_fraction=min_overlap_fraction,
        min_aligned_bases=min_aligned_bases,
        timeout=TIMEOUT,
        left_coding_flank=left_coding_flank,
        right_coding_flank=right_coding_flank,
        racon_rounds=racon_rounds,
        max_polish_reads=max_polish_reads,
        keep_intermediates=keep_intermediates,
    )

    status = result.get('status', 'unknown')
    cid = result.get('cluster_id', '?')
    n_total = result.get('n_total', 0)
    n_kept = result.get('n_kept', 0)
    n_out = result.get('n_filtered_out', 0)

    if status == 'ok':
        parts = [f"reads: {n_kept}/{n_total} kept"]
        if n_out > 0:
            parts.append(f"{n_out} filtered out")
        if result.get('n_polish_reads') is not None and result['n_polish_reads'] != n_kept:
            parts.append(f"polish_reads={result['n_polish_reads']}")
        if result.get('barcode_filter'):
            parts.append(f"barcode_filter={result['barcode_filter']}")
        if result.get('barcode_interval_retry'):
            parts.append(f"barcode_retry={result['barcode_interval_retry']}")
        if result.get('snap_edits', 0) > 0:
            parts.append(f"snap_edits={result['snap_edits']}")
        if result.get('flank_edits', 0) > 0:
            parts.append(f"flank_edits={result['flank_edits']}")
        if result.get('snap_warning'):
            parts.append(result['snap_warning'])
        if result.get('flank_warning'):
            parts.append(result['flank_warning'])
        print(f"  {cid}: OK — {', '.join(parts)}")
    elif status == 'purity_rejected':
        fs = result.get('filter_stats', {})
        print(f"  {cid}: SKIPPED — reads: {n_kept}/{n_total} in dominant cluster — "
              f"{fs.get('reason', '?')}")
    else:
        print(f"  {cid}: {status}")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser(
        description="Generate consensus: purity filter -> minimap2 draft -> "
                    "Medaka with optional barcode diagnostics. "
                    "Provide --output_dir to process all clusters (parallel), "
                    "or --input_fn for a single cluster."
    )

    # Mutually exclusive: batch vs single
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output_dir', type=str, default=None,
                      help='Directory containing clusters/ -- processes ALL clusters')
    mode.add_argument('--input_fn', type=str, default=None,
                      help='Single cluster FASTQ file to process')

    # Shared parameters
    parser.add_argument("--threads", type=int, default=8,
                        help="Total thread budget (split across workers in "
                             "parallel mode, default: 8)")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: auto, "
                             "= min(n_clusters, n_cpus/2))")
    parser.add_argument("--medaka_model", type=str, default="none")
    parser.add_argument("--racon_rounds", type=int, default=2,
                        help="Number of Racon polishing rounds when Medaka is "
                             "disabled (default: 2). More rounds can improve "
                             "accuracy but with diminishing returns past 2.")
    parser.add_argument("--max_polish_reads", type=int, default=24,
                        help="Maximum reads to send into Racon/Medaka per "
                             "cluster (default: 24). Use 0 or a negative "
                             "value to disable capping.")
    parser.add_argument("--barcode_template", type=str, default=None,
                        help="Degenerate barcode template (IUPAC codes). "
                             "If provided, the consensus is scanned for the "
                             "best-matching barcode interval for diagnostics "
                             "and flank anchoring.")
    parser.add_argument("--left_coding_flank", type=str, default=None,
                        help="Optional left coding flank to snap exactly in the "
                             "final consensus before mapping.")
    parser.add_argument("--right_coding_flank", type=str, default=None,
                        help="Optional right coding flank to snap exactly in the "
                             "final consensus before mapping.")
    parser.add_argument("--max_mismatches", type=int, default=5,
                        help="Max substitution errors for barcode snapping "
                             "(default: 5)")
    parser.add_argument("--max_indels", type=int, default=3,
                        help="Max indels for barcode snapping (default: 3)")
    parser.add_argument("--min_window_score", type=float, default=0.7,
                        help="Minimum fraction of positions matching template "
                             "to accept a window as the barcode region "
                             "(default: 0.7)")

    # Purity filter parameters
    parser.add_argument("--min_identity", type=float, default=0.85,
                        help="Minimum pairwise alignment identity for reads "
                             "to be considered part of the same sub-cluster "
                             "(default: 0.85)")
    parser.add_argument("--min_dominant_fraction", type=float, default=0.5,
                        help="Minimum fraction of reads in the dominant "
                             "sub-cluster to proceed with consensus. Clusters "
                             "below this are skipped entirely (default: 0.5)")
    parser.add_argument("--min_overlap_fraction", type=float,
                        default=DEFAULT_MIN_OVERLAP_FRACTION,
                        help="Minimum fraction of the shorter read that must "
                             "be covered by an overlap for that read pair to "
                             "enter the purity graph (default: "
                             f"{DEFAULT_MIN_OVERLAP_FRACTION})")
    parser.add_argument("--min_aligned_bases", type=int,
                        default=DEFAULT_MIN_ALIGNED_BASES,
                        help="Minimum aligned span in bases for an overlap to "
                             "enter the purity graph (default: "
                             f"{DEFAULT_MIN_ALIGNED_BASES})")

    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds for subprocess calls "
                             "(default: 3600 = 1h)")
    parser.add_argument("--keep_intermediates", action="store_true",
                        help="Keep draft FASTAs, medaka dirs, and PAFs for debugging")

    args = parser.parse_args()

    global TIMEOUT
    TIMEOUT = args.timeout

    print(f"=== DNABARMAP Consensus ===")
    print(f"Purity filter: min_identity={args.min_identity}, "
          f"min_dominant_fraction={args.min_dominant_fraction}")
    print(f"  Overlap graph: min_overlap_fraction={args.min_overlap_fraction}, "
          f"min_aligned_bases={args.min_aligned_bases}")

    if args.barcode_template:
        print(f"Barcode diagnostics: ON")
        print(f"  Template:  {args.barcode_template[:40]}... "
              f"({len(args.barcode_template)}bp)")
        print(f"  Max mismatches: {args.max_mismatches}, "
              f"Max indels: {args.max_indels}")
        print(f"  Min window score: {args.min_window_score}")

    if args.medaka_model and args.medaka_model.lower() != "none":
        args.medaka_model = _resolve_medaka_model(args.medaka_model)
        print(f"Polishing: Medaka (model={args.medaka_model})")
    else:
        print(f"Polishing: Racon ({args.racon_rounds} round(s))")
    if args.max_polish_reads <= 0:
        args.max_polish_reads = None
        print("Polish read cap: disabled")
    else:
        print(f"Polish read cap: {args.max_polish_reads}")

    print(f"Timeout: {TIMEOUT}s ({TIMEOUT / 3600:.1f}h)")
    start = time.time()

    if args.output_dir:
        print(f"Processing all clusters in {args.output_dir} (parallel)...")
        determine_consensus_parallel(
            output_dir=args.output_dir,
            total_threads=args.threads,
            medaka_model=args.medaka_model,
            barcode_template=args.barcode_template,
            left_coding_flank=args.left_coding_flank,
            right_coding_flank=args.right_coding_flank,
            max_mismatches=args.max_mismatches,
            max_indels=args.max_indels,
            min_window_score=args.min_window_score,
            min_identity=args.min_identity,
            min_dominant_fraction=args.min_dominant_fraction,
            min_overlap_fraction=args.min_overlap_fraction,
            min_aligned_bases=args.min_aligned_bases,
            n_workers=args.n_workers,
            timeout=args.timeout,
            racon_rounds=args.racon_rounds,
            max_polish_reads=args.max_polish_reads,
            keep_intermediates=args.keep_intermediates,
        )
    else:
        print(f"Determining consensus for {args.input_fn}...")
        determine_consensus(
            threads=args.threads,
            input_fn=args.input_fn,
            medaka_model=args.medaka_model,
            barcode_template=args.barcode_template,
            left_coding_flank=args.left_coding_flank,
            right_coding_flank=args.right_coding_flank,
            max_mismatches=args.max_mismatches,
            max_indels=args.max_indels,
            min_window_score=args.min_window_score,
            min_identity=args.min_identity,
            min_dominant_fraction=args.min_dominant_fraction,
            min_overlap_fraction=args.min_overlap_fraction,
            min_aligned_bases=args.min_aligned_bases,
            racon_rounds=args.racon_rounds,
            max_polish_reads=args.max_polish_reads,
            keep_intermediates=args.keep_intermediates,
        )

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == '__main__':
    cli()
