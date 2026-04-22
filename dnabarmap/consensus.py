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
import os
import signal
import subprocess
import argparse
import time
import shutil
import uuid
import re
import csv
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import statistics
import sys

from dnabarmap.utils import get_cluster_shard_dir

TIMEOUT = 3600  # 1 hour timeout for all subprocess calls
TIMEOUT_KILL_GRACE_SECONDS = 5
DEFAULT_MIN_OVERLAP_FRACTION = 0.5
DEFAULT_MIN_ALIGNED_BASES = 80
CONSENSUS_STRICTNESS_PRESETS = {
    'relaxed': {
        'min_identity': 0.80,
        'min_dominant_fraction': 0.35,
        'min_overlap_fraction': 0.45,
        'min_aligned_bases': 60,
        'max_mismatches': 8,
        'max_indels': 5,
        'min_window_score': 0.55,
    },
    'default': {
        'min_identity': 0.83,
        'min_dominant_fraction': 0.4,
        'min_overlap_fraction': DEFAULT_MIN_OVERLAP_FRACTION,
        'min_aligned_bases': DEFAULT_MIN_ALIGNED_BASES,
        'max_mismatches': 6,
        'max_indels': 4,
        'min_window_score': 0.65,
    },
    'strict': {
        'min_identity': 0.88,
        'min_dominant_fraction': 0.55,
        'min_overlap_fraction': 0.65,
        'min_aligned_bases': 120,
        'max_mismatches': 4,
        'max_indels': 2,
        'min_window_score': 0.78,
    },
}
ADVANCED_TUNING_KEYS = (
    'min_identity',
    'min_dominant_fraction',
    'min_overlap_fraction',
    'min_aligned_bases',
    'max_mismatches',
    'max_indels',
    'min_window_score',
)


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


def _terminate_process_group(proc, grace_seconds=TIMEOUT_KILL_GRACE_SECONDS):
    """Terminate a subprocess and any children it spawned."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _run_subprocess(cmd, *, timeout=None, check=False, **kwargs):
    """
    Run a subprocess in its own process group so a timeout can kill the entire
    spawned tree, not just the immediate wrapper process.
    """
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        **kwargs,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            stdout = None
            stderr = None
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout,
                                        output=stdout, stderr=stderr)

    completed = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )
    return completed


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


def _barcode_distance(barcode_a, barcode_b, min_comparable=12):
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

        result = _run_subprocess(
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


def _write_single_sequence_fasta(sequence, fasta_path, record_id):
    SeqIO.write(
        SeqRecord(Seq(sequence), id=record_id, description=""),
        fasta_path,
        "fasta",
    )


def _apply_final_consensus_corrections(cluster_fq, filtered_fq, consensus_seq,
                                       barcode_template, max_mismatches,
                                       max_indels, left_coding_flank=None,
                                       right_coding_flank=None):
    corrected_seq = consensus_seq
    authoritative_barcode = None
    authoritative_barcode_stats = None
    preferred_coding_len = None
    flank_stats = None
    barcode_reinjection_edits = 0

    if barcode_template:
        authoritative_barcode, authoritative_barcode_stats = _build_fullread_barcode_consensus(
            cluster_fq,
            consensus_seq=corrected_seq,
            barcode_template=barcode_template,
            max_mismatches=max_mismatches,
            max_indels=max_indels,
        )
        interval = None if authoritative_barcode_stats is None else authoritative_barcode_stats.get('interval')
        if authoritative_barcode and interval:
            injected_seq, _ = _inject_barcode_at_interval(
                corrected_seq,
                authoritative_barcode,
                interval[0],
                interval[1],
            )
            if injected_seq is not None and injected_seq != corrected_seq:
                barcode_reinjection_edits = (
                    sum(base_a != base_b for base_a, base_b in zip(corrected_seq, injected_seq))
                    + abs(len(injected_seq) - len(corrected_seq))
                )
                corrected_seq = injected_seq

    if left_coding_flank and right_coding_flank:
        preferred_coding_len = _estimate_coding_length_from_reads(
            filtered_fq,
            left_coding_flank,
            right_coding_flank,
        )
        corrected_with_flanks, flank_stats = snap_coding_flanks_in_consensus(
            corrected_seq,
            left_coding_flank=left_coding_flank,
            right_coding_flank=right_coding_flank,
            barcode_seq=authoritative_barcode,
            preferred_coding_len=preferred_coding_len,
        )
        if not flank_stats.get('rejected'):
            corrected_seq = corrected_with_flanks

    return corrected_seq, {
        'authoritative_barcode': authoritative_barcode,
        'authoritative_barcode_stats': authoritative_barcode_stats,
        'preferred_coding_len': preferred_coding_len,
        'flank_stats': flank_stats,
        'barcode_reinjection_edits': barcode_reinjection_edits,
    }


def _record_correction_stats(result, correction_info, original_seq, corrected_seq):
    authoritative_barcode_stats = correction_info.get('authoritative_barcode_stats')
    flank_stats = correction_info.get('flank_stats')

    if authoritative_barcode_stats is not None:
        result['n_barcode_consensus_reads'] = authoritative_barcode_stats.get('n_used', 0)
        if authoritative_barcode_stats.get('right_shift_retry'):
            result['barcode_interval_retry'] = 'end+1'
        elif authoritative_barcode_stats.get('left_shift_retry'):
            result['barcode_interval_retry'] = f"start-{authoritative_barcode_stats['left_shift_retry']}"
        elif authoritative_barcode_stats.get('projected_local_retry'):
            result['barcode_interval_retry'] = 'projected_local'
        else:
            result.pop('barcode_interval_retry', None)
        tail_ratio = authoritative_barcode_stats.get('tail_penultimate_ratio')
        if tail_ratio is not None:
            result['barcode_tail_penult'] = tail_ratio
        head_ratio = authoritative_barcode_stats.get('head_first_ratio')
        if head_ratio is not None:
            result['barcode_head_first'] = head_ratio
        if authoritative_barcode_stats.get('total_edits', 0) > 0:
            result['barcode_consensus_edits'] = authoritative_barcode_stats['total_edits']
        else:
            result.pop('barcode_consensus_edits', None)
    else:
        result.pop('n_barcode_consensus_reads', None)
        result.pop('barcode_interval_retry', None)
        result.pop('barcode_tail_penult', None)
        result.pop('barcode_head_first', None)
        result.pop('barcode_consensus_edits', None)

    if corrected_seq != original_seq and correction_info.get('barcode_reinjection_edits', 0) > 0:
        result['barcode_reinjected'] = True
        result['barcode_reinjection_edits'] = correction_info['barcode_reinjection_edits']
    else:
        result.pop('barcode_reinjected', None)
        result.pop('barcode_reinjection_edits', None)

    preferred_coding_len = correction_info.get('preferred_coding_len')
    if preferred_coding_len is not None:
        result['preferred_coding_len'] = preferred_coding_len
    else:
        result.pop('preferred_coding_len', None)

    if flank_stats is not None:
        if not flank_stats.get('rejected'):
            if flank_stats.get('total_edits', 0) > 0:
                result['flank_edits'] = flank_stats['total_edits']
            else:
                result.pop('flank_edits', None)
            result.pop('flank_warning', None)
        else:
            result['flank_warning'] = flank_stats.get('reason', 'unknown')
            result.pop('flank_edits', None)
    else:
        result.pop('flank_edits', None)
        result.pop('flank_warning', None)


def _summarize_consensus_support(input_fq, consensus_seq, threads=1):
    if not input_fq or not path.exists(input_fq) or not consensus_seq:
        return None

    with tempfile.TemporaryDirectory(prefix='dnabarmap_consensus_support_') as temp_dir:
        consensus_fasta = path.join(temp_dir, "consensus.fasta")
        _write_single_sequence_fasta(consensus_seq.upper(), consensus_fasta, "consensus")
        result = _run_subprocess(
            ["minimap2", "-x", "map-ont", "-c", "-t", str(threads), consensus_fasta, input_fq],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT,
        )
        if result.returncode != 0:
            return {
                'rejected': True,
                'reason': 'support_map_failed',
                'stderr_tail': result.stderr[-500:],
            }

    best_by_read = {}
    for line in result.stdout.splitlines():
        if not line or line.startswith('['):
            continue
        cols = line.split('\t')
        if len(cols) < 11:
            continue
        qname = cols[0]
        qlen = int(cols[1])
        qstart = int(cols[2])
        qend = int(cols[3])
        tstart = int(cols[7])
        tend = int(cols[8])
        nmatch = int(cols[9])
        aln_len = int(cols[10])
        if aln_len <= 0 or qlen <= 0:
            continue
        metrics = {
            'identity': nmatch / aln_len,
            'consensus_coverage': (tend - tstart) / max(1, len(consensus_seq)),
            'read_coverage': (qend - qstart) / qlen,
            'aln_len': aln_len,
        }
        prev = best_by_read.get(qname)
        if prev is None or metrics['aln_len'] > prev['aln_len']:
            best_by_read[qname] = metrics

    if not best_by_read:
        return {
            'rejected': True,
            'reason': 'no_support_alignments',
        }

    identities = sorted(m['identity'] for m in best_by_read.values())
    consensus_coverages = sorted(m['consensus_coverage'] for m in best_by_read.values())
    read_coverages = sorted(m['read_coverage'] for m in best_by_read.values())
    p10_idx = max(0, int(len(identities) * 0.1) - 1)

    return {
        'rejected': False,
        'n_aligned': len(best_by_read),
        'median_identity': round(statistics.median(identities), 4),
        'p10_identity': round(identities[p10_idx], 4),
        'median_consensus_coverage': round(statistics.median(consensus_coverages), 4),
        'p10_consensus_coverage': round(consensus_coverages[p10_idx], 4),
        'median_read_coverage': round(statistics.median(read_coverages), 4),
        'p10_read_coverage': round(read_coverages[p10_idx], 4),
    }


def _should_run_full_refine_fallback(result, correction_info, support_stats):
    authoritative_barcode_stats = correction_info.get('authoritative_barcode_stats') or {}
    flank_stats = correction_info.get('flank_stats') or {}

    fallback_reasons = []
    if result.get('polish_warning'):
        fallback_reasons.append('polish_no_effect')
    if authoritative_barcode_stats.get('right_shift_retry'):
        fallback_reasons.append('barcode_shift_right')
    if authoritative_barcode_stats.get('left_shift_retry'):
        fallback_reasons.append('barcode_shift_left')
    if authoritative_barcode_stats.get('projected_local_retry'):
        fallback_reasons.append('barcode_projected_retry')
    if flank_stats.get('rejected'):
        fallback_reasons.append('flank_rejected')
    if support_stats and not support_stats.get('rejected'):
        if support_stats.get('median_consensus_coverage', 1.0) < 0.9:
            fallback_reasons.append('low_consensus_coverage')
        elif support_stats.get('p10_consensus_coverage', 1.0) < 0.75:
            fallback_reasons.append('partial_consensus_tail')

    return fallback_reasons


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
    min_comparable = min(barcode_len, max(8, int(barcode_len * 0.7)))
    max_distance = max(3, int(round(barcode_len * 0.16)))

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


def _select_polishing_records(records, draft_support=None, max_reads=50):
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


def _load_single_fasta_sequence(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return None
    return str(records[0].seq)


def _prepare_unique_fastq_ids(input_reads, output_dir):
    """
    Racon rejects FASTQs that reuse a read ID for different sequences.
    Rewrite only duplicates, preserving qualities, when needed.
    """
    seen = Counter()
    rewritten = []
    needs_rewrite = False

    for record in SeqIO.parse(input_reads, "fastq"):
        seen[record.id] += 1
        if seen[record.id] > 1:
            needs_rewrite = True
            record.id = f"{record.id}__dup{seen[record.id]}"
            record.name = record.id
            record.description = ""
        rewritten.append(record)

    if not needs_rewrite:
        return input_reads

    sanitized_reads = path.join(output_dir, "reads_unique.fastq")
    with open(sanitized_reads, "w") as out_handle:
        SeqIO.write(rewritten, out_handle, "fastq")
    return sanitized_reads
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


def _format_log_tail(log_content, max_chars=2000, max_lines=40):
    if not log_content:
        return "(no log output captured)"

    lines = [line.rstrip() for line in log_content.splitlines()]
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return "(log contained only blank lines)"

    tail_lines = non_empty[-max_lines:]
    tail = "\n".join(tail_lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _extract_medaka_failure_summary(log_content, allow_generic=False):
    if not log_content:
        return None

    lines = [re.sub(r'\s+', ' ', line).strip()
             for line in log_content.splitlines()
             if line.strip()]
    if not lines:
        return None

    strong_markers = (
        "Failed to run alignment",
        "Failed to run inference",
        "Failed to run sequence",
        "FileNotFoundError",
        "No such file or directory",
        "is not a recognised basecaller model",
        "is not a recognized basecaller model",
        "Traceback (most recent call last)",
        "Segmentation fault",
        "Killed",
    )

    for marker in strong_markers:
        for idx, line in enumerate(lines):
            if marker in line:
                if marker == "Traceback (most recent call last)":
                    for follow in reversed(lines[idx + 1:]):
                        if follow and "Traceback (most recent call last)" not in follow:
                            return follow
                return line

    if allow_generic:
        generic_patterns = (
            re.compile(r'\b(error|failed|exception)\b', re.IGNORECASE),
            re.compile(r'^\[[^\]]+\]\s*error', re.IGNORECASE),
        )
        for line in reversed(lines):
            if any(pattern.search(line) for pattern in generic_patterns):
                return line

    return None


def _medaka_runtime_error(prefix, cmd, log_content, detail=None):
    tail = _format_log_tail(log_content)
    first_line = prefix
    if detail:
        first_line = f"{first_line}: {detail}"
    return RuntimeError(
        f"{first_line}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"LOG tail:\n{tail}"
    )


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
        result = _run_subprocess(
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

    # medaka_consensus bash script does not always propagate internal failures
    # via exit code, so inspect the log for high-confidence failure markers.
    internal_failure = _extract_medaka_failure_summary(log_content)
    if internal_failure:
        raise _medaka_runtime_error(
            "medaka_consensus failed internally",
            cmd,
            log_content,
            detail=internal_failure,
        )

    if result.returncode != 0:
        failure_detail = _extract_medaka_failure_summary(
            log_content,
            allow_generic=True,
        )
        raise _medaka_runtime_error(
            f"medaka_consensus failed (exit {result.returncode})",
            cmd,
            log_content,
            detail=failure_detail,
        )

    # medaka_consensus outputs to consensus.fasta in the output dir
    consensus_out = path.join(output_dir, "consensus.fasta")

    if not path.exists(consensus_out) or path.getsize(consensus_out) == 0:
        fastas = glob(path.join(output_dir, "*.fasta"))
        fastas = [f for f in fastas
                  if 'draft' not in path.basename(f).lower()
                  and path.basename(f) != path.basename(draft_fasta)
                  and path.getsize(f) > 0]
        if fastas:
            consensus_out = fastas[0]
        else:
            failure_detail = _extract_medaka_failure_summary(
                log_content,
                allow_generic=True,
            )
            raise _medaka_runtime_error(
                f"medaka_consensus produced no FASTA output in {output_dir}",
                cmd,
                log_content,
                detail=failure_detail,
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
    input_reads = _prepare_unique_fastq_ids(input_reads, output_dir)

    current_draft = draft_fasta

    for rnd in range(rounds):
        paf = path.join(output_dir, f"overlaps_r{rnd}.paf")
        out = path.join(output_dir, f"racon_r{rnd}.fasta")

        # Align reads to current draft
        with open(paf, "w") as paf_handle:
            result = _run_subprocess(
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
            result = _run_subprocess(
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
                         max_polish_reads=50,
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
    sub_dir = get_cluster_shard_dir(cluster_id, consensus_dir)

    work_dir = path.join(sub_dir, f"{cluster_id}_work_{uuid.uuid4().hex[:8]}")
    makedirs(work_dir, exist_ok=True)

    work_cluster_fq = path.join(work_dir, path.basename(cluster_fq))
    shutil.copy2(cluster_fq, work_cluster_fq)

    medaka_dir = f"{work_dir}/{cluster_id}_medaka"

    final_consensus = f"{sub_dir}/{cluster_id}_consensus.fasta"
    draft_fasta = f"{work_dir}/{cluster_id}_draft.fasta"
    paf_file = f"{work_dir}/{cluster_id}.paf"

    result = {'cluster_id': cluster_id, 'status': 'ok'}

    # ── Step 1: minimap2 self-map ────────────────────────────────
    try:
        with open(paf_file, "w") as paf_handle:
            map_result = _run_subprocess(
                ["minimap2", "-x", "ava-ont", "-c", "-t", str(threads),
                 work_cluster_fq, work_cluster_fq],
                stdout=paf_handle,
                stderr=subprocess.PIPE,
                timeout=TIMEOUT
            )
        if map_result.returncode != 0:
            stderr_tail = map_result.stderr.decode(errors='replace')[-500:]
            raise RuntimeError(f"minimap2 self-map failed\nSTDERR:\n{stderr_tail}")
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout_selfmap'
        if path.exists(paf_file):
            remove(paf_file)
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result
    except Exception as e:
        result['status'] = f'error_selfmap: {e}'
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    # ── Step 2: Purity filter ────────────────────────────────────
    filtered_fq, draft_support, filter_stats = filter_cluster_reads(
        work_cluster_fq, paf_file,
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

    barcode_filter_input_n = filter_stats.get('n_kept', result['n_total']) if filtered_fq else result['n_total']
    should_try_barcode_filter = barcode_template and barcode_filter_input_n >= 8

    if should_try_barcode_filter:
        barcode_filter_input = filtered_fq if filtered_fq is not None else work_cluster_fq
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
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    use_barcode_filter = filtered_fq is None and barcode_filtered_fq is not None

    if use_barcode_filter:
        if filtered_fq and filtered_fq != work_cluster_fq and filtered_fq != barcode_filtered_fq:
            _cleanup_filtered(filtered_fq, work_cluster_fq)
        filtered_fq = barcode_filtered_fq
        result['n_kept'] = barcode_filter_stats.get('n_kept', result['n_kept'])
        result['n_filtered_out'] = result['n_total'] - result['n_kept']
        result['barcode_group_size'] = barcode_filter_stats.get('n_kept', 0)
        if barcode_filter_stats.get('filtered'):
            result['barcode_filter'] = (
                f"{barcode_filter_stats.get('n_kept', 0)}/"
                f"{barcode_filter_stats.get('n_total', 0)} kept"
            )
    elif (
        barcode_filtered_fq
        and barcode_filtered_fq != work_cluster_fq
        and barcode_filtered_fq != filtered_fq
    ):
        _cleanup_filtered(barcode_filtered_fq, work_cluster_fq)

    # ── Step 3: Pick best draft read ─────────────────────────────
    filtered_records = list(SeqIO.parse(filtered_fq, "fastq"))
    if barcode_keep_ids:
        draft_support = {rid: score for rid, score in draft_support.items()
                         if rid in barcode_keep_ids}

    if not filtered_records:
        result['status'] = 'no_draft'
        _cleanup_filtered(filtered_fq, work_cluster_fq)
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    if not draft_support and barcode_keep_ids:
        draft_support = {record.id: 0.0 for record in filtered_records}

    draft_record = _choose_draft_record(filtered_records, draft_support=draft_support)
    if draft_record is None:
        result['status'] = 'no_draft'
        _cleanup_filtered(filtered_fq, work_cluster_fq)
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    # Give the draft a unique identifier so downstream polishers do not see
    # the draft target as a duplicate of one of the input reads.
    _write_single_sequence_fasta(str(draft_record.seq), draft_fasta, f"{cluster_id}_draft")

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
    pileup_dir = f"{work_dir}/{cluster_id}_pileup"
    racon_dir = f"{work_dir}/{cluster_id}_racon"
    full_refine_dir = f"{work_dir}/{cluster_id}_full_refine"

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

    consensus_seq = _load_single_fasta_sequence(polished)
    if consensus_seq is None:
        result['status'] = 'no_consensus'
        _cleanup_filtered(filtered_fq, work_cluster_fq)
        if not keep_intermediates and path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        return result

    # ── Diagnostic: check if polishing actually changed the draft ─
    if path.exists(draft_fasta):
        draft_seq = _load_single_fasta_sequence(draft_fasta)
        if draft_seq is not None and consensus_seq == draft_seq:
            result['polish_warning'] = 'consensus identical to draft (polishing had no effect)'

    # ── Step 5: Final correction; full refine only as fallback ───
    pre_correction_seq = consensus_seq
    consensus_seq, correction_info = _apply_final_consensus_corrections(
        work_cluster_fq,
        filtered_fq,
        consensus_seq,
        barcode_template,
        max_mismatches,
        max_indels,
        left_coding_flank=left_coding_flank,
        right_coding_flank=right_coding_flank,
    )
    _record_correction_stats(result, correction_info, pre_correction_seq, consensus_seq)

    full_refine_rounds = 0
    support_stats = _summarize_consensus_support(filtered_fq, consensus_seq, threads=threads)
    if support_stats is not None:
        if support_stats.get('rejected'):
            result['consensus_support_warning'] = support_stats.get('reason', 'unknown')
        else:
            result['median_consensus_coverage'] = support_stats['median_consensus_coverage']
            result['median_read_coverage'] = support_stats['median_read_coverage']
            result['median_polish_identity'] = support_stats['median_identity']

    full_refine_reasons = _should_run_full_refine_fallback(result, correction_info, support_stats)
    if full_refine_reasons and polish_fq and path.exists(polish_fq):
        round_dir = path.join(full_refine_dir, "round_1")
        corrected_draft_fasta = path.join(full_refine_dir, f"{cluster_id}_corrected_draft.fasta")
        makedirs(full_refine_dir, exist_ok=True)
        _write_single_sequence_fasta(pre_correction_seq, corrected_draft_fasta, f"{cluster_id}_corrected")

        try:
            if medaka_model and medaka_model.lower() != "none":
                refined_fasta = run_medaka(
                    input_reads=polish_fq,
                    draft_fasta=corrected_draft_fasta,
                    output_dir=round_dir,
                    model=medaka_model,
                    threads=threads,
                )
            else:
                refined_fasta = run_racon(
                    input_reads=polish_fq,
                    draft_fasta=corrected_draft_fasta,
                    output_dir=round_dir,
                    threads=threads,
                    rounds=1,
                )

            refined_seq = _load_single_fasta_sequence(refined_fasta)
            if refined_seq is not None and refined_seq != consensus_seq:
                pre_correction_seq = refined_seq
                consensus_seq, correction_info = _apply_final_consensus_corrections(
                    work_cluster_fq,
                    filtered_fq,
                    refined_seq,
                    barcode_template,
                    max_mismatches,
                    max_indels,
                    left_coding_flank=left_coding_flank,
                    right_coding_flank=right_coding_flank,
                )
                _record_correction_stats(result, correction_info, pre_correction_seq, consensus_seq)
                full_refine_rounds = 1
                result['full_refine_trigger'] = ','.join(full_refine_reasons)
        except subprocess.TimeoutExpired:
            result['full_refine_warning'] = 'timeout'
        except Exception as e:
            result['full_refine_warning'] = str(e)

    if full_refine_rounds > 0:
        result['full_refine_rounds'] = full_refine_rounds
        if result.get('consensus_method') == 'medaka':
            result['consensus_method'] = 'medaka + full refine'
        elif result.get('consensus_method') == 'racon':
            result['consensus_method'] = 'racon + full refine'
        elif result.get('consensus_method'):
            result['consensus_method'] = f"{result['consensus_method']} + full refine"

    # ── Write final consensus ────────────────────────────────────
    header_fields = [
        f"n_cluster_reads={result.get('n_total', 0)}",
        f"n_consensus_reads={result.get('n_kept', 0)}",
        f"n_polish_reads={result.get('n_polish_reads', 0)}",
    ]
    record = SeqRecord(
        Seq(consensus_seq),
        id=cluster_id,
        description=" ".join(header_fields),
    )
    SeqIO.write(record, final_consensus, "fasta")

    # ── Cleanup ──────────────────────────────────────────────────
    if not keep_intermediates:
        if path.exists(draft_fasta):
            remove(draft_fasta)
        if path.isdir(medaka_dir):
            shutil.rmtree(medaka_dir, ignore_errors=True)
        if path.isdir(racon_dir):
            shutil.rmtree(racon_dir, ignore_errors=True)
        if path.isdir(full_refine_dir):
            shutil.rmtree(full_refine_dir, ignore_errors=True)
        if path.isdir(pileup_dir):
            shutil.rmtree(pileup_dir, ignore_errors=True)
        _cleanup_filtered(filtered_fq, work_cluster_fq)
        _cleanup_filtered(polish_fq, filtered_fq)
        if path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


def _cleanup_filtered(filtered_fq, original_fq):
    """Remove the filtered FASTQ if it's a separate file."""
    if filtered_fq and filtered_fq != original_fq and path.exists(filtered_fq):
        remove(filtered_fq)


def _is_consensus_intermediate_fastq(filename):
    return path.basename(filename).endswith((
        '_filtered.fastq',
        '_barcode_filtered.fastq',
        '_polish.fastq',
    ))


def _validate_consensus_input_fastq(input_fn):
    if not input_fn:
        raise ValueError("No input FASTQ provided for consensus")
    if not path.exists(input_fn):
        raise FileNotFoundError(f"Consensus input FASTQ not found: {input_fn}")
    if _is_consensus_intermediate_fastq(input_fn):
        raise ValueError(
            "Consensus was given an intermediate FASTQ instead of an original cluster FASTQ: "
            f"{input_fn}. Submit files like cluster_*.fastq from clusters/full_seqs/, "
            "not *_filtered.fastq, *_barcode_filtered.fastq, or *_polish.fastq."
        )


def _apply_consensus_strictness(args):
    preset_values = CONSENSUS_STRICTNESS_PRESETS[args.strictness]
    advanced_overrides = {
        key: getattr(args, key)
        for key in ADVANCED_TUNING_KEYS
        if getattr(args, key) is not None
    }

    for key, value in preset_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return advanced_overrides


def _format_barcode_retry_label(retry_value):
    if not retry_value:
        return None
    if retry_value == 'end+1':
        return "barcode interval shifted right"
    if retry_value == 'projected_local':
        return "barcode interval recalculated"
    if retry_value.startswith('start-'):
        shift = retry_value.split('-', 1)[1]
        return f"barcode interval shifted left {shift} bp"
    return f"barcode interval {retry_value}"


def _short_warning_text(text):
    if not text:
        return None
    short = str(text).strip().splitlines()[0].strip()
    return re.sub(r'\s+', ' ', short)


def _format_cluster_status(res):
    status = res.get('status', 'unknown')
    cid = res.get('cluster_id', '?')
    n_total = res.get('n_total', 0)
    n_kept = res.get('n_kept', 0)

    if status == 'ok':
        parts = [f"kept {n_kept}/{n_total} reads"]

        method = res.get('consensus_method')
        if method:
            parts.append(method)

        if res.get('n_polish_reads') is not None and res['n_polish_reads'] != n_kept:
            parts.append(f"polished {res['n_polish_reads']} reads")

        if res.get('full_refine_rounds'):
            rounds = res['full_refine_rounds']
            parts.append(f"full refine x{rounds}")

        if res.get('barcode_filter'):
            parts.append("barcode filter applied")

        retry_label = _format_barcode_retry_label(res.get('barcode_interval_retry'))
        if retry_label:
            parts.append(retry_label)

        if res.get('flank_edits', 0) > 0:
            parts.append("coding flanks corrected")

        warnings = []
        if res.get('snap_warning'):
            warnings.append(_short_warning_text(res['snap_warning']))
        if res.get('flank_warning'):
            warnings.append(_short_warning_text(res['flank_warning']))
        if res.get('consensus_support_warning'):
            warnings.append(f"support: {_short_warning_text(res['consensus_support_warning'])}")
        if res.get('medaka_warning'):
            warnings.append(f"medaka fallback: {_short_warning_text(res['medaka_warning'])}")
        if res.get('racon_warning'):
            warnings.append(f"racon fallback: {_short_warning_text(res['racon_warning'])}")
        if res.get('full_refine_warning'):
            warnings.append(f"full refine: {_short_warning_text(res['full_refine_warning'])}")
        if res.get('polish_warning'):
            warnings.append(_short_warning_text(res['polish_warning']))

        message = f"{cid}: ok | " + " | ".join(parts)
        if warnings:
            message += " | warning: " + "; ".join(warnings)
        return message

    if status == 'purity_rejected':
        fs = res.get('filter_stats', {})
        reason = fs.get('reason', '?')
        return f"{cid}: skipped | kept {n_kept}/{n_total} reads after filtering | {reason}"

    return f"{cid}: failed | {status}"


# ═════════════════════════════════════════════════════════════════
# BATCH CONSENSUS (parallel)
# ═════════════════════════════════════════════════════════════════

def determine_consensus_parallel(output_dir, total_threads=8,
                                  medaka_model='none',
                                  barcode_template=None,
                                  left_coding_flank=None,
                                  right_coding_flank=None,
                                  max_mismatches=6, max_indels=4,
                                  min_window_score=0.65,
                                  min_identity=0.83,
                                  min_dominant_fraction=0.4,
                                  min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                                  min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES,
                                  n_workers=None,
                                  timeout=3600,
                                  racon_rounds=2,
                                  max_polish_reads=50,
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
        if not _is_consensus_intermediate_fastq(f)
    ]

    if not cluster_files:
        print(f"  WARNING: No cluster FASTQ files found in {full_seqs_dir}/")
        return

    print(f"  Clusters: {len(cluster_files)}")

    # ── Determine parallelism ────────────────────────────────────
    n_cpus = multiprocessing.cpu_count()

    if n_workers is None:
        n_workers = min(len(cluster_files), max(1, n_cpus // 2))

    threads_per_worker = max(1, total_threads // n_workers)

    print(f"  Workers: {n_workers} x {threads_per_worker} threads "
          f"({n_workers * threads_per_worker} total, {n_cpus} CPUs available)")

    # ── Validate medaka if needed ────────────────────────────────
    if medaka_model and medaka_model.lower() != "none":
        if not _check_medaka_available():
            raise RuntimeError(
                "Medaka not found. Install with: pip install medaka\n"
                "Or disable with --medaka_model none"
            )
        print(f"  Medaka: {medaka_model}")

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
                      f"(ok={succeeded}, skipped={purity_rejected}, failed={failed})")

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
                print(f"  {_format_cluster_status(res)}")
            elif status == 'purity_rejected':
                purity_rejected += 1
                total_filtered_out += n_total  # entire cluster rejected
                print(f"  {_format_cluster_status(res)}")
            else:
                failed += 1
                print(f"  {_format_cluster_status(res)}")

    print(f"\n  === Summary ===")
    print(f"  Clusters: {succeeded} ok, {purity_rejected} skipped, {failed} failed")
    print(f"  Reads: {total_reads_kept} kept, {total_filtered_out} removed")


# ═════════════════════════════════════════════════════════════════
# SINGLE-CLUSTER ENTRY POINT (backward compat)
# ═════════════════════════════════════════════════════════════════

def determine_consensus(threads, input_fn, medaka_model,
                        barcode_template=None, left_coding_flank=None,
                        right_coding_flank=None, max_mismatches=6, max_indels=4,
                        min_window_score=0.65, min_identity=0.83,
                        min_dominant_fraction=0.4,
                        min_overlap_fraction=DEFAULT_MIN_OVERLAP_FRACTION,
                        min_aligned_bases=DEFAULT_MIN_ALIGNED_BASES,
                        racon_rounds=2,
                        max_polish_reads=50,
                        keep_intermediates=False):
    """Single-cluster entry point. Calls the worker function directly."""
    _validate_consensus_input_fastq(input_fn)
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
        print(f"  {_format_cluster_status(result)}")
    elif status == 'purity_rejected':
        print(f"  {_format_cluster_status(result)}")
    else:
        print(f"  {_format_cluster_status(result)}")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser(
        description="Generate consensus: purity filter -> minimap2 draft -> "
                    "polishing, with optional barcode diagnostics. Provide "
                    "--output_dir to process all clusters in parallel, or "
                    "--input_fn for a single cluster FASTQ."
    )

    # Mutually exclusive: batch vs single
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output_dir', type=str, default=None,
                      help='Pipeline output directory containing clusters/. '
                           'Processes all clusters under clusters/full_seqs/.')
    mode.add_argument('--input_fn', type=str, default=None,
                      help='Single cluster FASTQ file to process.')

    # Shared parameters
    parser.add_argument("--threads", type=int, default=8,
                        help="Total thread budget (split across workers in "
                             "parallel mode, default: 8)")
    parser.add_argument("--workers", dest="n_workers", type=int, default=None,
                        help="Parallel worker count when processing an "
                             "output_dir (default: auto).")
    parser.add_argument("--n_workers", dest="n_workers", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--medaka_model", type=str, default="none",
                        help="Medaka model name, or 'none' to skip Medaka and "
                             "use the non-Medaka consensus path.")
    parser.add_argument("--racon_rounds", type=int, default=2,
                        help="Number of Racon polishing rounds when Medaka is "
                             "disabled (default: 2). More rounds can improve "
                             "accuracy but with diminishing returns past 2.")
    parser.add_argument("--max_polish_reads", type=int, default=100,
                        help="Maximum reads to send into Racon/Medaka per "
                             "cluster (default: 100). Use 0 or a negative "
                             "value to disable capping.")
    parser.add_argument("--strictness", choices=tuple(CONSENSUS_STRICTNESS_PRESETS),
                        default="default",
                        help="Consensus filtering preset. 'relaxed' keeps more "
                             "reads, 'strict' filters harder, 'default' is the "
                             "recommended balance.")
    parser.add_argument("--barcode_template", type=str, default=None,
                        help="Degenerate barcode template (IUPAC codes). "
                             "If provided, the consensus is scanned for the "
                             "best-matching barcode interval for diagnostics "
                             "and barcode interval recalculation.")
    parser.add_argument("--left_coding_flank", type=str, default=None,
                        help="Optional left coding flank to snap exactly in the "
                             "final consensus before mapping.")
    parser.add_argument("--right_coding_flank", type=str, default=None,
                        help="Optional right coding flank to snap exactly in the "
                             "final consensus before mapping.")
    parser.add_argument("--max_mismatches", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max_indels", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--min_window_score", type=float, default=None,
                        help=argparse.SUPPRESS)

    # Expert-only filter overrides
    parser.add_argument("--min_identity", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--min_dominant_fraction", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--min_overlap_fraction", type=float,
                        default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--min_aligned_bases", type=int,
                        default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds for subprocess calls "
                             "(default: 3600 = 1h)")
    parser.add_argument("--keep_intermediates", action="store_true",
                        help="Keep draft FASTAs, Medaka directories, and PAFs "
                             "for debugging.")

    args = parser.parse_args()
    advanced_overrides = _apply_consensus_strictness(args)

    global TIMEOUT
    TIMEOUT = args.timeout

    print(f"=== DNABARMAP Consensus ===")
    print(f"Settings:")
    print(f"  Strictness: {args.strictness}")
    if advanced_overrides:
        print(f"  Expert overrides: on")

    if args.barcode_template:
        print(f"  Barcode diagnostics: on ({len(args.barcode_template)} bp template)")

    if args.medaka_model and args.medaka_model.lower() != "none":
        args.medaka_model = _resolve_medaka_model(args.medaka_model)
        print(f"  Polishing: medaka")
    else:
        print(f"  Polishing: racon ({args.racon_rounds} round(s))")
    if args.max_polish_reads <= 0:
        args.max_polish_reads = None
        print("  Polish read cap: disabled")
    else:
        print(f"  Polish read cap: {args.max_polish_reads}")

    print(f"  Timeout: {TIMEOUT / 3600:.1f}h")
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
