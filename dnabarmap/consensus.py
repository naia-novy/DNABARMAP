#!/usr/bin/env python3

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
from collections import defaultdict
import sys

TIMEOUT = 3600  # 1 hour timeout for all subprocess calls


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


def _find_barcode_region(consensus_seq, template, search_margin=3):
    template_len = len(template)
    seq_upper = consensus_seq.upper()
    seq_len = len(seq_upper)

    best_start = best_end = -1
    best_score = -1.0
    best_orient = None
    best_tmpl = template

    rc_template = _reverse_complement_template(template)

    for tmpl, orient in [(template, 'fwd'), (rc_template, 'rc')]:
        for window_len in range(template_len - search_margin, template_len + search_margin + 1):
            if window_len < 1 or window_len > seq_len:
                continue
            if window_len == template_len:
                for start in range(seq_len - window_len + 1):
                    subseq = seq_upper[start:start + window_len]
                    score = _template_match_score(subseq, tmpl)
                    if score > best_score:
                        best_score = score
                        best_start = start
                        best_end = start + window_len
                        best_orient = orient
                        best_tmpl = tmpl
            else:
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
                               min_window_score=0.7):
    consensus_upper = consensus_seq.upper()
    template_upper = barcode_template.upper()

    start, end, orient, window_score, effective_template = _find_barcode_region(
        consensus_upper, template_upper, search_margin=max_indels
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


# ═════════════════════════════════════════════════════════════════
# MEDAKA DETECTION
# ═════════════════════════════════════════════════════════════════

def _detect_medaka():
    import shutil
    if shutil.which("medaka"):
        return "inference"
    if shutil.which("medaka_consensus"):
        return "consensus"
    return None

_MEDAKA_INTERFACE = None

def _get_medaka_interface():
    global _MEDAKA_INTERFACE
    if _MEDAKA_INTERFACE is None:
        _MEDAKA_INTERFACE = _detect_medaka()
        if _MEDAKA_INTERFACE is None:
            raise RuntimeError(
                "Medaka not found. Install with: pip install medaka\n"
                "Or disable with --medaka_model none"
            )
        print(f"  Medaka interface: '{_MEDAKA_INTERFACE}'")
    return _MEDAKA_INTERFACE


# ═════════════════════════════════════════════════════════════════
# MEDAKA POLISHING
# ═════════════════════════════════════════════════════════════════

def run_medaka(input_reads, draft_fasta, output_dir, model, threads):
    makedirs(output_dir, exist_ok=True)

    bam = path.join(output_dir, "calls_to_draft.bam")
    hdf = path.join(output_dir, "consensus_probs.hdf")
    out = path.join(output_dir, "consensus.fasta")

    # Step 1: Align
    align_cmd = (
        f"minimap2 -ax map-ont -t {threads} {draft_fasta} {input_reads} | "
        f"samtools sort -@ {threads} -o {bam}"
    )
    result = subprocess.run(align_cmd, shell=True, capture_output=True, text=True,
                            timeout=TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Alignment failed\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    # Index BAM
    subprocess.run(["samtools", "index", bam], check=True, timeout=TIMEOUT)

    # Step 2: Medaka inference
    result = subprocess.run(
        ["medaka", "inference", "--threads", str(threads),
         "--model", model, bam, hdf],
        capture_output=True, text=True, timeout=TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Medaka inference failed\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    # Step 3: Medaka sequence
    result = subprocess.run(
        ["medaka", "sequence", "--threads", str(threads),
         hdf, draft_fasta, out],
        capture_output=True, text=True, timeout=TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Medaka sequence failed\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    for f in [bam, bam + ".bai", hdf]:
        if path.exists(f):
            remove(f)

    return out


# ═════════════════════════════════════════════════════════════════
# CONSENSUS GENERATION
# ═════════════════════════════════════════════════════════════════

def determine_consensus(threads, input_fq, medaka_model,
                        barcode_template=None, max_mismatches=5, max_indels=3,
                        min_window_score=0.7):
    cluster_id = input_fq.split(".")[0].split('/')[-1]
    consensus_dir = input_fq.split('clusters')[0] + '/consensus/'
    sub_dir = consensus_dir + cluster_id[-2:]
    medaka_dir = f"{sub_dir}/{cluster_id}_medaka"

    makedirs(sub_dir, exist_ok=True)

    final_consensus = f"{sub_dir}/{cluster_id}_consensus.fasta"
    draft_fasta = f"{sub_dir}/{cluster_id}_draft.fasta"

    # ── Pick best read via minimap2 self-map ─────────────────────
    paf_file = f"{sub_dir}/{cluster_id}.paf"

    try:
        subprocess.run(
            ["minimap2", "-x", "map-ont", "-t", str(threads),
             input_fq, input_fq],
            stdout=open(paf_file, "w"),
            check=True,
            timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"  WARNING: minimap2 self-map timed out for {cluster_id} (1h limit)")
        if path.exists(paf_file):
            remove(paf_file)
        return

    read_counts = defaultdict(int)
    with open(paf_file) as f:
        for line in f:
            read_counts[line.split('\t')[5]] += 1
    remove(paf_file)

    if not read_counts:
        print(f"  WARNING: No alignments for {cluster_id}")
        return

    best_read_id = max(read_counts, key=read_counts.get)

    for record in SeqIO.parse(input_fq, "fastq"):
        if record.id == best_read_id:
            SeqIO.write(record, draft_fasta, "fasta")
            break

    # ── Run Medaka (skip if model is 'none') ────────────────────
    if medaka_model and medaka_model.lower() != "none":
        try:
            polished = run_medaka(
                input_reads=input_fq,
                draft_fasta=draft_fasta,
                output_dir=medaka_dir,
                model=medaka_model,
                threads=threads
            )
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Medaka timed out for {cluster_id} (1h limit), keeping draft.")
            polished = draft_fasta
        except Exception as e:
            print(f"  WARNING: Medaka failed for {cluster_id}, keeping draft.")
            print(f"    {e}")
            polished = draft_fasta
    else:
        polished = draft_fasta

    records = list(SeqIO.parse(polished, "fasta"))
    if not records:
        print(f"  WARNING: No consensus produced for {cluster_id}")
        return

    consensus_seq = str(records[0].seq)

    # ── Snap barcode to degenerate template ──────────────────────
    if barcode_template:
        corrected_seq, snap_stats = snap_barcode_in_consensus(
            consensus_seq,
            barcode_template=barcode_template,
            max_mismatches=max_mismatches,
            max_indels=max_indels,
            min_window_score=min_window_score,
        )

        if corrected_seq is not None:
            consensus_seq = corrected_seq
            if snap_stats['total_edits'] > 0:
                print(f"  {cluster_id}: barcode snapped "
                      f"(edits={snap_stats['total_edits']}, "
                      f"mis={snap_stats['mismatches']}, "
                      f"ins={snap_stats['insertions']}, "
                      f"del={snap_stats['deletions']}, "
                      f"orient={snap_stats.get('orientation', '?')}, "
                      f"window={snap_stats.get('window_score', 0):.3f})")
        else:
            reason = snap_stats.get('reason', 'unknown')
            print(f"  {cluster_id}: barcode snap failed ({reason}), keeping original")

    # ── Write final consensus ────────────────────────────────────
    record = SeqRecord(
        Seq(consensus_seq),
        id=cluster_id,
        description=""
    )
    SeqIO.write(record, final_consensus, "fasta")

    # Cleanup
    if path.exists(draft_fasta):
        remove(draft_fasta)
    if path.isdir(medaka_dir):
        shutil.rmtree(medaka_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser(
        description="Generate consensus: minimap2 draft -> Medaka -> barcode snap"
    )
    parser.add_argument('--input_fq', required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--medaka_model", type=str,
                        default="r941_min_high_g360")

    parser.add_argument("--barcode_template", type=str, default=None,
                        help="Degenerate barcode template (IUPAC codes). "
                             "If provided, the consensus is scanned for the "
                             "best-matching region and corrected in place.")
    parser.add_argument("--max_mismatches", type=int, default=5,
                        help="Max substitution errors for barcode snapping (default: 5)")
    parser.add_argument("--max_indels", type=int, default=3,
                        help="Max indels for barcode snapping (default: 3)")
    parser.add_argument("--min_window_score", type=float, default=0.7,
                        help="Minimum fraction of positions matching template "
                             "to accept a window as the barcode region (default: 0.7)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds for subprocess calls (default: 3600 = 1h)")

    args = parser.parse_args()

    global TIMEOUT
    TIMEOUT = args.timeout

    if args.medaka_model.lower() != "none":
        _get_medaka_interface()

    if args.barcode_template:
        print(f"Barcode template snapping: ON")
        print(f"  Template:  {args.barcode_template[:40]}... ({len(args.barcode_template)}bp)")
        print(f"  Max mismatches: {args.max_mismatches}, Max indels: {args.max_indels}")
        print(f"  Min window score: {args.min_window_score}")

    print(f"Timeout: {TIMEOUT}s ({TIMEOUT / 3600:.1f}h)")
    print("Determining consensus...")
    start = time.time()

    determine_consensus(
        threads=args.threads,
        input_fq=args.input_fq,
        medaka_model=args.medaka_model,
        barcode_template=args.barcode_template,
        max_mismatches=args.max_mismatches,
        max_indels=args.max_indels,
        min_window_score=args.min_window_score,
    )

    print(f"Done in {round(time.time() - start, 2)}s")


if __name__ == '__main__':
    cli()