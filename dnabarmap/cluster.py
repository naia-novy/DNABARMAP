import subprocess
import tempfile
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from glob import glob
from dnabarmap.utils import import_cupy_numpy
from os import makedirs, path, remove
import argparse
import time
from shutil import rmtree

import shutil
from pathlib import Path

np = import_cupy_numpy()


# ═════════════════════════════════════════════════════════════════
# HOMOPOLYMER COMPRESSION FOR BARCODES
# ═════════════════════════════════════════════════════════════════

def hp_compress_simple(seq):
    """Collapse homopolymer runs to single bases: AAACCCGT → ACGT"""
    return re.sub(r'(.)\1+', r'\1', seq)


def create_hp_compressed_fasta(input_fasta, output_fasta):
    """
    Write an HP-compressed version of a barcode FASTA for clustering.
    Keeps the same record IDs so cluster assignments map back to originals.

    Returns:
        n_records: number of records written
        n_changed: number of records where compression changed the sequence
    """
    n_records = 0
    n_changed = 0
    with open(output_fasta, 'w') as out:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            original = str(record.seq).upper()
            compressed = hp_compress_simple(original)
            if compressed != original:
                n_changed += 1
            out.write(f'>{record.id}\n{compressed}\n')
            n_records += 1
    return n_records, n_changed


# ═════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════
# CLUSTER PARSING AND SAVING
# ═════════════════════════════════════════════════════════════════

def parse_cluster_tsv(tsv_path, input_fasta, min_sequences, output_dir):
    """
    Parse MMseqs2 cluster TSV and write cluster FASTA files.

    Uses the ORIGINAL (uncompressed) barcode sequences from input_fasta,
    even though clustering was done on HP-compressed sequences.
    """
    sequences = {}
    for record in SeqIO.parse(input_fasta, 'fasta'):
        sequences[record.id] = str(record.seq)

    clusters_by_rep = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            rep_id, member_id = parts[0], parts[1]
            if rep_id not in clusters_by_rep:
                clusters_by_rep[rep_id] = {}
            if member_id in sequences:
                clusters_by_rep[rep_id][f'>{member_id}'] = sequences[member_id]
            else:
                print(f"WARNING: member {member_id} from cluster TSV not found in input FASTA")

    total_reads = len(sequences)
    number_passing = 0
    reads_in_passing_clusters = 0
    out_path = f'{output_dir}/clusters/barcodes/'
    for rep_id, members in clusters_by_rep.items():
        if len(members) >= min_sequences:
            save_clusters_to_files(rep_id, members, out_path)
            number_passing += 1
            reads_in_passing_clusters += len(members)

    print(f'Found {number_passing} clusters with >= {min_sequences} sequences.')
    print(f'{reads_in_passing_clusters:,} / {total_reads:,} reads retained in passing clusters '
          f'({reads_in_passing_clusters / total_reads * 100:.1f}%).')


def _find_barcode_fasta(output_dir, output_fn=None):
    """
    Locate the barcode FASTA produced by align().
    Checks output_fn first (if provided), then searches output_dir.
    """
    # If run.py passed output_fn, check it directly
    if output_fn and path.exists(output_fn):
        return output_fn

    # Search output_dir recursively for barcode FASTAs
    candidates = sorted(Path(output_dir).rglob('*_barcodes.fasta'))
    candidates = [f for f in candidates if f.name != 'combined_barcodes.fasta'
                  and 'hp_compressed' not in f.name
                  and 'clusters' not in str(f)]

    if len(candidates) == 1:
        return str(candidates[0])
    elif len(candidates) > 1:
        # Multiple batch files — combine them
        combined = f'{output_dir}/combined_barcodes.fasta'
        print(f"  Found {len(candidates)} barcode files, combining...")
        with open(combined, 'wb') as out:
            for f in candidates:
                with open(f, 'rb') as inp:
                    shutil.copyfileobj(inp, out)
        return combined
    else:
        raise FileNotFoundError(
            f"No *_barcodes.fasta files found in {output_dir}. "
            f"Did the align step complete successfully?"
        )


def _find_reoriented_fastq(output_dir, input_fq=None, reoriented_fn=None):
    """
    Locate the reoriented FASTQ produced by align().

    Check order:
      1. Explicit reoriented_fn from kwargs (lives next to input file)
      2. Search inside output_dir
      3. Fall back to original input FASTQ
    """
    # 1. Check the explicit path from run.py (e.g., syndata_reoriented.fastq)
    if reoriented_fn and path.exists(reoriented_fn):
        return reoriented_fn

    # 2. Search output_dir
    candidates = sorted(Path(output_dir).rglob('*_reoriented.fastq'))
    candidates = [f for f in candidates if 'combined_' not in f.name
                  and 'clusters' not in str(f)]

    if len(candidates) == 1:
        return str(candidates[0])
    elif len(candidates) > 1:
        combined = f'{output_dir}/combined_reoriented.fastq'
        print(f"  Found {len(candidates)} reoriented files, combining...")
        with open(combined, 'wb') as out:
            for f in candidates:
                with open(f, 'rb') as inp:
                    shutil.copyfileobj(inp, out)
        return combined

    # 3. Fall back to original input
    if input_fq and path.exists(input_fq):
        print(f"  No reoriented FASTQ found, using original input: {input_fq}")
        return input_fq
    raise FileNotFoundError(
        f"No *_reoriented.fastq found in {output_dir} and no input_fq fallback. "
        f"Did the align step complete successfully?"
    )


def cluster(min_sequences, threads, id, c, output_dir,
            hp_compress=True, **kwargs):
    """
    Cluster barcode sequences using MMseqs2.

    Automatically locates the barcode FASTA and reoriented FASTQ produced
    by the align step, wherever they are in output_dir.

    Parameters:
    -----------
    min_sequences : int
        Minimum cluster size
    threads : int
        Number of threads
    id : float
        Minimum sequence identity for clustering
    c : float
        Minimum coverage for clustering
    output_dir : str
        Output directory
    hp_compress : bool
        If True, HP-compress barcodes before clustering to handle nanopore
        homopolymer errors. Original sequences are preserved in output.
    """

    # ── Find the barcode and reoriented files from align step ────
    barcode_fasta = _find_barcode_fasta(output_dir, kwargs.get('output_fn'))
    reoriented_fastq = _find_reoriented_fastq(output_dir, kwargs.get('input_fq'),
                                              kwargs.get('reoriented_fn'))

    print(f"  Barcode FASTA: {barcode_fasta}")
    print(f"  Reoriented FASTQ: {reoriented_fastq}")

    # Store these so save_full_seqs can find them
    kwargs['_barcode_fasta'] = barcode_fasta
    kwargs['_reoriented_fastq'] = reoriented_fastq

    # ── HP-compress barcodes for clustering ──────────────────────
    if hp_compress:
        compressed_fn = f'{output_dir}/barcodes_hp_compressed.fasta'
        print("  HP-compressing barcodes for clustering...")
        n_records, n_changed = create_hp_compressed_fasta(barcode_fasta, compressed_fn)
        print(f"  {n_records:,} barcodes, {n_changed:,} modified by HP compression "
              f"({100 * n_changed / max(n_records, 1):.1f}%)")
        clustering_input = compressed_fn
    else:
        clustering_input = barcode_fasta

    # ── Run MMseqs2 ──────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(dir=output_dir, prefix='mmseqs_tmp_')
    cluster_prefix = f'{output_dir}/clusters/barcodes/cluster-result'

    try:
        cmd = [
            'mmseqs',
            'easy-cluster',
            '--threads', str(threads),
            '--kmer-per-seq', '100',
            '--cluster-steps', '2',
            '--cluster-reassign', '1',
            '--max-iterations', '10',
            '--alignment-mode', '2',
            '--cluster-mode', '1',
            '--min-seq-id', str(id),
            '-c', str(c),
            '-k', '5',
            '--similarity-type', '1',
            '--remove-tmp-files', '1',
            '--cov-mode', '0',
            clustering_input,
            cluster_prefix,
            tmp_dir,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"mmseqs easy-cluster failed:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

    finally:
        if path.exists(tmp_dir):
            rmtree(tmp_dir, ignore_errors=True)

    # Clean up compressed file
    if hp_compress and path.exists(compressed_fn):
        remove(compressed_fn)

    # ── Parse TSV using ORIGINAL (uncompressed) barcodes ─────────
    tsv_path = f'{cluster_prefix}_cluster.tsv'
    if not path.exists(tsv_path):
        raise FileNotFoundError(
            f"Expected MMseqs2 cluster TSV at {tsv_path}. "
            f"Check mmseqs output files at {cluster_prefix}_*"
        )

    parse_cluster_tsv(tsv_path, barcode_fasta, min_sequences, output_dir)


def save_full_seqs(output_dir, **kwargs):
    """Write full FASTQ records for each cluster using disk-backed indexing."""
    # Find the reoriented FASTQ — use cached path from cluster() if available,
    # otherwise search for it
    reoriented = kwargs.get('_reoriented_fastq')
    if not reoriented or not path.exists(reoriented):
        reoriented = _find_reoriented_fastq(output_dir, kwargs.get('input_fq'),
                                            kwargs.get('reoriented_fn'))

    print(f"Indexing full FASTQ: {reoriented}")
    fastq_records = SeqIO.index(reoriented, "fastq")
    print(f"Indexed {len(fastq_records):,} reads from full FASTQ")

    cluster_fastas = glob(f"{output_dir}/clusters/barcodes/*/cluster_*.fasta")
    makedirs(f"{output_dir}/clusters/full_seqs/", exist_ok=True)

    written = 0
    missing = 0
    for fasta_path in cluster_fastas:
        cluster_id = path.basename(fasta_path).split(".")[0]
        sub_dir = get_output_subdir(cluster_id, output_dir + '/clusters/full_seqs')
        out_fastq = f"{sub_dir}/{cluster_id}.fastq"

        cluster_records = []
        for rec in SeqIO.parse(fasta_path, "fasta"):
            read_id = rec.id
            if read_id not in fastq_records:
                missing += 1
                continue
            cluster_records.append(fastq_records[read_id])

        if cluster_records:
            SeqIO.write(cluster_records, out_fastq, "fastq")
            written += 1

    if missing:
        print(f"WARNING: {missing} reads referenced in clusters were not found in full FASTQ")
    print(f"Wrote {written} clusters with full FASTQ sequences.")

    fastq_records.close()


def get_output_subdir(cluster_id, cluster_dir):
    sub_dir = cluster_dir + '/' + cluster_id[-2:]
    makedirs(sub_dir, exist_ok=True)
    return sub_dir


def save_clusters_to_files(cluster_id, clusters, output_dir):
    sub_dir = get_output_subdir(cluster_id, output_dir)
    filename = f"{sub_dir}/cluster_{cluster_id}.fasta"
    with open(filename, 'w') as f:
        for seq_id, seq in clusters.items():
            f.write(seq_id + '\n')
            f.write(seq + '\n')


def main(**kwargs):
    kwargs['id'] = round(kwargs['id'], 2)
    kwargs['c'] = round(kwargs['c'], 2)

    print('Clustering barcodes...')
    cluster_start_time = time.time()
    cluster(**kwargs)
    save_full_seqs(**kwargs)
    cluster_time = time.time() - cluster_start_time
    print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')


def cli():
    parser = argparse.ArgumentParser()

    # Directories and filenames
    parser.add_argument('--output_dir', type=str, default=None, required=True)

    # Cluster parameters
    parser.add_argument("--id", type=float, default=0.75,
                        help="Minimum sequence identity for clustering (0-1). "
                             "Recommended >0.75, can be reduced for small libraries, deep sequencing, "
                             "or extra long barcodes.")
    parser.add_argument("-c", type=float, default=0.25,
                        help="Minimum coverage for clustering (0-1). Low because we already extracted the correct region")
    parser.add_argument("--min_sequences", type=int, default=10,
                        help="Minimum number of sequences for a valid cluster. "
                             "Aim for at least 3x the expected depth.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for clustering.")

    # HP compression
    parser.add_argument("--no_hp_compress", action="store_true", default=False,
                        help="Disable homopolymer compression before clustering. "
                             "By default, barcodes are HP-compressed so that reads "
                             "differing only in homopolymer run lengths (a common "
                             "nanopore error mode) cluster together correctly.")

    all_args = parser.parse_known_args()
    args = all_args[0]

    # Convert flag to positive bool
    args.hp_compress = not args.no_hp_compress
    del args.no_hp_compress

    # Set up directories and filenames
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'

    # Remove previous iteration
    if path.exists(args.cluster_dir):
        rmtree(args.cluster_dir)
    if path.exists(args.consensus_dir):
        rmtree(args.consensus_dir)

    makedirs(args.cluster_dir + '/barcodes', exist_ok=True)
    makedirs(args.cluster_dir + '/full_seqs', exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)

    main(**vars(args))


if __name__ == '__main__':
    cli()