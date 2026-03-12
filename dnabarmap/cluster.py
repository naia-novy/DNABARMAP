import subprocess
import tempfile
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from glob import glob
from dnabarmap.utils import import_cupy_numpy
from os import makedirs, path
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


def cluster(input_fn, reoriented_fn, min_sequences, threads, id, c, output_dir,
            hp_compress=True, **kwargs):
    """
    Cluster barcode sequences using MMseqs2.

    Parameters:
    -----------
    input_fn : str
        Path to combined barcode FASTA (original, uncompressed sequences)
    reoriented_fn : str
        Path to combined reoriented FASTQ
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
        homopolymer errors. The original (uncompressed) sequences are preserved
        in the output cluster files. (default=True)
    """

    # Combine reoriented files
    input_files = list(Path(f'{output_dir}/aligned').rglob('*_reoriented.fastq'))
    input_files = [i for i in input_files if 'combined_' not in str(i)]
    with open(reoriented_fn, 'wb') as outfile:
        for fasta in input_files:
            with open(fasta, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

    # Combine barcodes
    input_files = list(Path(f'{output_dir}/aligned').rglob('*_barcodes.fasta'))
    input_files = [i for i in input_files if 'combined_' not in str(i)]
    with open(input_fn, 'wb') as outfile:
        for fasta in input_files:
            with open(fasta, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

    # ── HP-compress barcodes for clustering ──────────────────────
    if hp_compress:
        compressed_fn = input_fn.replace('.fasta', '_hp_compressed.fasta')
        print("HP-compressing barcodes for clustering...")
        n_records, n_changed = create_hp_compressed_fasta(input_fn, compressed_fn)
        print(f"  {n_records:,} barcodes, {n_changed:,} modified by HP compression "
              f"({100 * n_changed / max(n_records, 1):.1f}%)")
        clustering_input = compressed_fn
    else:
        clustering_input = input_fn

    # Use a proper temp directory under output_dir so it gets cleaned up
    tmp_dir = tempfile.mkdtemp(dir=output_dir, prefix='mmseqs_tmp_')
    cluster_prefix = f'{output_dir}/clusters/barcodes/cluster-result'

    try:
        cmd = [
            'mmseqs',
            'easy-cluster',
            '--threads', str(threads),
            '--kmer-per-seq', '25',
            '--cluster-steps', '2',
            '--cluster-reassign', '0',
            '--max-iterations', '10',
            '--alignment-mode', '2',
            '--cluster-mode', '0',
            '--min-seq-id', str(id),
            '-c', str(c),
            '-k', '10',
            '--similarity-type', '1',
            '--remove-tmp-files', '1',
            '--cov-mode', '0',
            clustering_input,  # HP-compressed or original
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
        # Clean up temp dir even if mmseqs fails
        if path.exists(tmp_dir):
            rmtree(tmp_dir, ignore_errors=True)

    # Clean up compressed file
    if hp_compress and path.exists(compressed_fn):
        from os import remove
        remove(compressed_fn)

    # Parse the TSV output — uses ORIGINAL (uncompressed) barcodes
    tsv_path = f'{cluster_prefix}_cluster.tsv'
    if not path.exists(tsv_path):
        raise FileNotFoundError(
            f"Expected MMseqs2 cluster TSV at {tsv_path}. "
            f"Check mmseqs output files at {cluster_prefix}_*"
        )

    parse_cluster_tsv(tsv_path, input_fn, min_sequences, output_dir)


def save_full_seqs(reoriented_fn, output_dir, **kwargs):
    """Write full FASTQ records for each cluster using disk-backed indexing."""
    print("Indexing full FASTQ…")
    # SeqIO.index is disk-backed / lazy — handles millions of reads without blowing RAM
    fastq_records = SeqIO.index(reoriented_fn, "fastq")
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
    parser.add_argument("--id", type=float, default=0.8,
                        help="Minimum sequence identity for clustering (0-1). "
                             "Recommended >0.75, can be reduced for small libraries "
                             "or extra long barcodes.")
    parser.add_argument("-c", type=float, default=0.8,
                        help="Minimum coverage for clustering (0-1).")
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
    args.reoriented_fn = f'{args.output_dir}/aligned/combined_reoriented.fastq'
    args.input_fn = f'{args.output_dir}/aligned/combined_barcodes.fasta'

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