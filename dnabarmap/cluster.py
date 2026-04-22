import subprocess
import tempfile
from collections import OrderedDict
from Bio import SeqIO
from glob import glob
from dnabarmap.utils import get_cluster_shard_dir, import_cupy_numpy
from os import makedirs, path
import argparse
import time
from shutil import rmtree

import shutil
from pathlib import Path

np = import_cupy_numpy()
DEFAULT_FASTQ_WRITE_BUFFER_BYTES = 256 * 1024


def _get_max_open_fastq_handles(default=128, reserve=32):
    """Pick a safe output-handle budget below the process open-file limit."""
    try:
        import resource
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit <= 0:
            return default
        return max(16, min(default, soft_limit - reserve))
    except Exception:
        return default


def _flush_fastq_buffer(handle_state):
    """Write any buffered FASTQ text for one output handle."""
    if not handle_state['parts']:
        return
    handle_state['handle'].write(''.join(handle_state['parts']))
    handle_state['parts'].clear()
    handle_state['chars'] = 0

def summarize_fasta_lengths(input_fasta):
    lengths = [len(record.seq) for record in SeqIO.parse(input_fasta, 'fasta')]
    if not lengths:
        raise ValueError(f'No sequences found in FASTA: {input_fasta}')
    return {
        'count': len(lengths),
        'min': int(min(lengths)),
        'median': int(np.median(np.asarray(lengths))),
        'max': int(max(lengths)),
    }

# ═════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════
# CLUSTER PARSING AND SAVING
# ═════════════════════════════════════════════════════════════════

def parse_cluster_tsv(tsv_path, input_fasta, min_sequences, output_dir):
    """
    Parse MMseqs2 cluster TSV and write cluster FASTA files.
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
    cluster_sizes = sorted(
        ((rep_id, len(members)) for rep_id, members in clusters_by_rep.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    passing_cluster_sizes = [
        (rep_id, size) for rep_id, size in cluster_sizes
        if size >= min_sequences
    ]

    sizes_tsv = f'{output_dir}/clusters/barcodes/cluster_sizes.tsv'
    with open(sizes_tsv, 'w') as handle:
        handle.write('cluster_rep_id\tsize\n')
        for rep_id, size in passing_cluster_sizes:
            handle.write(f'{rep_id}\t{size}\n')

    # print(f'Passing cluster sizes (>= {min_sequences}, descending):')
    # for rep_id, size in passing_cluster_sizes:
    #     print(f'  {rep_id}: {size}')

    for rep_id, members in clusters_by_rep.items():
        if len(members) >= min_sequences:
            save_clusters_to_files(rep_id, members, out_path)
            number_passing += 1
            reads_in_passing_clusters += len(members)

    print(f'Found {number_passing} clusters with >= {min_sequences} sequences.')
    print(f'{reads_in_passing_clusters:,} / {total_reads:,} reads retained in passing clusters '
          f'({reads_in_passing_clusters / total_reads * 100:.1f}%).')
    print(f'Cluster sizes written to {sizes_tsv}')


def _find_barcode_fasta(output_dir, output_fn=None, cluster_input_fn=None):
    """
    Locate the barcode FASTA produced by align().
    Checks an explicit cluster input first (if provided), then the explicit
    path passed from run.py, then searches output_dir.
    """
    if cluster_input_fn:
        if path.exists(cluster_input_fn):
            return cluster_input_fn
        raise FileNotFoundError(
            f"Requested barcode FASTA does not exist: {cluster_input_fn}"
        )

    # If run.py passed output_fn, treat it as authoritative.
    if output_fn:
        if path.exists(output_fn):
            return output_fn
        raise FileNotFoundError(
            f"Expected barcode FASTA was not created: {output_fn}. "
            f"The align step may have failed or written to a different output directory."
        )

    output_dir = output_dir + '/aligned/'

    # Search output_dir recursively for barcode FASTAs
    candidates = sorted(Path(output_dir).rglob('*_barcodes.fasta'))
    candidates = [f for f in candidates if f.name != 'combined_barcodes.fasta'
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
    if reoriented_fn:
        if path.exists(reoriented_fn):
            return reoriented_fn
        raise FileNotFoundError(
            f"Expected reoriented FASTQ was not created: {reoriented_fn}. "
            f"The align step may have failed or written to a different output directory."
        )

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


def cluster(min_sequences, threads, id, c, output_dir, **kwargs):
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
    """

    # ── Find the barcode and reoriented files from align step ────
    barcode_fasta = _find_barcode_fasta(
        output_dir,
        output_fn=kwargs.get('output_fn'),
        cluster_input_fn=kwargs.get('cluster_input_fn'),
    )
    reoriented_fastq = _find_reoriented_fastq(output_dir, kwargs.get('input_fq'),
                                              kwargs.get('reoriented_fn'))

    print(f"  Barcode FASTA: {barcode_fasta}")
    print(f"  Reoriented FASTQ: {reoriented_fastq}")

    barcode_stats = summarize_fasta_lengths(barcode_fasta)
    print('  Barcode length stats: '
          f"min {barcode_stats['min']}, median {barcode_stats['median']}, "
          f"max {barcode_stats['max']}")

    barcode_template = kwargs.get('barcode_template')
    extra = max(0, int(kwargs.get('extra', 0)))
    if barcode_template:
        expected_len = len(barcode_template) + 2 * extra
        allowed_len = expected_len + max(20, expected_len // 2)
        if barcode_stats['median'] > allowed_len:
            raise ValueError(
                'Aligned barcode FASTA looks wrong for clustering: '
                f"median length {barcode_stats['median']} is much larger than the "
                f'expected barcode-window length around {expected_len}. '
                'This usually means align wrote full reads instead of barcode windows. '
                'Rerun align with the current code into a fresh output directory.'
            )

    # Store these so save_full_seqs can find them
    kwargs['_barcode_fasta'] = barcode_fasta
    kwargs['_reoriented_fastq'] = reoriented_fastq

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
            '--cluster-steps', '1',
            '--max-iterations', '10',
            '--alignment-mode', '3',
            '--cluster-mode', '0',
            '--cov-mode', '0',
            '--min-seq-id', str(id),
            '-c', str(c),
            '-k', '10',
            '--gap-open', '1',
            '-s', '3',
            '-e', '0.01',
            '--remove-tmp-files', '1',
            '--adjust-kmer-len', '0',
            '--rescore-mode', '3',
            '--similarity-type', '2',
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

    # ── Parse TSV using the aligned barcode windows ───────────────
    tsv_path = f'{cluster_prefix}_cluster.tsv'
    if not path.exists(tsv_path):
        raise FileNotFoundError(
            f"Expected MMseqs2 cluster TSV at {tsv_path}. "
            f"Check mmseqs output files at {cluster_prefix}_*"
        )

    parse_cluster_tsv(
        tsv_path,
        barcode_fasta,
        min_sequences,
        output_dir,
    )


def save_full_seqs(output_dir, **kwargs):
    """Write full FASTQ records for each cluster with a single FASTQ pass."""
    # Find the reoriented FASTQ — use cached path from cluster() if available,
    # otherwise search for it
    reoriented = kwargs.get('_reoriented_fastq')
    if not reoriented or not path.exists(reoriented):
        reoriented = _find_reoriented_fastq(output_dir, kwargs.get('input_fq'),
                                            kwargs.get('reoriented_fn'))

    cluster_fastas = glob(f"{output_dir}/clusters/barcodes/*/cluster_*.fasta")
    makedirs(f"{output_dir}/clusters/full_seqs/", exist_ok=True)

    read_to_output = {}
    output_paths = {}
    for fasta_path in cluster_fastas:
        cluster_id = path.basename(fasta_path).split(".")[0]
        sub_dir = get_output_subdir(cluster_id, output_dir + '/clusters/full_seqs')
        out_fastq = f"{sub_dir}/{cluster_id}.fastq"
        for rec in SeqIO.parse(fasta_path, "fasta"):
            if rec.id in read_to_output:
                print(f"WARNING: read {rec.id} appears in multiple barcode clusters; keeping first assignment")
                continue
            read_to_output[rec.id] = out_fastq
        output_paths[out_fastq] = cluster_id

    print(f"Streaming full FASTQ once: {reoriented}")
    seen_in_clusters = set()
    written_counts = {out_fastq: 0 for out_fastq in output_paths}
    max_open_handles = kwargs.get('max_open_fastq_handles')
    if max_open_handles is None:
        max_open_handles = _get_max_open_fastq_handles()
    max_open_handles = max(1, int(max_open_handles))
    write_buffer_bytes = kwargs.get('fastq_write_buffer_bytes')
    if write_buffer_bytes is None:
        write_buffer_bytes = DEFAULT_FASTQ_WRITE_BUFFER_BYTES
    write_buffer_bytes = max(1, int(write_buffer_bytes))
    print(f"Using up to {max_open_handles} open cluster FASTQ handles at a time")
    print(f"FASTQ write buffer per handle: {write_buffer_bytes:,} bytes")

    handles = OrderedDict()
    total_reads = 0
    try:
        for record in SeqIO.parse(reoriented, "fastq"):
            total_reads += 1
            out_fastq = read_to_output.get(record.id)
            if out_fastq is None:
                continue
            handle = handles.get(out_fastq)
            if handle is None:
                if len(handles) >= max_open_handles:
                    _, old_state = handles.popitem(last=False)
                    _flush_fastq_buffer(old_state)
                    old_state['handle'].close()
                handle = {
                    'handle': open(out_fastq, 'a', buffering=write_buffer_bytes),
                    'parts': [],
                    'chars': 0,
                }
                handles[out_fastq] = handle
            else:
                handles.move_to_end(out_fastq)
            fastq_text = record.format("fastq")
            handle['parts'].append(fastq_text)
            handle['chars'] += len(fastq_text)
            if handle['chars'] >= write_buffer_bytes:
                _flush_fastq_buffer(handle)
            written_counts[out_fastq] += 1
            seen_in_clusters.add(record.id)
    finally:
        for handle_state in handles.values():
            _flush_fastq_buffer(handle_state)
            handle_state['handle'].close()

    written = sum(1 for count in written_counts.values() if count > 0)
    missing = len(read_to_output) - len(seen_in_clusters)
    print(f"Scanned {total_reads:,} reads from full FASTQ")
    if missing:
        print(f"WARNING: {missing} reads referenced in clusters were not found in full FASTQ")
    print(f"Wrote {written} clusters with full FASTQ sequences.")


def get_output_subdir(cluster_id, cluster_dir):
    return get_cluster_shard_dir(cluster_id, cluster_dir)


def save_clusters_to_files(cluster_id, clusters, output_dir):
    file_stem = f"cluster_{cluster_id}"
    sub_dir = get_output_subdir(file_stem, output_dir)
    filename = f"{sub_dir}/{file_stem}.fasta"
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
    mmseqs_time = time.time() - cluster_start_time
    print(f'Finished MMseqs clustering in {round(mmseqs_time / 60, 1)} minutes')

    if kwargs.get('skip_full_seqs', False):
        cluster_time = time.time() - cluster_start_time
        print('Skipping full cluster FASTQ writing (--skip_full_seqs).')
        print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')
        return

    save_start_time = time.time()
    save_full_seqs(**kwargs)
    save_time = time.time() - save_start_time

    cluster_time = time.time() - cluster_start_time
    print(f'Finished saving full cluster FASTQs in {round(save_time / 60, 1)} minutes')
    print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')


def cli():
    parser = argparse.ArgumentParser(
        description="Cluster an aligned barcode FASTA. If --input_fn is not "
                    "provided, the command searches output_dir/aligned/ for "
                    "the barcode FASTA produced by align or dnabarmap."
    )

    # Directories and filenames
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help="Pipeline output directory containing aligned/ "
                             "from a previous align or dnabarmap run.")
    parser.add_argument('--input_fn', dest='cluster_input_fn', type=str, default=None,
                        help="Optional aligned barcode FASTA to cluster. "
                             "If omitted, cluster searches output_dir/aligned/.")

    # Cluster parameters
    parser.add_argument("--id", type=float, default=0.75,
                        help="MMseqs minimum identity for first-pass barcode "
                             "clustering (0-1). Higher values split more; "
                             "lower values merge more.")
    parser.add_argument("-c", type=float, default=0.5,
                        help="MMseqs minimum coverage for first-pass barcode "
                             "clustering (0-1).")
    parser.add_argument("--min_sequences", type=int, default=10,
                        help="Minimum reads required for a cluster to be kept.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Thread count for MMseqs clustering.")
    parser.add_argument("--max_open_fastq_handles", type=int, default=None,
                        help="Maximum cluster FASTQ files to keep open at once "
                             "when writing clusters/full_seqs. Lower this on "
                             "servers with strict open-file limits.")
    parser.add_argument("--skip_full_seqs", action="store_true",
                        help="Skip writing clusters/full_seqs/*.fastq. Use "
                             "this when you only need barcode-cluster sizes "
                             "or barcode FASTAs and are not running consensus "
                             "yet.")
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            "Unrecognized arguments: "
            + " ".join(unknown)
            + ". The standalone cluster command uses an explicit barcode FASTA "
              "from --input_fn or files already present in output_dir/aligned/."
        )

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
