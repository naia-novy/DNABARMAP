import subprocess
from Bio import SeqIO
from glob import glob
from dnabarmap.utils import import_cupy_numpy
from os import makedirs, path
import argparse
import time
from shutil import rmtree

import shutil
from pathlib import Path

np = import_cupy_numpy()


def parse_clusters(file_path, min_sequences, output_dir):
    clusters = {}
    current_cluster = None
    cluster_id, last_id = None, None
    number_passing = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if it's a cluster representative (>n >n sequence)
            if line.startswith('>'):
                if last_id is None:  # first observation
                    last_id = line[1:]
                    current_cluster = last_id
                elif last_id == line[1:]:  # cluster representative
                    # save previous clusters
                    if len(clusters) >= min_sequences:
                        save_clusters_to_files(current_cluster, clusters, f'{output_dir}/clusters/barcodes/')
                        number_passing += 1
                    current_cluster = last_id
                    clusters = {}  # overwrite clusters
                last_id = line[1:]
            else:
                clusters['>' + last_id] = line
        if len(clusters) >= min_sequences:
            save_clusters_to_files(current_cluster, clusters, f'{output_dir}/clusters/barcodes/')
            number_passing += 1

        print(f'Found {number_passing} clusters with >= {min_sequences} sequences.')


def cluster(input_fn, reoriented_fn, min_sequences, threads, id, c, output_dir, **kwargs):

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


    cluster_out = f'{output_dir}/clusters/barcodes/cluster-result_all_seqs.fasta'
    with open(cluster_out, "w") as out_fn:
        cmd = ['mmseqs',
               'easy-cluster',
               '--threads', str(threads),
               '--kmer-per-seq', '1000',
               '--cluster-steps', '5',
               '--cluster-reassign', '1',
               '--max-iterations', '2000',
               '--alignment-mode', '3',
               '--cluster-mode', '2',
               '--min-seq-id', str(id),
               '-c', str(c),
               '-k', '7',
               '-s', '7.0',
               '--similarity-type', '1',
               '--remove-tmp-files', '1',
               '--shuffle', '0',
               '--cov-mode', '1',
               input_fn, f'{output_dir}/clusters/barcodes/cluster-result', 'temp']

        result = subprocess.run(
            cmd,
            stdout=out_fn,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"mmseqs failed on {out_fn}:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

    # Parse the clusters
    parse_clusters(cluster_out, min_sequences, output_dir)

def save_full_seqs(reoriented_fn, output_dir, **kwargs):
    # Read entire FASTQ into memory as a dict: id → SeqRecord
    # This is fast enough for millions of reads.
    print("Indexing full FASTQ…")
    fastq_records = SeqIO.to_dict(SeqIO.parse(reoriented_fn, "fastq"))
    print(f"Loaded {len(fastq_records):,} reads from full FASTQ")

    cluster_fastas = glob(f"{output_dir}/clusters/barcodes/*/cluster_*.fasta")
    makedirs(f"{output_dir}/clusters/full_seqs/", exist_ok=True)

    written = 0
    for fasta_path in cluster_fastas:
        # Cluster number is whatever appears at end of filename
        cluster_id = path.basename(fasta_path).split(".")[0]
        sub_dir = get_output_subdir(cluster_id, output_dir+'/clusters/full_seqs')
        out_fastq = f"{sub_dir}/{cluster_id}.fastq"
        # out_fastq = f"{output_dir}/clusters/full_seqs/{cluster_id}.fastq"

        cluster_records = []
        for rec in SeqIO.parse(fasta_path, "fasta"):
            read_id = rec.id

            if read_id not in fastq_records:
                print(f"WARNING: read {read_id} not found in full FASTQ")
                continue

            cluster_records.append(fastq_records[read_id])

        SeqIO.write(cluster_records, out_fastq, "fastq")
        written += 1

    print(f"Wrote {written} clusters with full FASTQ sequences.")


def get_output_subdir(cluster_id, cluster_dir):
    sub_dir = cluster_dir + '/' + cluster_id[-2:]
    makedirs(sub_dir, exist_ok=True)
    return sub_dir


def save_clusters_to_files(cluster_id, clusters, output_dir):
    sub_dir = get_output_subdir(cluster_id, output_dir)
    filename = f"{sub_dir}/cluster_{cluster_id}.fasta"
    with open(filename, 'w') as f:
        for id, seq in clusters.items():
            f.write(id + '\n')
            f.write(seq + '\n')


def main(**kwargs):
    kwargs['id'] = round(kwargs['id'], 2)
    kwargs['c'] = 0.8 # round(kwargs['id'], 2)

    # Cluster aligned barcodes using vsearch
    print('Clustering barcodes...')
    cluster_start_time = time.time()
    cluster(**kwargs)
    save_full_seqs(**kwargs)
    cluster_time = time.time() - cluster_start_time
    print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')


def cli():
    parser = argparse.ArgumentParser()

    # Directories and filenaemes
    parser.add_argument('--output_dir', type=str, default=None, required=True)
    # cluster parameters
    parser.add_argument("--id", type=float, default=0.9, help="Value between 0 and 1 for "
                                                                           "minimum identify between barcodes for clustering."
                                                                           "Reccomended >0.75, but can be reduced for small "
                                                                            "libraries or extra long barcodes")
    parser.add_argument("--min_sequences", type=int, default=10,
                        help="Minimum num_sequences for cluster to be valid >= /"
                             "aim for at least 3x the expected depth")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for clustering")

    all_args = parser.parse_known_args()
    args = all_args[0]

    # Set up directories and filenames
    # args.barcode_directory = 'barcode_' + args.input_fn.split('/barcode')[-1].split('/')[0].split('_')[0]
    # args.barcode_directory = 'sample' if args.barcode_directory == '' else args.barcode_directory
    # args.output_dir = f'temp/{args.barcode_directory}/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'
    args.reoriented_fn = f'{args.output_dir}/aligned/combined_reoriented.fastq'
    args.input_fn = f'{args.output_dir}/aligned/combined_barcodes.fasta'

    # remove previous iteration
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