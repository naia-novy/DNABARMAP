import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
from glob import glob
from dnabarmap.utils import import_cupy_numpy
from Bio.SeqRecord import SeqRecord
from os import makedirs, path
import uuid

np = import_cupy_numpy()


def parse_clusters(file_path, min_sequences, barcode_directory):
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
                        save_clusters_to_files(current_cluster, clusters, f'temp/{barcode_directory}/clusters/barcodes/')
                        number_passing += 1
                    current_cluster = last_id
                    clusters = {}  # overwrite clusters
                last_id = line[1:]
            else:
                clusters['>' + last_id] = line
        if len(clusters) >= min_sequences:
            save_clusters_to_files(current_cluster, clusters, f'temp/{barcode_directory}/clusters/barcodes/')
            number_passing += 1

        print(f'Found {number_passing} clusters with >= {min_sequences} sequences.')

def cluster(output_fn, min_sequences, threads, id, c, barcode_directory, **kwargs):
    cluster_out = f'temp/{barcode_directory}/clusters/barcodes/cluster-result_all_seqs.fasta'
    with open(cluster_out, "w") as out_fn:
        cmd = ['mmseqs',
               'easy-cluster',
               '--threads', str(threads),
               '--kmer-per-seq', '1000',
               '--cluster-steps', '3',
               '--cluster-reassign', '1',
               '--max-iterations', '1000',
               '--alignment-mode', '3',
               '--cluster-mode', '0',
               '--min-seq-id', str(id),
               '-c', str(c),
               '-k', '7',
               '-s', '1.0',
               '--similarity-type', '1',
               '--remove-tmp-files', '0',
               '--shuffle', '0',
               '--cov-mode', '1',
               output_fn, f'temp/{barcode_directory}/clusters/barcodes/cluster-result', 'temp']

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
    parse_clusters(cluster_out, min_sequences, barcode_directory)

def save_full_seqs(reoriented_fn, barcode_directory, **kwargs):
    # Read entire FASTQ into memory as a dict: id → SeqRecord
    # This is fast enough for millions of reads.
    print("Indexing full FASTQ…")
    fastq_records = SeqIO.to_dict(SeqIO.parse(reoriented_fn, "fastq"))
    print(f"Loaded {len(fastq_records):,} reads from full FASTQ")

    cluster_fastas = glob(f"temp/{barcode_directory}/clusters/barcodes/cluster_*.fasta")
    makedirs(f"temp/{barcode_directory}/clusters/full_seqs/", exist_ok=True)

    written = 0
    for fasta_path in cluster_fastas:
        # Cluster number is whatever appears at end of filename
        cluster_id = path.basename(fasta_path).split("_")[-1].split(".")[0]
        out_fastq = f"temp/{barcode_directory}/clusters/full_seqs/cluster_{cluster_id}.fastq"

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


# Usage example
def save_clusters_to_files(cluster_id, clusters, output_dir):
    filename = f"{output_dir}/cluster_{cluster_id}.fasta"
    with open(filename, 'w') as f:
        for id, seq in clusters.items():
            f.write(id + '\n')
            f.write(seq + '\n')

