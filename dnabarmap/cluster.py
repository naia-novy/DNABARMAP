import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
from glob import glob
from dnabarmap.utils import import_cupy_numpy
from Bio.SeqRecord import SeqRecord

np = import_cupy_numpy()


def parse_clusters(file_path, min_sequences):
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
                        save_clusters_to_files(current_cluster, clusters, 'temp/clusters/barcodes/')
                        number_passing += 1
                    current_cluster = last_id
                    clusters = {}  # overwrite clusters
                last_id = line[1:]
            else:
                clusters['>' + last_id] = line
        if len(clusters) >= min_sequences:
            save_clusters_to_files(current_cluster, clusters, 'temp/clusters/barcodes/')
            number_passing += 1

        print(f'Found {number_passing} clusters with >= {min_sequences} sequences.')


def cluster(output_fn, min_sequences, threads, id, c, **kwargs):
    cluster_out = 'temp/clusters/barcodes/cluster-result_all_seqs.fasta'
    with open(cluster_out, "w") as out_fn:
        cmd = ['mmseqs',
               'easy-cluster',
               '--threads', str(threads),
               '--kmer-per-seq', '1000',
               '--cluster-steps', '5',
               '--max-iterations', '1000',
               '--alignment-mode', '3',
               '--cluster-mode', '1',
               '--min-seq-id', str(id),
               '-c', str(c),
               '-k', '3',
               '--similarity-type', '1',
               '--remove-tmp-files', '0',
               output_fn, 'temp/clusters/barcodes/cluster-result', 'temp']

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
    parse_clusters(cluster_out, min_sequences)

def save_full_seqs(reoriented_fn, **kwargs):
    cluster_map = defaultdict(list)

    clusters = glob("temp/clusters/barcodes/cluster_*.fasta")
    for i, cluster in enumerate(clusters):
        cluster_id = cluster.split('_')[-1].split('.')[0]
        with open(cluster) as f:
            for L in f.readlines():
                if L.startswith('>'):
                    id = L[1:].strip().split()[0]
                    cluster_map[cluster_id].append(id)


    pos_index = build_position_index(reoriented_fn)

    written_clusters = 0
    for cluster, idxs in cluster_map.items():
        with open(f"temp/clusters/full_seqs/cluster_{cluster}.fastq", "w") as f:
            for idx in idxs:
                seq = get_sequence_by_position(reoriented_fn, pos_index[int(idx)])
                SeqIO.write(seq, f, "fastq")
            written_clusters += 1

    print(f"Wrote {written_clusters} clusters to full seq fastq files.")


def build_position_index(fastq_file):
    header_to_position = {}
    pos = 0
    with open(fastq_file, 'r') as f:
        while True:
            char_pos = f.tell()  # Get current file position
            line = f.readline()
            if not line:
                break

            if line.startswith('@'):

                header_id = line[1:].strip().split()[0]
                header_to_position[pos] = char_pos

                # Skip the next 3 lines (sequence, +, quality)
                f.readline()  # sequence
                f.readline()  # +
                f.readline()  # quality
                pos += 1

    print(f"Index complete: {len(header_to_position):,} positions stored")
    return header_to_position


def get_sequence_by_position(fastq_file, position):
    with open(fastq_file, 'r') as f:
        f.seek(position)
        header = f.readline().strip()[1:]
        sequence = f.readline().strip()
        plus = f.readline().strip()
        quality = f.readline().strip()

        record = SeqRecord(
            Seq(sequence),
            id=header,
            description="",
            letter_annotations={'phred_quality': [ord(c) - 33 for c in quality]})
        return record


# Usage example
def save_clusters_to_files(cluster_id, clusters, output_dir):
    filename = f"{output_dir}/cluster_{cluster_id}.fasta"
    with open(filename, 'w') as f:
        for id, seq in clusters.items():
            f.write(id + '\n')
            f.write(seq + '\n')

