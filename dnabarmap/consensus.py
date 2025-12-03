from Bio import SeqIO
from os import remove
from pathlib import Path
from glob import glob
import subprocess
from Bio.Align import PairwiseAligner
import argparse
import time
from os import makedirs, path

from dnabarmap.utils import import_cupy_numpy

np = import_cupy_numpy()


def select_centroid(records, max_comparisons=50):
    """Select sequence with highest average identity to others."""
    if len(records) <= 2:
        return records[0]
    # Subsample if too many
    if len(records) > max_comparisons:
        indices = np.random.choice(len(records), max_comparisons, replace=False)
        subset = [records[i] for i in indices]
    else:
        subset = records
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    # Compute pairwise identity scores
    scores = np.zeros(len(subset))
    for i, rec_i in enumerate(subset):
        for j, rec_j in enumerate(subset):
            if i != j:
                score = aligner.score(str(rec_i.seq), str(rec_j.seq))
                max_len = max(len(rec_i.seq), len(rec_j.seq))
                scores[i] += score / max_len
    best_idx = np.argmax(scores)
    return subset[best_idx]

def get_output_subdir(cluster_id, base_dir, levels=1):
    # Hash the cluster_id to get consistent distribution
    hash_val = hash(str(cluster_id)) & 0xFFFFFFFF  # Ensure positive

    parts = [base_dir]
    for i in range(levels):
        bin_idx = (hash_val >> (16 * i)) & 0xFF  # Extract byte
        parts.append(f"{bin_idx:02x}")

    return "/".join(parts)

def determine_consensus_consecutive(threads, barcode_directory, **kwargs):
    makedirs(f"temp/{barcode_directory}/consensus/draft/", exist_ok=True)

    clusters = glob(f"temp/{barcode_directory}/clusters/full_seqs/cluster_*.fastq")
    failures = 0
    to_remove = []

    for i, fn in enumerate(clusters):
        if i % 100 == 0:
            sub_dir = f"temp/{barcode_directory}/consensus/consensus_{i}"
            makedirs(sub_dir + "/draft", exist_ok=True)
            if i != 0:
                print(f"Consensus sequence generated for {i} clusters")

        cluster_id = fn.split("_")[-1].split(".")[0]
        rep_fastq = f"{sub_dir}/draft/cluster_{cluster_id}_rep.fastq"
        rep_paf = f"{sub_dir}/draft/cluster_{cluster_id}_rep.paf"
        consensus_path = f"{sub_dir}/cluster_{cluster_id}_consensus.fasta"

        # Find the longest sequence (or median-length) as representative
        try:
            records = list(SeqIO.parse(fn, "fastq"))
            representative = select_centroid(records, 10)

            # Write representative to file
            with open(rep_fastq, "w") as out_f:
                SeqIO.write(representative, out_f, "fastq")

        except Exception as e:
            print(f"[Warning] Failed to extract representative for cluster {cluster_id}: {e}")
            failures += 1
            continue

        # Generate overlaps of representative vs all reads
        try:
            Path(rep_paf).parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "minimap2",
                "-x", "map-ont",
                "-t", str(threads),
                rep_fastq,
                fn
            ]
            with open(rep_paf, "w") as out:
                subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            print(f"[Warning] minimap2 failed for cluster {cluster_id}")
            failures += 1
            continue

        # Racon polishing
        try:
            with open(consensus_path, "w") as out_cons:
                cmd = [
                    "racon",
                    fn,  # all reads
                    rep_paf,  # overlaps
                    rep_fastq,  # representative sequence
                    # "--no-trimming",
                    "-q", "8",
                    "-w", "200",
                    "-t", str(threads),
                ]
                subprocess.run(cmd, stdout=out_cons, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"[Warning] racon failed for cluster {cluster_id}")
            failures += 1
            continue

        to_remove.extend([rep_fastq, rep_paf])

        # Cleanup every 100 clusters
        if (i + 1) % 100 == 0:
            for f in to_remove:
                if path.exists(f):
                    remove(f)
            to_remove = []

    # Final cleanup
    for f in to_remove:
        if path.exists(f):
            remove(f)

    print(f"Consensus sequence generation complete for {i + 1} clusters")
    print(f"Number of failed clusters: {failures}")

def determine_consensus(threads, input_fq, **kwargs):
    cluster_id = input_fq.split("-")[-1].split(".")[0]
    consensus_dir = input_fq.split('clusters')[0] + '/consensus/'
    sub_dir = get_output_subdir(cluster_id, consensus_dir)
    draft_dir = sub_dir + '/draft/'

    makedirs(sub_dir, exist_ok=True)
    makedirs(draft_dir, exist_ok=True)

    rep_fastq = f"{draft_dir}/cluster_{cluster_id}_rep.fastq"
    rep_paf = f"{draft_dir}/cluster_{cluster_id}_rep.paf"
    consensus_path = f"{sub_dir}/cluster_{cluster_id}_consensus.fasta"

    # Find the representative sequence
    try:
        records = list(SeqIO.parse(input_fq, "fastq"))
        representative = select_centroid(records, 10)

        # Write representative to file
        with open(rep_fastq, "w") as out_f:
            SeqIO.write(representative, out_f, "fastq")

    except Exception as e:
        print(f"[Warning] Failed to extract representative for cluster {cluster_id}: {e}")
        exit()

    # Generate overlaps of representative vs all reads
    try:
        Path(rep_paf).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "minimap2",
            "-x", "map-ont",
            "-t", str(threads),
            rep_fastq,
            input_fq
        ]
        with open(rep_paf, "w") as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        print(f"[Warning] minimap2 failed for cluster {cluster_id}")
        exit()

    # Racon polishing
    try:
        with open(consensus_path, "w") as out_cons:
            cmd = [
                "racon",
                input_fq,          # all reads
                rep_paf,     # overlaps
                rep_fastq,   # representative sequence
                # "--no-trimming",
                "-q", "8",
                "-w", "200",
                "-t", str(threads),
            ]
            subprocess.run(cmd, stdout=out_cons, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"[Warning] racon failed for cluster {cluster_id}")
        exit()

    # Cleanup
    for f in [rep_fastq, rep_paf]:
        if path.exists(f):
            remove(f)

    print(f"Consensus sequence generation complete for cluster id {cluster_id}")


def cli():
    parser = argparse.ArgumentParser()

    # Directories and filenaemes
    parser.add_argument('--input_fq', type=str, default=None, required=True,
                        help='Input full sequence fastq clusters')

    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for clustering")

    all_args = parser.parse_known_args()
    args = all_args[0]

    # Determine consensus seqeunces for clusters using minimap2 and racon
    print('Determining consensus sequences...')
    consensus_start_time = time.time()
    determine_consensus(**vars(args))
    consensus_time = time.time() - consensus_start_time
    # print(f'Finished determining consensus sequences in {round(consensus_time / 60, 1)} minutes\n')


if __name__ == '__main__':
    cli()