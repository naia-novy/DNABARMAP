from Bio import SeqIO
from glob import glob
from os import makedirs, remove, path
from pathlib import Path
import subprocess
from Bio.Align import PairwiseAligner


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

def determine_consensus(threads, barcode_directory, **kwargs):
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
            # Use 90th percentile length
            # records = list(SeqIO.parse(fn, "fastq"))
            # lengths = np.array([len(r.seq) for r in records])
            # q90_value = np.quantile(lengths, 0.9)
            #
            # # Find index of sequence closest to the 90th percentile length
            # idx = np.argmin(np.abs(lengths - q90_value))
            # representative = records[idx]

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
                    fn,          # all reads
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

    print(f"Consensus sequence generation complete for {i+1} clusters")
    print(f"Number of failed clusters: {failures}")
