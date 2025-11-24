import subprocess
from glob import glob
from os import makedirs, remove, path
from Bio import SeqIO

from dnabarmap.utils import import_cupy_numpy

np = import_cupy_numpy()
#
# def determine_consensus(threads, barcode_directory, **kwargs):
#     makedirs('temp/{barcode_directory}/consensus/draft/', exist_ok=True)
#
#     # Draft consesnus stage with vsearch
#     clusters = glob(f"temp/{barcode_directory}/clusters/full_seqs/cluster_*.fastq")
#     to_remove = []
#     for i, fn in enumerate(clusters):
#         if i % 100 == 0:
#             sub_dir = f"temp/{barcode_directory}/consensus/consensus_{i}"
#             makedirs(sub_dir+'/draft', exist_ok=True)
#
#             if i != 0:
#                 print(f"Consensus sequence generated for {i} clusters")
#         cluster_id = fn.split('_')[-1].split('.')[0]
#         draft_path = f"{sub_dir}/draft/cluster_{cluster_id}_consensus.fastq"
#         draft_paf = f"{sub_dir}/draft/cluster_{cluster_id}_consensus.paf"
#         consensus_path = f"{sub_dir}/cluster_{cluster_id}_consensus.fasta"
#
#         with open(draft_path, "w") as out_fn:
#             cmd = ['abpoa',
#                    '-m 1',
#                    #'-O 5,5',
#                    #'-E 2,2',
#                    '-Q',
#                    '-r 5',
#                    #'-a', '1',
#                    fn]
#         #     subprocess.run(cmd, stdout=out_fn, stderr=subprocess.DEVNULL, check=True)
#             result = subprocess.run(
#                 cmd,
#                 stdout=out_fn,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )
#             if result.returncode != 0:
#                 print(f"abpoa failed on {fn}:\n{result.stderr}")
#                 raise subprocess.CalledProcessError(result.returncode, cmd)
#
#         with open(draft_paf, "w") as out_paf:
#             cmd = ['minimap2',
#                    '-x', 'map-ont',
#                    '-t', str(threads),
#                    draft_path,
#                    fn]
#             subprocess.run(cmd, stdout=out_paf, stderr=subprocess.DEVNULL, check=True)
#
#         with open(consensus_path, "w") as out_cons:
#             cmd = ['racon',
#                    fn,
#                    draft_paf,
#                    draft_path,
#                    '--no-trimming',
#                    '-q 10',
#                    '-w 2000', # perform poa on majority/all sequence length since they are already clustered
#                    '-t', str(threads)]
#             subprocess.run(cmd, stdout=out_cons, stderr=subprocess.DEVNULL, check=True)
#
#         to_remove.append(draft_path)
#         to_remove.append(draft_paf)
#
#         if len(to_remove) >= 100:
#             # time.sleep(3) # allow system to register writing of new files
#             for fn in to_remove:
#                 remove(fn)
#             to_remove = []
#
#     if len(to_remove) > 0:
#         # time.sleep(5)  # allow system to register writing of new files
#         for fn in to_remove:
#             remove(fn)
#
#     print(f"Consensus sequence generatation complete for {i+1} clusters")
from glob import glob
from os import makedirs, remove, path
from pathlib import Path
import subprocess

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

        # Copy first sequence as representative
        try:
            with open(fn, "r") as in_f, open(rep_fastq, "w") as out_f:
                for _ in range(4):
                    out_f.write(in_f.readline())
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
                    "--no-trimming",
                    "-q", "7",
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
