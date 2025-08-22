import subprocess
from glob import glob
from os import makedirs, remove
import time

from utils import import_cupy_numpy

np = import_cupy_numpy()

def determine_consensus(threads, **kwargs):
    makedirs('tmp/consensus/draft/', exist_ok=True)

    # Draft consesnus stage with vsearch
    clusters = glob("tmp/clusters/cluster_*.fastq")
    to_remove = []
    for i, fn in enumerate(clusters):
        if i % 100 == 0:
            sub_dir = f"tmp/consensus/consensus_{i}"
            makedirs(sub_dir, exist_ok=True)
        print(i, fn)
        cluster_id = fn.split('_')[-1].split('.')[0]
        draft_path = f"tmp/consensus/draft/cluster_{cluster_id}_consensus.fastq"
        draft_paf = f"tmp/consensus/draft/cluster_{cluster_id}_consensus.paf"
        consensus_path = f"{sub_dir}/cluster_{cluster_id}_consensus.fasta"

        with open(draft_path, "w") as out_fn:
            cmd = ['abpoa',
                   '-a 0',
                   '-m 1',
                   '-O 4,4',
                   '-Q',
                   '-i',
                   '-s',
                   '-r 5',
                   fn]
            subprocess.run(cmd, stdout=out_fn, stderr=subprocess.DEVNULL, check=True)

        with open(draft_paf, "w") as out_paf:
            cmd = ['minimap2',
                   '-x', 'map-ont',
                   '-t', str(threads),
                   draft_path,
                   fn]
            subprocess.run(cmd, stdout=out_paf, stderr=subprocess.DEVNULL, check=True)

        with open(consensus_path, "w") as out_cons:
            cmd = ['racon',
                   fn,
                   draft_paf,
                   draft_path,
                   '--no-trimming',
                   '-q 5',
                   '-w 2500', # perform poa on majority/all sequence length since they are already clustered
                   '-t', str(threads)]
            subprocess.run(cmd, stdout=out_cons, stderr=subprocess.DEVNULL, check=True)

        to_remove.append(draft_path)
        to_remove.append(draft_paf)

        if len(to_remove) >= 100:
            # time.sleep(3) # allow system to register writing of new files
            for fn in to_remove:
                remove(fn)
            to_remove = []

    if len(to_remove) > 0:
        # time.sleep(5)  # allow system to register writing of new files
        for fn in to_remove:
            remove(fn)

    print(f"Consensus generation complete.")
