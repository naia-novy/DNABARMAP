import subprocess
from glob import glob
from os import makedirs
from utils import import_cupy_numpy
np = import_cupy_numpy()

def determine_consensus(threads, **kwargs):
    makedirs('tmp/consensus/draft/', exist_ok=True)

    # Draft consesnus stage with vsearch
    clusters = glob("tmp/clusters/cluster_*.fastq")
    for fn in clusters:
        print(fn)
        cluster_id = fn.split('_')[-1].split('.')[0]
        draft_path = f"tmp/consensus/draft/cluster_{cluster_id}_consensus.fastq"
        draft_paf = f"tmp/consensus/draft/cluster_{cluster_id}_consensus.paf"
        consensus_path = f"tmp/consensus/cluster_{cluster_id}_consensus.fasta"

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
                   draft_path,
                   fn]
            subprocess.run(cmd, stdout=out_paf, stderr=subprocess.DEVNULL, check=True)

        with open(consensus_path, "w") as out_cons:
            cmd = ['racon',
                   fn,
                   draft_paf,
                   draft_path,
                   '--no-trimming',
                   '-q 8',
                   '-w 10000', # perform poa on whole sequences since they are already clustered
                   '-t', str(threads)]
            subprocess.run(cmd, stdout=out_cons, stderr=subprocess.DEVNULL, check=True)

    print(f"Consensus generation complete.")
