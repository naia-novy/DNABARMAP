from Bio import SeqIO
import subprocess, tempfile
from glob import glob
import numpy as np


def determine_consensus(**kwargs):
    for fn in glob("tmp/clusters/cluster_*.fasta"):
        cluster_id = fn.split('_')[-1].split('.')[0]
        outpath = f"tmp/consensus/cluster_{cluster_id}_consensus.fasta"

        recs = list(SeqIO.parse(fn, "fasta"))
        if not recs:
            print("skip empty", fn);
            continue

        with tempfile.TemporaryDirectory() as temp:
            aln = temp + "/aln.fasta"
            subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--retree", "2", fn],
                           stdout=open(aln, "w"), stderr=subprocess.DEVNULL, check=True)
            plurality = max(1, np.ceil(len(recs) * 0.5))
            cons_out = temp + "/cons.fasta"

            plurality = 1
            subprocess.run(["cons", "-sequence", aln, "-outseq", cons_out,
                            "-name", f"consensus_{cluster_id}", "-plurality", str(plurality), "-auto"],
                           check=True)
            final = next(SeqIO.parse(cons_out, "fasta"))
            final.id = f"consensus_{cluster_id}";
            final.description = ""
            SeqIO.write(final, outpath, "fasta")
            print("wrote", outpath)


# def determine_consensus(**kwargs):
#     # For each file of clustered full sequences, determine a single consensus sequence
#     clusters = glob("tmp/clusters/cluster_*.fasta")
#     for fn in clusters:
#         cluster_id = fn.split('_')[-1].split('.')[0]
#         consensus_path = f"tmp/consensus/cluster_{cluster_id}_consensus.fasta"
#
#         # assumme everything assigned here already belongs in the correct cluster, just want to determine consensus
#         cmd = ["vsearch", "--cluster_fast", fn,
#             "--id", "0.8",
#             "--threads", str(kwargs['threads']),
#             "--consout", consensus_path,
#                "--clusterout_sort",
#                "--gapopen", "1",
#                "--gapext", "2",
#                "--mismatch", "-2",
#                ]
#
#         subprocess.run(cmd,
#                        check=True,
#                        stdout=subprocess.DEVNULL,
#                        stderr=subprocess.DEVNULL)
#
#     print(f"Consensus generation complete.")

