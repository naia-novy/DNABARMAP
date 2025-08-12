from os import remove, path
import subprocess
from Bio import SeqIO
from glob import glob

def determine_consensus(**kwargs):
    # For each file of clustered full sequences, determine a single consensus sequence
    clusters = glob("tmp/clusters/cluster_*.fasta")
    for fn in clusters:
        cluster_id = fn.split('_')[-1].split('.')[0]

        # Read the cluster file (these should contain sequence headers from original reads)
        cluster_records = list(SeqIO.parse(fn, "fasta"))
        read_ids = [rec.id.split(';')[0] for rec in cluster_records]  # strip any annotations like ;size=

        ref_fasta = f"tmp/clusters/cluster_{cluster_id}_ref.fasta"
        reads_fasta = f"tmp/clusters/cluster_{cluster_id}_read.fasta"
        paf_path = f"tmp/clusters/{cluster_id}.paf"
        consensus_path = f"tmp/consensus/cluster_{cluster_id}_consensus.fasta"

        ref_record = cluster_records[0]
        other_records = cluster_records[1:]

        SeqIO.write(ref_record, ref_fasta, "fasta")
        SeqIO.write(other_records, reads_fasta, "fasta")

        # Align with minimap2
        minimap2_cmd = [
            "minimap2", "-x", "map-ont", ref_fasta, reads_fasta
        ]
        with open(paf_path, "w") as paf_file:
            subprocess.run(minimap2_cmd, stdout=paf_file, stderr=subprocess.DEVNULL, check=True)

        if path.getsize(paf_path) == 0:
            print(f"No overlaps for cluster {cluster_id}; skipping racon.")
            continue

        # Consensus with racon
        racon_cmd = ["racon", reads_fasta, paf_path, ref_fasta]
        result = subprocess.run(
            racon_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
        )

        if result.stdout.strip():
            with open(consensus_path, "w") as f:
                f.write(result.stdout)

            # Optional fix header
            record = next(SeqIO.parse(consensus_path, "fasta"))
            record.id = cluster_id + ":" + ",".join(read_ids)
            record.description = ""
            SeqIO.write(record, consensus_path, "fasta")

        # Cleanup
        remove(ref_fasta)
        remove(reads_fasta)
        remove(paf_path)

    print(f"Consensus generation complete.")

