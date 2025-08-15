import subprocess
from os import makedirs, path
from Bio import SeqIO
from collections import defaultdict
from dnabarmap.utils import import_cupy_numpy
np = import_cupy_numpy()

def run_vsearch(
    output_fn,
    cluster_dir,
    threads=16,
    cluster_iterations=5,
    **kwargs):
    makedirs(cluster_dir, exist_ok=True)
    # Use vsearch clustering iterativly to cluster sequences by similarity
    # Do not allow indels since this was already approximated in the alignment step

    # Step 1: First clustering round
    input_fasta = output_fn
    adj_cluster_iterations = cluster_iterations - 1
    values = list(np.linspace(0.97, 0.85, adj_cluster_iterations)) + [0.97]
    for i in range(cluster_iterations):
        out = path.join(cluster_dir, f"consensus_r{i+1}.fasta")
        uc = path.join(cluster_dir, f"clusters_r{i+1}.uc")
        id = values[i]

        cmd = [
            "vsearch", "--cluster_size", input_fasta,
            "--id", str(id),
            "--threads", str(threads),
            "--consout", out,
            "--uc", uc,
            "--sizeout",
            "--sizein",
            "--clusterout_sort",
            "--cons_truncate",
            # "--gapopen", "2",
            # "--gapext", "4",
            # "--mismatch", "-4",
        ]
        print(f"Running clustering iteration {i+1}")
        subprocess.run(cmd,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        input_fasta = out

    # Final mapping
    final_uc = path.join(cluster_dir, "clustered_barcodes.uc")
    cmd = [
        "vsearch", "--usearch_global", input_fasta,
        "--db", out,
        "--id", str(id),
        "--threads", str(threads),
        "--uc", final_uc,
        "--sizein",
        # "--gapopen", "2",
        # "--gapext", "4",
        # "--mismatch", "-4",
    ]
    subprocess.run(cmd,
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def save_full_seqs(fasta_fn, min_sequences, cluster_iterations, seq_limit_for_debugging=None, **kwargs):
    # Using cluster information, save all full seqs for a given cluster to a file
    if seq_limit_for_debugging is None:
        seq_limit_for_debugging = np.inf
    expansions = {}
    approved_clusters = set()
    for i in range(1, cluster_iterations+1):
        observed = set()
        expand = defaultdict()
        with open(f"tmp/clusters/clusters_r{i}.uc") as uc:
            for L in uc:
                if L[0] not in ("S", "H"):
                    continue
                parts = L.strip().split("\t")
                label, idx = parts[0], parts[-2].split(';')[0].split('=')[-1]
                if idx in observed:
                    continue

                expanded_id = idx if label == "S" else parts[-1].split(';')[0].split('=')[-1]
                expand[idx] = expanded_id
                observed.add(idx)

                if i == cluster_iterations:
                    if label == 'S':
                        size = int(parts[-2].split('=')[-1])
                    else:
                        size = int(parts[-1].split('=')[-1])

                    if size >= min_sequences:
                        approved_clusters.add(idx)

        expansions[i] = expand

    print(f"Found {len(approved_clusters)} clusters with >= {min_sequences} sequences.")

    # Group full sequences for clustering
    clustered_sequences = defaultdict(list)
    with open(fasta_fn) as handle:
        for seq_idx, record in enumerate(SeqIO.parse(handle, "fasta")):
            if seq_idx >= seq_limit_for_debugging:
                break
            id_ = record.id
            cluster_id = resolve_final_cluster(id_, expansions)
            if cluster_id not in approved_clusters:
                continue

            clustered_sequences[cluster_id].append(record)

    for cluster, seqs in clustered_sequences.items():
        with open(f"tmp/clusters/cluster_{cluster}.fasta", "w") as f:
            for seq in seqs:
                SeqIO.write(seq, f, "fasta")

def resolve_final_cluster(seq_id, expansion_maps):
    # Iterate over expansions from low to high. When reading from full fasta, use index to map through expansions and
    # determine cluster, then store in this cluster
    current = seq_id
    for i in range(1, len(expansion_maps) + 1):
        current = expansion_maps[i][str(current)]
    return current

