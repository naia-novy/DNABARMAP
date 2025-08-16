import subprocess
from glob import glob
import shutil
from os import makedirs
from Bio import SeqIO

def determine_consensus(medaka_model, threads, **kwargs):
    if medaka_model == 'default':
        true_medaka_model = None
    else:
        true_medaka_model = medaka_model

    makedirs("tmp/consensus/medaka", exist_ok=True)
    makedirs("tmp/consensus/draft", exist_ok=True)
    # Draft consesnus stage with vsearch
    clusters = glob("tmp/clusters/cluster_*.fastq")
    for fn in clusters:
        cluster_id = fn.split('_')[-1].split('.')[0]
        draft_path = f"tmp/consensus/cluster_{cluster_id}_draft.fasta"
        consensus_path = f"tmp/consensus/medaka/cluster_{cluster_id}_consensus.fasta"

        # assumme everything assigned here already belongs in the correct cluster, just want to determine consensus
        cmd = ["vsearch", "--cluster_size", fn,
            "--id", "0.7",
            "--threads", str(threads),
            "--consout", draft_path,
               "--clusterout_sort",
               "--gapopen", "2",
               "--gapext", "0",
               "--mismatch", "-2",
               "--match", "5",
               ]

        subprocess.run(cmd,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # Make file with only first (best) consensus from vsearch
        first_draft = draft_path.replace('.fasta', '_first.fasta')
        with open(draft_path) as handle:
            first_record = next(SeqIO.parse(handle, "fasta"))
            with open(first_draft, "w") as out_handle:
                SeqIO.write(first_record, out_handle, "fasta")

        if medaka_model is not None:
            # Refine stage with Medaka
            model_line = '' if true_medaka_model is None else f"-m {true_medaka_model}"
            cmd = ["medaka_consensus", "-i", fn,
                   "-d", first_draft,
                   "-t", str(threads),
                   "-o", consensus_path,
                   '-f', '-x',
                   model_line]

            subprocess.run(cmd,
                           check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

            final_path = consensus_path.replace("medaka/", "")
            shutil.move(consensus_path + '/consensus.fasta', final_path)
            draft_files = glob('tmp/consensus/*_draft*fasta*')
            for f in draft_files:
                shutil.move(f, f.replace('/consensus/', '/consensus/draft/'))
        else:
            final_path = consensus_path.replace("medaka/", "")
            shutil.move(draft_path, final_path)

        print(fn)

    print(f"Consensus generation complete.")

