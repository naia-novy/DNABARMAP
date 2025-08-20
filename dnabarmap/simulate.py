import subprocess
import tempfile
from os import remove
import mappy as mp
import pandas as pd
import random
import gzip


def simulate_nanopore(sequence):
    # Create a temporary FASTA file for the sequence
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as tmp_fasta:
        fasta_filename = tmp_fasta.name
        # Write the sequence to the FASTA file
        tmp_fasta.write(f">temp_sequence\n{sequence}\n")

    fastq_filename = fasta_filename.replace('.fasta', '.fastq')
    # Paths
    pbsim_dir = 'pbsim3'
    random_seed = random.randint(0, 2**16 - 1)  # Generate a random seed

    # PBSIM command
    pbsim_command = [
        '/usr/local/bin/pbsim',
        '--strategy', 'templ',
        '--method', 'qshmm',
        '--qshmm', 'pbsim3-master/data/QSHMM-ONT.model',  # 'pbsim3-master/data/QSHMM-ONT.model'
        '--template', fasta_filename,
        '--prefix', fasta_filename.replace('.fasta', ''),
        '--depth', '1',
        '--seed', str(random_seed)
    ]

    # Run PBSIM and capture output
    result = subprocess.run(
        pbsim_command,
        capture_output=True,
        text=True,
        cwd=pbsim_dir,
        check=True)

    # Read the simulated FASTQ output
    with open(fastq_filename, 'r') as fastq_file:
        lines = fastq_file.readlines()
        simulated_sequence = lines[1][:-1]
        simulated_quality = lines[3][:-1]

    # Clean up the temporary FASTA file
    remove(fasta_filename)
    remove(fastq_filename)

    return simulated_sequence, simulated_quality


def save_seqs_to_csv(input_fasta_fn, first_n=80, align=False):
    csv_fn = input_fasta_fn.replace('.fasta', '.csv')
    df = pd.read_csv(csv_fn)

    sequences = []
    qualities = []
    for name, seq, qual in mp.fastx_read(input_fasta_fn.replace('.fasta', '.fastq')):
        sequences.append(seq[:first_n])
        qualities.append(qual[:first_n])

    df.loc[:, 'sim_seq'] = sequences
    df.loc[:, 'sim_qual'] = qualities

    df.to_csv(csv_fn, index=False)
    print(df)
    print(df.columns)

def simulate_many(sequences):
    # Create a temporary FASTA file for the sequence
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as tmp_fasta:
        fasta_filename = tmp_fasta.name
        for idx, sequence in enumerate(sequences):
            # Write the sequence to the FASTA file
            tmp_fasta.write(f">{idx}\n")
            tmp_fasta.write(f"{sequence}\n")

    prefix = fasta_filename.replace('.fasta', '')
    pbsim_command = [
        'pbsim',
        '--strategy', 'templ',
        '--method', 'qshmm',
        '--qshmm', 'data/QSHMM-ONT-HQ.model',
        '--template', fasta_filename,
        '--prefix', prefix,
        '--depth', '1'
    ]
    subprocess.run(pbsim_command, capture_output=True, cwd='./', text=True, check=True)

    fastq_filename = f"{prefix}.fq.gz"  # correct expected filename

    # Read the simulated FASTQ output
    with gzip.open(fastq_filename, 'rt') as fastq_file:  # 'rt' = read text mode
        lines = fastq_file.readlines()
        sequences = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 1) % 4) == 0]
        qualities = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 3) % 4) == 0]

    # Clean up the temporary FASTA file
    remove(fasta_filename)
    remove(fastq_filename)

    return sequences, qualities
