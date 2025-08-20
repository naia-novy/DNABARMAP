import subprocess
import tempfile
from os import remove
import gzip

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
        '--qshmm', 'pbsim3_models/QSHMM-ONT-HQ.model',
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
