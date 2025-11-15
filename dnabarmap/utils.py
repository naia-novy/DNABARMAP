import pandas as pd
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import gzip

def import_cupy_numpy(print_note=False):
    gpu_available = False
    try:
        import cupy as np
        # Test if GPU is available by querying device count
        device_count = np.cuda.runtime.getDeviceCount()
        if device_count == 0:
            print("No GPU devices found")
        else:
            if print_note:
                print(f"{device_count} GPU devices found")
            gpu_available = True
    except Exception:
        import numpy as np

    if print_note:
        print(f"Using {'CuPy (GPU)' if gpu_available else 'NumPy (CPU)'} as backend")

    return np

np = import_cupy_numpy(print_note=True)

AAs = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Degenerate nucleotide map
degenerate_map = {
    'A': {'A'}, 'T': {'T'}, 'C': {'C'}, 'G': {'G'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
    'K': {'G', 'T'}, 'M': {'A', 'C'}, 'B': {'C', 'G', 'T'},
    'D': {'A', 'G', 'T'}, 'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
    'N': {'A', 'T', 'C', 'G'}  # Fully degenerate base
}

nuc_dict = {'A': ['A'],
            'T': ['T'],
            'C': ['C'],
            'G': ['G'],
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            'M': ['A', 'C'],
            'K': ['G', 'T'],
            'S': ['C', 'G'],
            'W': ['A', 'T'],
            'H': ['A', 'C', 'T'],
            'B': ['C', 'G', 'T'],
            'V': ['A', 'C', 'G'],
            'D': ['A', 'G', 'T'],
            'N': ['A', 'T', 'C', 'G'],
            }

# Define degenerate base mapping
degenerate_base_mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'R': [0.5, 0, 0.5, 0],  # A or G
    'Y': [0, 0.5, 0, 0.5],  # C or T
    'S': [0, 0.5, 0.5, 0],  # C or G
    'W': [0.5, 0, 0, 0.5],  # A or T
    'K': [0, 0, 0.5, 0.5],  # G or T
    'M': [0.5, 0.5, 0, 0],  # A or C
    'B': [0, 0.333, 0.333, 0.333],  # C, G, or T
    'D': [0.333, 0, 0.333, 0.333],  # A, G, or T
    'H': [0.333, 0.333, 0, 0.333],  # A, C, or T
    'V': [0.333, 0.333, 0.333, 0],  # A, C, or G
    'N': [0.25, 0.25, 0.25, 0.25],  # A, C, G, or T (any base)
    '-': [0,0,0,0]
}

# Define degenerate base mapping
hot_degenerate_base_mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'R': [1, 0, 1, 0],  # A or G
    'Y': [0, 1, 0, 1],  # C or T
    'S': [0, 1, 1, 0],  # C or G
    'W': [1, 0, 0, 1],  # A or T
    'K': [0, 0, 1, 1],  # G or T
    'M': [1, 1, 0, 0],  # A or C
    'B': [0, 1, 1, 1],  # C, G, or T
    'D': [1, 0, 1, 1],  # A, G, or T
    'H': [1, 1, 0, 1],  # A, C, or T
    'V': [1, 1 ,1, 0],  # A, C, or G
    'N': [1, 1, 1, 1],  # A, C, G, or T (any base)
    '-': [6, 6, 6, 6]
}

# Define a mapping for degenerate bases
degenerate_nucleotide_mapping = {
    (1, 0, 0, 0): 'A',
    (0, 1, 0, 0): 'C',
    (0, 0, 1, 0): 'G',
    (0, 0, 0, 1): 'T',
    (1, 0, 1, 0): 'R',  # A or G
    (0, 1, 0, 1): 'Y',  # C or T
    (0, 1, 1, 0): 'S',  # C or G
    (1, 0, 0, 1): 'W',  # A or T
    (0, 0, 1, 1): 'K',  # G or T
    (1, 1, 0, 0): 'M',  # A or C
    (0, 1, 1, 1): 'B',  # C, G, or T
    (1, 0, 1, 1): 'D',  # A, G, or T
    (1, 1, 0, 1): 'H',  # A, C, or T
    (1, 1, 1, 0): 'V',  # A, C, or G
    (1, 1, 1, 1): 'N',
    (6, 6, 6, 6): '-' # A, C, G, or T (any base)
}

int_to_degenerate = {1: ['A', 'T', 'C', 'G'],
                     2: ['R', 'Y', 'M', 'K', 'S', 'W'],
                     3: ['H', 'B', 'D', 'V'],
                     4: ['N']}

degenerate_complement = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
    'R': 'Y',  # A or G => T or C
    'Y': 'R',  # C or T => G or A
    'M': 'K',  # A or C => T or G
    'K': 'M',  # G or T => C or A
    'S': 'S',  # C or G => G or C (symmetric)
    'W': 'W',  # A or T => T or A (symmetric)
    'H': 'D',  # A, C, T => T, G, A
    'B': 'V',  # C, G, T => G, C, A
    'V': 'B',  # A, C, G => T, G, C
    'D': 'H',  # A, G, T => T, C, A
    'N': 'N',  # any base => any base
}


# Precompute the degenerate base mapping as a NumPy array
degenerate_base_array = {
    base: np.array(mapping, dtype=np.float64) for base, mapping in degenerate_base_mapping.items()}

# Define nucleotide sets for alignment
nucleotides_A = ['A', 'T', 'C', 'G', 'N']
nucleotides_B = ['R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D', 'N']
nucleotides = alphabet = nucleotides_A + nucleotides_B
pairs = [(n1, n2) for n1 in nucleotides for n2 in nucleotides]
pairs_to_remove = [(n1, n2) for n1 in nucleotides_B for n2 in nucleotides_B]
pairs_to_remove = [i for i in pairs_to_remove]
# Removing duplicates while keeping the order intact
unique_pairs = []
seen = set()
for pair in pairs:
    if pair not in seen and (pair[1], pair[0]) not in seen:
        if pair not in pairs_to_remove and (pair[1], pair[0]) not in pairs_to_remove:
            unique_pairs.append(pair)
            seen.add(pair)
pairs = unique_pairs

def reverse_complement(seq):
    return ''.join(degenerate_complement[base] for base in reversed(seq))


def write_full_fastq(sequences, directions, barcode_fn, full_fn, filtered_fn):
    # if synthetic data read from fastq for filtering
    if full_fn.endswith('.pkl'):
        full_fn = full_fn.replace('.pkl', '.fastq')

    # build index -> barcode mapping
    idx_to_barcode = {int(i): s for i, s in sequences}

    # write barcode fasta
    with open(barcode_fn, "w") as bf:
        for idx, seq in idx_to_barcode.items():
            bf.write(f">{idx}\n{seq}\n")

    matched_records = []
    for i, rec in enumerate(SeqIO.parse(full_fn, "fastq")):
        if i in idx_to_barcode:
            if directions[i] == 1:  # reverse complement required
                new_seq = rec.seq.reverse_complement() # reverse complement sequence
                new_quals = rec.letter_annotations["phred_quality"][::-1] # reverse quality scores
                rec = SeqRecord(
                    new_seq,
                    id=rec.id,
                    description=rec.description,
                    letter_annotations={"phred_quality": new_quals})
            matched_records.append(rec)

    # write filtered full reads with qualities preserved
    SeqIO.write(matched_records, filtered_fn, "fastq")
    print(f"Wrote {len(matched_records)} barcodes to {barcode_fn}")
    print(f"Wrote {len(matched_records)} full reads to {filtered_fn}")


def write_synthetic_fastq(sequences, filename, qualities=None, headers=None):
    with open(filename, "w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers is not None else str(i)

            f.write(f"@{header}\tmv:syntheticdata\n")
            f.write(f"{seq}\n")
            f.write("+\n")

            if qualities is not None:
                qual = qualities[i]
                if len(qual) != len(seq):
                    raise ValueError(f"Quality length does not match sequence at index {i}")
                f.write(f"{qual}\n")
            else:
                # Write dummy high quality (Phred 40 = 'I')
                f.write(f"{'I' * len(seq)}\n")

def read_fastq(filename, seq_limit=None):
    headers = []
    sequences = []
    i = 0
    with open(filename, "r") as f:
        while True:
            header = f.readline()
            if not header:
                break  # EOF
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()
            headers.append(header[1:].strip())  # remove '@'
            sequences.append(seq)
            i += 1
            if seq_limit is not None and i >= seq_limit:
                break

    return sequences, headers

def read_fastqgz(filename, seq_limit=None):
    headers = []
    sequences = []
    i = 0

    # Use gzip.open for .gz files, in text mode
    with gzip.open(filename, "rt") as f:
        while True:
            header = f.readline()
            if not header:
                break  # EOF
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()

            headers.append(header[1:].strip())  # remove '@'
            sequences.append(seq)

            i += 1
            if seq_limit is not None and i >= seq_limit:
                break

    return sequences, headers
def convert_AA_to_nucleotide(aa_seq_list):
    codon_usage = pd.read_csv('ecoli_codon_usage.csv')

    nuc_list = []
    for aa_seq in aa_seq_list:
        dna_seq = ''
        for i in range(0, len(aa_seq)):
            AA = aa_seq[i]
            dna = codon_usage[codon_usage.aa == AA].sort_values('freq', ascending=False).iloc[0]['dna']
            dna_seq += dna
        nuc_list.append(dna_seq)
    return nuc_list

def read_file_in_batches(fn, batch_size=100, last_position=None):
    last_position = 0 if last_position is None else last_position # To track the last position read from the file
    current_position = 0
    delete_lines = True if fn == 'temp.fasta' else False
    step = 2
    if fn[-1] == 'q':
        step += 2

    reading = True
    while reading:
        batch = []
        names = []
        with open(fn, 'r') as file:
            file.seek(last_position)  # Move to the last read position
            for _ in range(batch_size*step):  # Read up to batch_size lines
                line = file.readline()
                if not line:  # No more lines to read
                    print(f"End of file reached: {fn}")
                    reading = False
                    break
                if current_position % step == 1:
                    batch.append(line.strip())
                else:
                    names.append(line.strip())
                current_position += 1

            # Update last_position after reading
            last_position = file.tell()

            temp_path = f"{fn}.temp"  # Temporary file path

            if delete_lines and batch:
                index = 0
                with open(fn, 'r') as original_file, open(temp_path, 'w') as temp_file:
                    for line in original_file:
                        if line.startswith('>'):  # Check for a header line
                            header = line  # Save the header for later
                        else:
                            if line.strip() not in batch:  # Check if the sequence is not in `batch`
                                if header:  # Write the stored header if it exists
                                    temp_file.write(header)  # Or use the original header
                                    temp_file.write(line)  # Write the sequence
                                    index += 1
                                header = None  # Reset the header

                os.replace(temp_path, fn)

            if batch:
                return batch, names, last_position  # Yield a batch of sequences

    return None, None, None

def generate_random_mut(coding_sequence, num_muts=1):
    coding_sequence = list(coding_sequence)
    for i in range(num_muts):
        mut_pos = np.random.randint(0, len(coding_sequence))
        mut_nt = np.random.choice(nucleotides_A[:-1])
        coding_sequence[mut_pos] = mut_nt
    return ''.join(coding_sequence)
