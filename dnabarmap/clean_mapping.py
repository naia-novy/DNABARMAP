import pandas as pd

from dnabarmap.utils import nuc_dict

def checker(barcode_template, test_barcode):
    result = [test_barcode[i] in nuc_dict[v] for i, v in enumerate(barcode_template)]
    return sum(result) == len(result)

def filter_true(sequence, options):
    return sequence in options


def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R',  # R=A/G, Y=C/T
        'S': 'S', 'W': 'W',  # S=G/C, W=A/T (self-complementary)
        'K': 'M', 'M': 'K',  # K=G/T, M=A/C
        'B': 'V', 'V': 'B',  # B=C/G/T (not A), V=A/C/G (not T)
        'D': 'H', 'H': 'D',  # D=A/G/T (not C), H=A/C/T (not G)
        'N': 'N'
    }
    return ''.join(complement[b] for b in reversed(seq.upper()))



# fns = ['high_depth/barcode22_mapping.tsv',
#        'high_depth/barcode23_mapping.tsv',
#        'low_depth/barcode22_mapping.tsv',
#        'low_depth/barcode23_mapping.tsv', ]


fns = ['mapping/barcode22_mapping.tsv',
       'mapping/barcode23_mapping.tsv']


# fns = ['mapping_direct_illumina/mapping_direct_small-promoter_joined.tsv',
#        'mapping_direct_illumina/mapping_direct_large-promoter_joined.tsv']


template_barcode = 'MNBRWHBWRYBYRYWNVYDRHKHSNDHKMRDWKDMBKWNVSWKWVNBVWKDVWDKVHVKNDHVKDMVHKHSKWBN'
promoter_path = 'updated_control_promoters2.csv'
barcode_len = len(template_barcode)

# require known promoter sequence
RBS = 'TTTAACTTTAAGAAGGAGATATACAT'

# Load data
promoter_options = pd.read_csv(promoter_path)
promoter_options.promoter = promoter_options.promoter + RBS

for fn in fns:
    df = pd.read_csv(fn, sep='\t')
    df = df.drop(columns=['filename'])
    df.dropna(inplace=True)
    print(len(df))

    # Filter by length
    df = df.loc[df.barcode.str.len() == barcode_len]
    print(len(df))

    # # Group
    # df = df.groupby('barcode').size().reset_index(name='counts')
    # print(len(df))
    #
    # df = df.loc[df.counts >= required_obs]
    # print(len(df))

    # # Ensure sequences are valid
    # Ensure barcodes match the template
    test = df.loc[df.barcode.apply(lambda x: checker(template_barcode, x))]

    if len(test) <= len(df) * 0.05:
        print(f"Trying reverse complement")
        df['barcode'] = df['barcode'].apply(reverse_complement)
        df = df.loc[df.barcode.apply(lambda x: checker(template_barcode, x))]
        df['coding_region'] = df['coding_region'].apply(reverse_complement)

    else:
        df = test

    df.to_csv(fn.replace('.tsv', '_final.tsv'), sep='\t', index=False)
