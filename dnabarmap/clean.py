import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, product

from dnabarmap.utils import nuc_dict


def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R',
        'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K',
        'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D',
        'N': 'N'
    }
    return ''.join(complement[b] for b in reversed(seq.upper()))


def checker(barcode_template, test_barcode):
    result = [test_barcode[i] in nuc_dict[v] for i, v in enumerate(barcode_template)]
    return sum(result) == len(result)


def cluster_barcodes_hamming(barcodes, max_distance=1):
    """Cluster barcodes by Hamming distance ≤ max_distance using Union-Find."""
    barcode_list = list(set(barcodes))
    barcode_to_idx = {bc: i for i, bc in enumerate(barcode_list)}

    # Union-Find with path compression
    parent = list(range(len(barcode_list)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    existing = set(barcode_list)
    alphabet = 'ACGT'

    for bc in barcode_list:
        bc_idx = barcode_to_idx[bc]
        bc_list = list(bc)
        k = len(bc)

        # Generate all neighbors at distance 1 to max_distance
        for dist in range(1, max_distance + 1):
            # Choose which positions to edit
            for positions in combinations(range(k), dist):
                orig_nucs = [bc_list[p] for p in positions]

                # Generate all possible substitutions at those positions
                for substitutions in product(alphabet, repeat=dist):
                    # Skip if any position unchanged (not a real edit)
                    if any(substitutions[i] == orig_nucs[i] for i in range(dist)):
                        continue

                    # Apply substitutions
                    for i, pos in enumerate(positions):
                        bc_list[pos] = substitutions[i]

                    neighbor = ''.join(bc_list)
                    if neighbor in existing:
                        union(bc_idx, barcode_to_idx[neighbor])

                    # Restore original
                    for i, pos in enumerate(positions):
                        bc_list[pos] = orig_nucs[i]

    barcode_to_cluster = {bc: find(barcode_to_idx[bc]) for bc in barcode_list}
    return barcode_to_cluster


def collapse_clusters(df, barcode_to_cluster):
    """Collapse clusters, keeping highest-count barcode and summing counts."""
    df = df.copy()
    df['cluster'] = df['barcode'].map(barcode_to_cluster)

    collapsed = []
    for cluster_id, group in df.groupby('cluster'):
        # Representative barcode = highest count
        best_row = group.loc[group['counts'].idxmax()]

        # Majority coding region (weighted by counts)
        coding_counts = group.groupby('coding_region')['counts'].sum()
        best_coding = coding_counts.idxmax()

        collapsed.append({
            'barcode': best_row['barcode'],
            'coding_region': best_coding,
            'counts': group['counts'].sum()
        })

    return pd.DataFrame(collapsed)


# =============================================================================
# Configuration
# =============================================================================

template_barcode = 'MNBRWHBWRYBYRYWNVYDRHKHSNDHKMRDWKDMBKWNVSWKWVNBVWKDVWDKVHVKNDHVKDMVHKHSKWBN'
barcode_len = len(template_barcode)

promoter_path = 'mapping/updated_control_promoters2.csv'

# fns = ['mapping/barcode22_mapping.tsv',
#        'mapping/barcode23_mapping.tsv']

fns = ['mapping_direct_illumina/mapping_direct_small-promoter_joined.tsv',
       'mapping_direct_illumina/mapping_direct_large-promoter_joined.tsv']


count_cutoff = 30
hamming_distance = 2 # <-- Set your desired Hamming distance here

RBS = 'TTTAACTTTAAGAAGGAGATATACAT'
promoter_options = pd.read_csv(promoter_path)
promoter_options.promoter = promoter_options.promoter + RBS

# =============================================================================
# Processing
# =============================================================================

for fn in fns:
    df = pd.read_csv(fn, sep='\t')
    df = df.drop(columns=['filename'])
    df.dropna(inplace=True)
    df = df.loc[df.barcode.str.len() == barcode_len]

    # Group by barcode + coding_region
    df_combined = df.groupby(['barcode', 'coding_region']).size().reset_index(name='counts')
    print(f"{fn}: {len(df_combined)} unique barcode+coding pairs")

    # Filter by frequency
    if df_combined.counts.mean() > 1.25:
        # ignore these steps if they were previously clustered
        df_combined = df_combined.loc[df_combined.counts >= count_cutoff]
        print(f"After count filter: {len(df_combined)}")

        # Cluster barcodes by Hamming distance
        barcode_to_cluster = cluster_barcodes_hamming(
            df_combined['barcode'].tolist(),
            max_distance=hamming_distance
        )
        df_combined = collapse_clusters(df_combined, barcode_to_cluster)
        print(f"After Hamming-{hamming_distance} clustering: {len(df_combined)}")

    # Ensure barcodes match the template
    test = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]

    if len(test) <= len(df_combined) * 0.05:
        print(f"Trying reverse complement for {fn}")
        df_combined['barcode'] = df_combined['barcode'].apply(reverse_complement)
        df_combined = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]
        df_combined['coding_region'] = df_combined['coding_region'].apply(reverse_complement)
    else:
        df_combined = test

    print(f"{fn}: {len(df_combined)} final clusters\n")
    sns.histplot(df_combined.counts)
    plt.title(fn)
    plt.show()

    df_combined.to_csv(fn.replace('.tsv', '_final.tsv'), sep='\t', index=False)