import time
import argparse
from os import makedirs, path
from shutil import rmtree
from pathlib import Path

from dnabarmap.array_align import align
from dnabarmap.cluster import cluster, save_full_seqs
from dnabarmap.map import consensus_mapping
from dnabarmap.consensus import determine_consensus_parallel


def _strip_input_suffix(input_path):
    basename = path.basename(input_path)
    for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq', '.pkl'):
        if basename.endswith(ext):
            return basename[:-len(ext)]
    return path.splitext(basename)[0]


def _default_output_dir(input_fn):
    input_path = Path(input_fn)
    return str(input_path.parent / _strip_input_suffix(input_path.name))


def main(**kwargs):
    kwargs['extra'] = kwargs.get('extra', 10)

    initial_time = time.time()
    kwargs['id'] = round(kwargs['id'], 2)
    kwargs['c'] = 0.5

    # Extract and align barcodes using approximate alignment to degenerate reference
    print('Aligning barcodes...')
    align_start_time = time.time()
    align(**kwargs)
    align_time = time.time() - align_start_time
    print(f'Finished aligning and extracting barcodes in {round(align_time / 60, 1)} minutes\n')

    # Cluster aligned barcodes using MMseqs2
    # cluster() now combines batch files internally — no need to pass input_fa
    print('Clustering barcodes...')
    cluster_start_time = time.time()
    cluster(**kwargs)
    save_full_seqs(**kwargs)
    cluster_time = time.time() - cluster_start_time
    print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')

    # Determine consensus sequences for clusters using minimap2 and medaka
    print('Determining consensus sequences...')
    consensus_start_time = time.time()
    determine_consensus_parallel(**kwargs)
    consensus_time = time.time() - consensus_start_time
    print(f'Finished determining consensus sequences in {round(consensus_time / 60, 1)} minutes\n')

    # Use regular expressions to map barcodes to coding sequences for consensus sequences
    print('Mapping barcodes to coding sequences...')
    mapping_start_time = time.time()
    consensus_mapping(**kwargs)
    mapping_time = time.time() - mapping_start_time
    print(f'Finished mapping barcodes in {round(mapping_time / 60, 1)} minutes\n')

    final_time = time.time() - initial_time
    print('Completed DNABARMAP mapping process')
    print(f'{round(final_time / 60 / 60, 1)} hours elapsed\n')


def cli():
    parser = argparse.ArgumentParser(
        description="Run the full DNABARMAP pipeline: align barcode windows, "
                    "cluster reads, build cluster consensuses, and write the "
                    "final barcode-to-variant mapping table."
    )

    # Directories and filenames
    parser.add_argument('--input_fn', type=str, required=True, default=None,
                        help="Input reads. Use a synthetic .pkl file for "
                             "validation runs, or a .fastq/.fastq.gz file for "
                             "real sequencing data.")
    parser.add_argument("--mapping_fn", default=None, required=True,
                        help="Output TSV filename for the final mapping results.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root directory for aligned/, clusters/, "
                             "consensus/, and mapping outputs. Defaults to "
                             "<input stem>/ next to the input file.")

    # Define barcode and sequence parameters
    parser.add_argument('--barcode_template', type=str, default=None, required=True,
                        help='Degenerate barcode template used for approximate '
                             'barcode alignment. For short or highly degenerate '
                             'barcodes, it often helps to include nearby '
                             'constant flanking sequence here; the constant '
                             'flanks can be removed later during mapping.')
    parser.add_argument("--left_coding_flank", required=True, default=None,
                        help="Constant sequence immediately left of the coding "
                             "region (in same reading frame as barcode). Used during final mapping. ")
    parser.add_argument("--right_coding_flank", required=True, default=None,
                        help="Constant sequence immediately right of the coding "
                             "region (in same reading frame as barcode). Used during final mapping.")

    # Alignment parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size for barcode alignment scoring. "
                             "Lower this if memory is tight.")
    parser.add_argument("--id", type=float, default=0.69,
                        help="MMseqs minimum identity for barcode clustering "
                             "(0-1). Higher values split more aggressively; "
                             "lower values merge more. A good starting range is "
                             "0.69 to 0.75.")
    parser.add_argument("--min_sequences", type=int, default=10,
                        help="Minimum reads required for a cluster to continue "
                             "to consensus.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Total thread budget for clustering, consensus, "
                             "and mapping.")
    parser.add_argument("--max_open_fastq_handles", type=int, default=None,
                        help="Maximum cluster FASTQ files to keep open at once "
                             "when materializing clusters/full_seqs. Lower "
                             "this on servers with strict open-file limits.")

    # Consensus parameters
    parser.add_argument("--medaka_model", type=str,
                        default="none",
                        help="Medaka model name, or 'none' to skip Medaka and "
                             "use the non-Medaka consensus path.")
    parser.add_argument("--max_mismatches", type=int, default=5,
                        help="Maximum substitutions allowed when validating the "
                             "barcode interval against the template.")
    parser.add_argument("--max_indels", type=int, default=3,
                        help="Maximum indels allowed when validating the "
                             "barcode interval against the template.")
    parser.add_argument("--min_window_score", type=float, default=0.7,
                        help="Minimum barcode-template match score required to "
                             "accept a candidate barcode interval.")
    parser.add_argument("--extra", type=int, default=1,
                        help="Bases of context to keep on each side of the "
                             "aligned barcode window. Use 0 to keep only the "
                             "barcode-length window.")
    parser.add_argument("--reference_seqs", type=str, default=None,
                        help="Optional reference sequences for coding-region "
                             "snapping during mapping.")
    parser.add_argument("--ref_seq_col", type=str, default=None,
                        help="Sequence column to use when --reference_seqs "
                             "points to a table or pickle.")
    parser.add_argument("--ref_name_col", type=str, default=None,
                        help="Name column to use when --reference_seqs points "
                             "to a table or pickle.")
    parser.add_argument("--max_edits_compressed", type=int, default=3,
                        help="Maximum edit distance in homopolymer-compressed "
                             "space for reference snapping prefilter.")
    parser.add_argument("--max_edits_full", type=int, default=5,
                        help="Maximum full-length edit distance to accept a "
                             "reference snap.")

    parser.add_argument("--synthetic_data_available", default=False, action='store_true',
                        help="Compare outputs to truth columns in a synthetic "
                             ".pkl input.")

    args = parser.parse_args()

    args.input_fq = args.input_fn.replace('.pkl', '.fastq')
    if args.synthetic_data_available:
        assert args.input_fn.endswith('.pkl'), 'Must provide pkl format for synthetic data'
    if args.input_fn.endswith('.pkl'):
        if args.reference_seqs is None:
            args.reference_seqs = args.input_fn
        if args.ref_seq_col is None:
            args.ref_seq_col = 'variant'
        args.max_edits_compressed = max(args.max_edits_compressed, 6)
        args.max_edits_full = max(args.max_edits_full, 15)

    # Set up directories and filenames
    inferred_output_dir = False
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.input_fn)
        inferred_output_dir = True
    output_dir = Path(args.output_dir)
    args.output_dir = str(output_dir) + '/'
    args.cluster_dir = str(output_dir / 'clusters') + '/'
    args.consensus_dir = str(output_dir / 'consensus') + '/'
    args.aligned_dir = str(output_dir / 'aligned') + '/'

    args.base_fn = str(Path(args.input_fq).with_suffix(''))
    args.barcodes_fn = args.base_fn + '_barcodes.fasta'  # used in array_align
    args.output_mapping_fn = 'DNABARMAP_outputs/' + args.base_fn.split('/')[-1] + '_mapping.tsv'

    if args.left_coding_flank is None:
        args.left_coding_flank = ''
    if args.right_coding_flank is None:
        args.right_coding_flank = ''

    if path.exists(args.output_dir):
        rmtree(args.output_dir)

    if args.min_sequences < 10:
        print('WARNING: min_sequences is less than 10, this is not recommended '
              'and may cause inaccurate consensus sequence determination')

    makedirs(args.cluster_dir + 'barcodes/', exist_ok=True)
    makedirs(args.cluster_dir + 'full_seqs/', exist_ok=True)
    makedirs(args.aligned_dir, exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)
    makedirs('DNABARMAP_outputs', exist_ok=True)

    input_stem = _strip_input_suffix(args.input_fq)
    # output_fn is only used by array_align for per-batch barcode output
    args.output_fn = args.aligned_dir + input_stem + '_barcodes.fasta'
    args.reoriented_fn = args.output_fn.replace('barcodes.fasta', 'reoriented.fastq')

    if inferred_output_dir:
        print(f'Output dir not provided; using inferred path: {args.output_dir}')
    else:
        print(f'Output dir: {args.output_dir}')
    print(f'Aligned barcode FASTA: {args.output_fn}')
    print(f'Reoriented FASTQ: {args.reoriented_fn}')

    args.seq_limit_for_debugging = None

    main(**vars(args))



if __name__ == '__main__':
    cli()
