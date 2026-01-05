import time
import argparse
from os import makedirs, path
from shutil import rmtree

from dnabarmap.array_align import align
from dnabarmap.cluster import cluster, save_full_seqs
from dnabarmap.consensus import determine_consensus_consecutive
from dnabarmap.map import consensus_mapping


def main(**kwargs):
    kwargs['extra'] = 10

    initial_time = time.time()
    kwargs['id'] = round(kwargs['id'], 2)
    kwargs['c'] = 0.75 #round(kwargs['id'], 2)

    # Extract and align barcodes using approximate alignment to degenerate reference
    print('Aligning barcodes...')
    align_start_time = time.time()
    align(**kwargs)
    align_time = time.time() - align_start_time
    print(f'Finished aligning and extracting barcodes in {round(align_time / 60, 1)} minutes\n')


    # Cluster aligned barcodes using vsearch
    print('Clustering barcodes...')
    cluster_start_time = time.time()
    cluster(**kwargs)
    save_full_seqs(**kwargs)
    cluster_time = time.time() - cluster_start_time
    print(f'Finished clustering barcodes in {round(cluster_time / 60, 1)} minutes\n')

    # Determine consensus seqeunces for clusters using minimap2 and racon
    print('Determining consensus sequences...')
    consensus_start_time = time.time()
    determine_consensus_consecutive(**kwargs)
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
    parser = argparse.ArgumentParser()

    # Directories and filenaemes
    parser.add_argument('--input_fq', type=str, required=True, default=None)
    parser.add_argument("--mapping_fn", default=None, required=True,
                        help="Final mapping output filename")

    # Define barcode and sequence parameters
    parser.add_argument('--barcode_template', type=str, default=None, required=True,
                        help='Degenerate reference for conducting approximate alignment of sequences')
    parser.add_argument("--left_coding_flank", required=True, default=None,
                        help="Left constant sequence of coding region")
    parser.add_argument("--right_coding_flank", required=True, default=None,
                        help="Right constant sequence of coding region")

    # Alignment parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--id", type=float, default=0.75, help="Value between 0 and 1 for "
                                                                           "minimum identify between barcodes for clustering."
                                                                           "Reccomended >0.75, but can be reduced for small "
                                                                            "libraries or extra long barcodes")
    parser.add_argument("--min_sequences", type=int, default=20,
                        help="Minimum num_sequences for cluster to be valid >= /"
                             "aim for at least 3x the expected depth")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for clustering")

    parser.add_argument("--save_intermediate_files", default=True, action='store_true',
                        help="Should not delete intermediate files generated during DNABARMAP")
    parser.add_argument("--synthetic_data_available", default=False, action='store_true',
                        help="Run comparisons to true values using synthetic data to validate functionality/accuracy")

    all_args = parser.parse_known_args()
    args = all_args[0]

    # Set up directories and filenames
    args.barcode_directory = 'barcode_' + args.input_fq.split('/barcode')[-1].split('/')[0].split('_')[0]
    args.barcode_directory = 'sample' if args.barcode_directory == '' else args.barcode_directory
    args.output_dir = f'temp/{args.barcode_directory}/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + 'consensus/'

    args.base_fn = '.'.join(args.input_fq.split('.')[:-1])
    args.barcodes_fn = args.base_fn + '_barcodes.fasta'  # used in array_align
    args.output_mapping_fn = 'DNABARMAP_outputs/' + args.base_fn.split('/')[-1] + '_mapping.tsv'

    if args.synthetic_data_available:
        assert args.input_fq.endswith('.pkl'), 'Must provide pkl format for synthetic data'

    if args.left_coding_flank is None:
        args.left_coding_flank = ''
    if args.right_coding_flank is None:
        args.right_coding_flank = ''

    # remove previous iterations
    if path.exists(args.cluster_dir):
        rmtree(args.cluster_dir)
    if path.exists(args.consensus_dir):
        rmtree(args.consensus_dir)
    if path.exists(args.output_dir):
        rmtree(args.output_dir)

    if args.synthetic_data_available:
        assert args.input_fq.endswith('.pkl'), 'Must provide pkl format for synthetic data'
    if args.min_sequences < 10:
        print('WARNING: min_sequences is less than 10, this is not recommended and may cause inaccurate consensus sequence determination')

    makedirs(args.cluster_dir+'/barcodes/', exist_ok=True)
    makedirs(args.cluster_dir+'/full_seqs/', exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)
    makedirs('DNABARMAP_outputs', exist_ok=True)

    args.output_fn = 'temp/'+args.input_fq.split('/')[-1].split('.')[0] + '_barcodes.fasta'
    args.reoriented_fn = args.input_fq.replace('.fastq', '_reoriented.fastq')

    args.seq_limit_for_debugging = None # 10000

    main(**vars(args))

    if not args.save_intermediate_files:
        rmtree('temp/')

if __name__ == '__main__':
    cli()