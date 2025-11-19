import time
import argparse
from os import makedirs, path
from shutil import rmtree

from dnabarmap.array_align import align
from dnabarmap.cluster import cluster, save_full_seqs
from dnabarmap.consensus import determine_consensus
from dnabarmap.map import determine_mapping


def main(**kwargs):
    extra = 10
    c = 0.75

    initial_time = time.time()
    kwargs['output_fn'] = kwargs['input_fn']
    kwargs['id'] = round(kwargs['id'] * len(kwargs['barcode_template'])/(extra*2+len(kwargs['barcode_template'])), 2)
    kwargs['c'] = round(c, 2)

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
    determine_consensus(**kwargs)
    consensus_time = time.time() - consensus_start_time
    print(f'Finished determining consensus sequences in {round(consensus_time / 60, 1)} minutes\n')

    # Use regular expressions to map barcodes to coding sequences for consensus sequences
    print('Mapping barcodes to coding sequences...')
    mapping_start_time = time.time()
    determine_mapping(**kwargs)
    mapping_time = time.time() - mapping_start_time
    print(f'Finished mapping barcodes in {round(mapping_time / 60, 1)} minutes\n')

    final_time = time.time() - initial_time
    print('Completed DNABARMAP mapping process')
    print(f'{round(final_time / 60 / 60, 1)} hours elapsed\n')


def cli():
    parser = argparse.ArgumentParser()

    # Directories and filenaemes
    parser.add_argument('--input_fn', type=str, default=None, required=True,
                        help='Combined input fasta file')
    parser.add_argument("--reoriented_fn", default=None, required=True,
                        help="Combined reoriented fastq file")
    parser.add_argument("--mapping_fn", default=None, required=True,
                        help="Final mapping output filename")

    # Define barcode and sequence parameters
    parser.add_argument('--barcode_template', type=str,
                        default='VHBKVBHBDMKNVBYDKVBYNKSSYSKNNYSKHYSDNBMKBNSHKBSDMBBKMBBRYSBH',
                        help='Degenerate reference for conducting approximate alignment of sequences')
    parser.add_argument("--left_coding_flank", default='TATCGT',
                        help="Left constant sequence of coding region")
    parser.add_argument("--right_coding_flank", default='ATCTAG',
                        help="Right constant sequence of coding region")

    # Alignment parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--buffer', type=int, default=60,
                        help='Expected constant region on the DNA fragment before the barcode to be shaved off')


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

    args = parser.parse_args()

    # Set up directories and filenames

    args.barcode_directory = 'barcode_' + args.input_fn.split('/barcode')[-1].split('/')[0].split('_')[0]
    args.barcode_directory = 'sample' if args.barcode_directory is None else args.barcode_directory
    args.output_dir = f'temp/{args.barcode_directory}/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'

    args.output_mapping_fn = args.mapping_fn

    if args.synthetic_data_available:
        assert args.fastq_fn.endswith('.pkl'), 'Must provide pkl format for synthetic data'

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
        assert args.fastq_fn.endswith('.pkl'), 'Must provide pkl format for synthetic data'
    if args.min_sequences < 15:
        print('WARNING: min_sequences is less than 15, this is not recommended and may cause inaccurate consensus sequence determination')

    makedirs(args.cluster_dir+'/barcodes/', exist_ok=True)
    makedirs(args.cluster_dir+'/full_seqs/', exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)
    makedirs('DNABARMAP_outputs', exist_ok=True)

    args.seq_limit_for_debugging = None # 10000

    main(**vars(args))

    if not args.save_intermediate_files:
        rmtree('temp/')

if __name__ == '__main__':
    cli()