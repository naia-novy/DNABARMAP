import time
import argparse
from os import makedirs, path
from shutil import rmtree

from dnabarmap.array_align import align
from dnabarmap.cluster import run_vsearch, save_full_seqs
from dnabarmap.consensus import determine_consensus
from dnabarmap.map import determine_mapping


def main(**kwargs):
    initial_time = time.time()
    kwargs['input_fn'] = kwargs['fastq_fn']
    kwargs['fastq_fn'] = kwargs['input_fn'].replace('.pkl', '.fastq') # in case synthetic data
    barcode_out = 'tmp/'+kwargs['input_fn'].split('/')[-1].split('.')[0] + '_barcodes.fasta'
    filtered_fn = barcode_out.replace('_barcodes.fasta', '_filtered.fastq')
    kwargs['output_fn'] = barcode_out
    kwargs['filtered_fn'] = filtered_fn

    # Extract and align barcodes using approximate alignment to degenerate reference
    print('Aligning barcodes...')
    align_start_time = time.time()
    align(**kwargs)
    align_time = time.time() - align_start_time
    print(f'Finished aligning and extracting barcodes in {round(align_time / 60, 1)} minutes\n')

    # Adjustment if using synthetic data for validation
    kwargs['fasta_fn'] = filtered_fn

    # Cluster aligned barcodes using vsearch
    print('Clustering barcodes...')
    cluster_start_time = time.time()
    run_vsearch(**kwargs)
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
    parser.add_argument('--fastq_fn', type=str, default=None)
    parser.add_argument('--fasta_fn', type=str, default=None)
    parser.add_argument("--mapping_fn", default=None,
                        help="Final mapping output filename")
    parser.add_argument("--base_fn", default='syndata/syndataC',
                        help="Filename base to use when fasta_fn, fastq_fn, or mapping_fn is not provided")

    # Define barcode and sequence parameters
    parser.add_argument('--barcode_template', type=str,
                        # default='YHWSBYRVWBYMDSKWWVSBWSSWDRKMDSYMWYSKRWYDRYSKMSYDYSWVYRYKRYVR',
                        # default='YBRHBRHSDHSDYVDYVWBVWBMDBMDSHDSHKVHKVYDVYDMBDMBWVBWVKHVKHRBH',
                        # default='RVYDVYDMBDMBRHBRHKVHKVWBVWBMDBMDSHDSHRBHRBWVBWVKHVKHSDHSDYVD',
                        default='NVKMRBSMDDYVMWYSBDYSDHMBWMKBWSDRYWBMNKYVDKMBSWMBDMWYRBDMHKSN',

                        help='Degenerate reference for conducting approximate alignment of sequences')
    parser.add_argument("--left_coding_flank", default='CTGCTATCGT',
                        help="Left constant sequence of coding region")
    parser.add_argument("--right_coding_flank", default='ATCTAGCATC',
                        help="Right constant sequence of coding region")

    # Alignment parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5,
                        help='How many times to try next best suggestion before giving up during alignment')
    parser.add_argument('--match_multiplier', type=float, default=10,
                        help='Multiply per base scores by this value to favor alignment to degenerates with less options')
    parser.add_argument('--minimum_match_fraction', type=float, default=0.80,
                        help='Require at least this fraction of bases to match any reference possiblity for inclusion in clustering')
    parser.add_argument('--max_len', type=int, default=150,
                        help='Shave off ends of sequences over this length for efficiency, '
                             'reccomended to be at least twice length of barcode')
    parser.add_argument('--buffer', type=int, default=30,
                        help='Expected constant region on the DNA fragment before the barcode to be shaved off')

    # Cluster parameters
    parser.add_argument("--cluster_iterations", type=int, default=10, help="Repeat greedy clustering this "
                                                                          "many times with decreasing stringency each iteration")
    parser.add_argument("--min_sequences", type=int, default=25,
                        help="Minimum num_sequences for cluster to be valid >=")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of threads for clustering")

    parser.add_argument("--save_intermediate_files", default=True, action='store_true',
                        help="Should not delete intermediate files generated during DNABARMAP")
    parser.add_argument("--synthetic_data_available", default=False, action='store_true',
                        help="Run comparisons to true values using synthetic data to validate functionality/accuracy")

    args = parser.parse_args()

    # Set up directories and filenames
    args.output_dir = 'tmp/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'

    if args.base_fn is None:
        name = args.fastq_fn if args.fastq_fn is None else args.fasta_fn is None
        assert name is not None, 'Must provide either fasta_fn, fastq_fn, or base_fn'
        args.base_fn = '.'.join(name.split('.')[:-1])
    if args.fastq_fn is None:
        args.fastq_fn = args.base_fn + '.fastq'
    if args.fasta_fn is None:
        args.fasta_fn = args.base_fn + '.fasta'
    if args.mapping_fn is None:
        args.mapping_fn = args.base_fn + '_mapping.tsv'
    args.barcodes_fn = args.base_fn + '_barcodes.fasta'  # used in array_align
    args.output_mapping_fn = 'DNABARMAP_outputs/' + args.base_fn.split('/')[-1] + '_mapping.tsv'

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
    # rmtree('tmp/')

    if args.synthetic_data_available:
        assert args.fastq_fn.endswith('.pkl'), 'Must provide pkl format for synthetic data'
    if args.min_sequences < 25:
        print('WARNING: min_sequences is less than 25, this is not reccomended and may cause innacurate consensus sequence determination')

    makedirs(args.cluster_dir, exist_ok=True)
    makedirs(args.consensus_dir, exist_ok=True)
    makedirs('DNABARMAP_outputs', exist_ok=True)

    args.seq_limit_for_debugging = None # 10000

    main(**vars(args))

    if not args.save_intermediate_files:
        rmtree('tmp/')

if __name__ == '__main__':
    cli()