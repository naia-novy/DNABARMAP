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
    kwargs['input_fn'] = kwargs['fastq_fn']
    kwargs['fastq_fn'] = kwargs['input_fn'].replace('.pkl', '.fastq') # in case synthetic data
    barcode_out = 'temp/'+kwargs['input_fn'].split('/')[-1].split('.')[0] + '_barcodes.fasta'
    kwargs['output_fn'] = barcode_out
    kwargs['reoriented_fn'] = kwargs['fastq_fn'].replace('.fastq', '_reoriented.fastq')
    kwargs['id'] = round(kwargs['id'] * len(kwargs['barcode_template'])/(extra*2+len(kwargs['barcode_template'])), 2)
    kwargs['c'] = round(c, 2)
    # kwargs['c'] = round(c * len(kwargs['barcode_template'])/(extra*2+len(kwargs['barcode_template'])), 2)

    # Extract and align barcodes using approximate alignment to degenerate reference
    print('Aligning barcodes...')
    align_start_time = time.time()
    align(extra=extra, **kwargs)
    align_time = time.time() - align_start_time
    print(f'Finished aligning and extracting barcodes in {round(align_time / 60, 1)} minutes\n')

    # Cluster aligned barcodes using vsearch
    print('Clustering barcodes...')
    cluster_start_time = time.time()
    # c = min(round(0.5*((len(kwargs['barcode_template'])+extra*2)/len(kwargs['barcode_template'])), 2), 0.95)
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
    parser.add_argument('--fastq_fn', type=str, default=None)
    parser.add_argument("--mapping_fn", default=None,
                        help="Final mapping output filename")
    parser.add_argument("--base_fn", default='syndata/syndataB',
                        help="Filename base to use when fasta_fn, fastq_fn, or mapping_fn is not provided")

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
    args.output_dir = 'temp/'
    args.cluster_dir = args.output_dir + '/clusters/'
    args.consensus_dir = args.output_dir + '/consensus/'

    if args.base_fn is None:
        name = args.fastq_fn
        assert name is not None, 'Must provide either fasta_fn, fastq_fn, or base_fn'
        args.base_fn = '.'.join(name.split('.')[:-1])
    if args.fastq_fn is None:
        args.fastq_fn = args.base_fn + '.fastq'
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
    if path.exists('temp/'):
        rmtree('temp/')

    if args.synthetic_data_available:
        assert args.fastq_fn.endswith('.pkl'), 'Must provide pkl format for synthetic data'
    if args.min_sequences < 20:
        print('WARNING: min_sequences is less than 20, this is not reccomended and may cause innacurate consensus sequence determination')

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