import subprocess
import tempfile
from os import remove, makedirs, path
import gzip
from pathlib import Path

PBSIM_QUALITY_PRESETS = {
    'default': {},
    'cleaner': {
        'accuracy_mean': 0.93,
        'difference_ratio': '39:24:36',
        'hp_del_bias': 1.0,
    },
    'harsh': {
        'accuracy_mean': 0.87,
        'difference_ratio': '39:24:36',
        'hp_del_bias': 2.0,
    },
}


def _resolve_model_path(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path.resolve())

    repo_root = Path(__file__).resolve().parent.parent
    repo_model = repo_root / 'pbsim3_models' / model_name
    if repo_model.exists():
        return str(repo_model.resolve())

    repo_relative = repo_root / model_name
    if repo_relative.exists():
        return str(repo_relative.resolve())

    raise FileNotFoundError(
        f"PBSIM model not found: {model_name}. "
        f"Tried {model_path}, {repo_model}, and {repo_relative}."
    )


def _build_pbsim_options(quality_preset='default', qshmm_model='QSHMM-ONT-HQ.model',
                         accuracy_mean=None, difference_ratio=None, hp_del_bias=None,
                         pass_num=None, length_mean=None, length_sd=None, seed=None):
    if quality_preset not in PBSIM_QUALITY_PRESETS:
        raise ValueError(
            f"Unknown quality preset: {quality_preset}. "
            f"Expected one of {tuple(PBSIM_QUALITY_PRESETS)}."
        )

    options = {
        'qshmm_model': _resolve_model_path(qshmm_model),
        **PBSIM_QUALITY_PRESETS[quality_preset],
    }

    overrides = {
        'accuracy_mean': accuracy_mean,
        'difference_ratio': difference_ratio,
        'hp_del_bias': hp_del_bias,
        'pass_num': pass_num,
        'length_mean': length_mean,
        'length_sd': length_sd,
        'seed': seed,
    }
    for key, value in overrides.items():
        if value is not None:
            options[key] = value

    return options


def simulate_many(sequences, quality_preset='cleaner',
                  qshmm_model='QSHMM-ONT-HQ.model',
                  accuracy_mean=None, difference_ratio=None, hp_del_bias=None,
                  pass_num=None, length_mean=None, length_sd=None, seed=None):
    makedirs('temp', exist_ok=True)
    # Create a temporary FASTA file for the sequence
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False, dir='temp') as tmp_fasta:
        fasta_filename = tmp_fasta.name
        for idx, sequence in enumerate(sequences):
            # Write the sequence to the FASTA file
            tmp_fasta.write(f">{idx}\n")
            tmp_fasta.write(f"{sequence}\n")

    sim_options = _build_pbsim_options(
        quality_preset=quality_preset,
        qshmm_model=qshmm_model,
        accuracy_mean=accuracy_mean,
        difference_ratio=difference_ratio,
        hp_del_bias=hp_del_bias,
        pass_num=pass_num,
        length_mean=length_mean,
        length_sd=length_sd,
        seed=seed,
    )

    prefix = fasta_filename.replace('.fasta', '')
    pbsim_command = [
        'pbsim',
        '--strategy', 'templ',
        '--method', 'qshmm',
        '--qshmm', sim_options['qshmm_model'],
        '--template', fasta_filename,
        '--prefix', prefix,
        '--depth', '1',
    ]
    if 'accuracy_mean' in sim_options:
        pbsim_command.extend(['--accuracy-mean', str(sim_options['accuracy_mean'])])
    if 'difference_ratio' in sim_options:
        pbsim_command.extend(['--difference-ratio', str(sim_options['difference_ratio'])])
    if 'hp_del_bias' in sim_options:
        pbsim_command.extend(['--hp-del-bias', str(sim_options['hp_del_bias'])])
    if 'pass_num' in sim_options:
        pbsim_command.extend(['--pass-num', str(sim_options['pass_num'])])
    if 'length_mean' in sim_options:
        pbsim_command.extend(['--length-mean', str(sim_options['length_mean'])])
    if 'length_sd' in sim_options:
        pbsim_command.extend(['--length-sd', str(sim_options['length_sd'])])
    if 'seed' in sim_options:
        pbsim_command.extend(['--seed', str(sim_options['seed'])])
    subprocess.run(pbsim_command, capture_output=True, cwd='./', text=True, check=True)

    try:
        fastq_filename = f"{prefix}.fastq"  # correct expected filename

        # Read the simulated FASTQ output
        with open(fastq_filename, 'rt') as fastq_file:  # 'rt' = read text mode
            lines = fastq_file.readlines()
            sequences = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 1) % 4) == 0]
            qualities = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 3) % 4) == 0]

    except :
        fastq_filename = f"{prefix}.fq.gz"  # correct expected filename

        # Read the simulated FASTQ output
        with gzip.open(fastq_filename, 'rt') as fastq_file:  # 'rt' = read text mode
            lines = fastq_file.readlines()
            sequences = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 1) % 4) == 0]
            qualities = [seq.rstrip("\n") for i, seq in enumerate(lines) if ((i - 3) % 4) == 0]

    # Clean up the temporary FASTA file
    remove(fasta_filename)
    remove(fastq_filename)
    for suffix in ('.maf', '.ref'):
        extra = f"{prefix}{suffix}"
        if path.exists(extra):
            remove(extra)

    return sequences, qualities
