from __future__ import annotations

import nbformat as nbf


def _path_bootstrap_cell() -> str:
    return (
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "\n"
        "def _find_repo_root() -> Path:\n"
        "    candidates = [Path.cwd(), *Path.cwd().parents]\n"
        "    for path in candidates:\n"
        "        if (path / 'pyproject.toml').exists() and (path / 'src' / 'cytof_archetypes').exists():\n"
        "            return path\n"
        "    fallback = Path('/Users/ronguy/Dropbox/Work/CyTOF/Experiments/ProbAE_Deconv')\n"
        "    if (fallback / 'src' / 'cytof_archetypes').exists():\n"
        "        return fallback\n"
        "    raise RuntimeError('Could not locate repository root containing src/cytof_archetypes')\n"
        "\n"
        "REPO_ROOT = _find_repo_root()\n"
        "SRC_DIR = REPO_ROOT / 'src'\n"
        "def _resolve_out_dir() -> Path:\n"
        "    env = os.environ.get('CYTOF_SUITE_OUTPUT_DIR')\n"
        "    if env:\n"
        "        return Path(env)\n"
        "    cfg_path = REPO_ROOT / 'configs' / 'experiment_suite.yaml'\n"
        "    if cfg_path.exists():\n"
        "        try:\n"
        "            import yaml\n"
        "            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}\n"
        "            out_raw = cfg.get('output_dir')\n"
        "            if out_raw:\n"
        "                out_path = Path(out_raw)\n"
        "                return out_path if out_path.is_absolute() else REPO_ROOT / out_path\n"
        "        except Exception:\n"
        "            pass\n"
        "    return REPO_ROOT / 'outputs' / 'experiment_suite'\n"
        "\n"
        "OUT_DIR = _resolve_out_dir()\n"
        "\n"
        "def _artifact_exists(path: Path) -> bool:\n"
        "    if path.exists():\n"
        "        return True\n"
        "    print(f'Missing artifact: {path}')\n"
        "    print('Run the suite first: python scripts/run_experiment_suite.py --config configs/experiment_suite.yaml')\n"
        "    return False\n"
        "if str(SRC_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC_DIR))\n"
        "print('Repo root:', REPO_ROOT)\n"
        "print('Using src dir:', SRC_DIR)\n"
        "print('Using suite output dir:', OUT_DIR)\n"
    )


def _mk_notebook(title: str, intro: str, code_cells: list[str]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(f"# {title}"),
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_code_cell(_path_bootstrap_cell()),
    ]
    cells.extend([nbf.v4.new_code_cell(code) for code in code_cells])
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


def notebook_00_dataset_overview() -> nbf.NotebookNode:
    return _mk_notebook(
        title="00 Dataset Overview",
        intro="Dataset summary, class counts, marker distributions, and split sanity checks.",
        code_cells=[
            "from pathlib import Path\nimport pandas as pd\n"
            "from cytof_archetypes.datasets import load_dataset_bundle\n"
            "dataset_cfg = {\n"
            "    'name': 'levine32',\n"
            "    'input_path': str(REPO_ROOT / 'data' / 'levine32_processed.h5ad'),\n"
            "    'label_column': 'label',\n"
            "    'cell_id_column': 'cell_id',\n"
            "    'val_fraction': 0.15,\n"
            "    'test_fraction': 0.15,\n"
            "}\n"
            "bundle = load_dataset_bundle(dataset_cfg, seed=42)\n"
            "print('markers:', len(bundle.markers))\n"
            "print('train/val/test sizes:', len(bundle.train.x), len(bundle.val.x), len(bundle.test.x))",
            "if bundle.train.labels is not None:\n"
            "    print('train label counts')\n"
            "    print(pd.Series(bundle.train.labels).value_counts())\n"
            "if bundle.val.labels is not None:\n"
            "    print('val label counts')\n"
            "    print(pd.Series(bundle.val.labels).value_counts())",
            "split_manifest = OUT_DIR / 'reports' / 'split_manifest.csv'\n"
            "if split_manifest.exists():\n"
            "    manifest = pd.read_csv(split_manifest)\n"
            "    display(manifest.head())\n"
            "    print(manifest['split'].value_counts())",
        ],
    )


def notebook_01_fit_vs_complexity() -> nbf.NotebookNode:
    return _mk_notebook(
        title="01 Fit vs Complexity",
        intro="Compare reconstruction metrics across methods and K/latent dimensions.",
        code_cells=[
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "table_path = OUT_DIR / 'tables' / 'fit_vs_complexity_summary.csv'\n"
            "if _artifact_exists(table_path):\n"
            "    table = pd.read_csv(table_path)\n"
            "    display(table.head())\n"
            "else:\n"
            "    table = None",
            "if table is not None:\n"
            "    summary = table.groupby(['method', 'k_or_latent_dim'], as_index=False)[['val_mse','test_mse','test_nll']].mean()\n"
            "    display(summary.sort_values(['method','k_or_latent_dim']).head(20))",
            "for fig in ['reconstruction_vs_k.png', 'nll_vs_k.png', 'parameter_count_vs_validation_error.png']:\n"
            "    p = OUT_DIR / 'plots' / fig\n"
            "    print(p, 'exists=', p.exists())",
        ],
    )


def notebook_02_deconvolution_quality() -> nbf.NotebookNode:
    return _mk_notebook(
        title="02 Deconvolution Quality",
        intro="Inspect component weights, entropy, and class-component structure.",
        code_cells=[
            "import pandas as pd\n"
            "deconv_path = OUT_DIR / 'tables' / 'deconvolution_quality_summary.csv'\n"
            "class_path = OUT_DIR / 'tables' / 'class_component_means.csv'\n"
            "entropy_path = OUT_DIR / 'tables' / 'per_cell_weight_entropy.csv'\n"
            "if _artifact_exists(deconv_path):\n"
            "    deconv = pd.read_csv(deconv_path)\n"
            "    display(deconv.sort_values(['method','k','seed']).head(20))",
            "if _artifact_exists(class_path):\n"
            "    class_means = pd.read_csv(class_path)\n"
            "    display(class_means.head())\n"
            "if _artifact_exists(entropy_path):\n"
            "    entropy = pd.read_csv(entropy_path)\n"
            "    display(entropy.groupby('method')['weight_entropy'].describe())",
        ],
    )


def notebook_03_component_biology() -> nbf.NotebookNode:
    return _mk_notebook(
        title="03 Component Biology",
        intro="Marker programs, archetype means/variances, and enrichment summaries.",
        code_cells=[
            "import pandas as pd\n"
            "profiles_path = OUT_DIR / 'tables' / 'component_marker_profiles.csv'\n"
            "enrichment_path = OUT_DIR / 'tables' / 'component_marker_enrichment.csv'\n"
            "top_path = OUT_DIR / 'tables' / 'component_top_markers.csv'\n"
            "if _artifact_exists(profiles_path):\n"
            "    profiles = pd.read_csv(profiles_path)\n"
            "    display(profiles.head())\n"
            "if _artifact_exists(enrichment_path):\n"
            "    enrichment = pd.read_csv(enrichment_path)\n"
            "    display(enrichment.head())\n"
            "if _artifact_exists(top_path):\n"
            "    top = pd.read_csv(top_path)\n"
            "    display(top.sort_values(['method','k','component','rank']).head(40))",
        ],
    )


def notebook_04_deterministic_vs_probabilistic() -> nbf.NotebookNode:
    return _mk_notebook(
        title="04 Deterministic vs Probabilistic",
        intro="Direct comparison of deterministic and probabilistic archetypal autoencoders.",
        code_cells=[
            "import pandas as pd\n"
            "comp_path = OUT_DIR / 'tables' / 'deterministic_vs_probabilistic_summary.csv'\n"
            "if _artifact_exists(comp_path):\n"
            "    comp = pd.read_csv(comp_path)\n"
            "    display(comp.sort_values(['k','seed']).head(20))\n"
            "else:\n"
            "    comp = None",
            "if comp is not None:\n"
            "    cols = ['det_test_mse','prob_test_mse','det_rare_class_error','prob_rare_class_error','wilcoxon_p','wilcoxon_q']\n"
            "    display(comp[cols].describe())\n"
            "    print('Probability model wins (lower test MSE):', (comp['prob_test_mse'] < comp['det_test_mse']).mean())",
        ],
    )


def notebook_05_k_selection() -> nbf.NotebookNode:
    return _mk_notebook(
        title="05 K Selection",
        intro="Identify the smallest near-optimal K preserving fit and biological structure.",
        code_cells=[
            "import pandas as pd\n"
            "k_path = OUT_DIR / 'tables' / 'k_selection_summary.csv'\n"
            "if _artifact_exists(k_path):\n"
            "    k = pd.read_csv(k_path)\n"
            "    display(k.sort_values(['method','k']).head(30))\n"
            "    recommendations = k.groupby('method', as_index=False)['recommended_k'].first()\n"
            "    display(recommendations)\n"
            "else:\n"
            "    k = None",
            "from pathlib import Path\n"
            "report = OUT_DIR / 'reports' / 'k_selection_recommendation.md'\n"
            "print(report.read_text()[:1200]) if report.exists() else print('missing report')",
        ],
    )


def notebook_06_secondary_dataset_validation() -> nbf.NotebookNode:
    return _mk_notebook(
        title="06 Secondary Dataset Validation",
        intro="Optional generalization check on a second CyTOF dataset.",
        code_cells=[
            "from pathlib import Path\nimport pandas as pd\n"
            "table_path = OUT_DIR / 'tables' / 'secondary_dataset_summary.csv'\n"
            "if table_path.exists():\n"
            "    sec = pd.read_csv(table_path)\n"
            "    display(sec.head())\n"
            "else:\n"
            "    print('Secondary dataset was not run.')",
        ],
    )


def notebook_07_auxiliary_representation_models() -> nbf.NotebookNode:
    return _mk_notebook(
        title="07 Auxiliary Representation Models",
        intro="Optional supplementary comparisons (DeepSets/Transformer/LSTM placeholders).",
        code_cells=[
            "from pathlib import Path\nimport pandas as pd\n"
            "table_path = OUT_DIR / 'tables' / 'auxiliary_representation_models_summary.csv'\n"
            "if table_path.exists():\n"
            "    aux = pd.read_csv(table_path)\n"
            "    display(aux.head())\n"
            "else:\n"
            "    print('Auxiliary models were not run.')",
        ],
    )


def notebook_08_test_suite_runner() -> nbf.NotebookNode:
    return _mk_notebook(
        title="08 Test Suite Runner",
        intro=(
            "Run the full pytest suite with progress bars. "
            "Default mode runs one pytest invocation per test node for granular progress."
        ),
        code_cells=[
            "import subprocess\n"
            "import sys\n"
            "import time\n"
            "from pathlib import Path\n"
            "\n"
            "try:\n"
            "    from tqdm.auto import tqdm\n"
            "except Exception:\n"
            "    class tqdm:  # lightweight fallback when tqdm is unavailable\n"
            "        def __init__(self, iterable, total=None, desc='', unit='it'):\n"
            "            self.iterable = list(iterable)\n"
            "            self.total = len(self.iterable) if total is None else total\n"
            "            self.desc = desc\n"
            "            self.unit = unit\n"
            "            self.idx = 0\n"
            "            print(f'{self.desc}: 0/{self.total} {self.unit}')\n"
            "        def __iter__(self):\n"
            "            for item in self.iterable:\n"
            "                yield item\n"
            "                self.idx += 1\n"
            "                print(f'{self.desc}: {self.idx}/{self.total} {self.unit}', end='\\r')\n"
            "            print()\n"
            "        def set_postfix(self, **kwargs):\n"
            "            return None\n"
            "\n"
            "TEST_DIR = REPO_ROOT / 'tests'\n"
            "assert TEST_DIR.exists(), f'Missing tests directory: {TEST_DIR}'\n"
            "print('Test directory:', TEST_DIR)",
            "def collect_test_nodeids(repo_root: Path) -> list[str]:\n"
            "    cmd = [sys.executable, '-m', 'pytest', '--collect-only', '-q', str(repo_root / 'tests')]\n"
            "    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)\n"
            "    if proc.returncode != 0:\n"
            "        raise RuntimeError('Collection failed:\\n' + proc.stdout + '\\n' + proc.stderr)\n"
            "    nodeids = []\n"
            "    for line in proc.stdout.splitlines():\n"
            "        line = line.strip()\n"
            "        if '::' in line and line.startswith('tests/'):\n"
            "            nodeids.append(line)\n"
            "    return nodeids\n"
            "\n"
            "nodeids = collect_test_nodeids(REPO_ROOT)\n"
            "print(f'Collected {len(nodeids)} test nodes')\n"
            "nodeids[:10]",
            "results = []\n"
            "start = time.time()\n"
            "for nodeid in tqdm(nodeids, total=len(nodeids), desc='Pytest progress', unit='test'):\n"
            "    cmd = [sys.executable, '-m', 'pytest', '-q', nodeid]\n"
            "    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)\n"
            "    passed = (proc.returncode == 0)\n"
            "    results.append({\n"
            "        'nodeid': nodeid,\n"
            "        'status': 'PASS' if passed else 'FAIL',\n"
            "        'returncode': proc.returncode,\n"
            "        'stdout': proc.stdout,\n"
            "        'stderr': proc.stderr,\n"
            "    })\n"
            "elapsed = time.time() - start\n"
            "print(f'Completed in {elapsed:.2f} sec')",
            "import pandas as pd\n"
            "res_df = pd.DataFrame(results)\n"
            "display(res_df[['nodeid', 'status']])\n"
            "print(res_df['status'].value_counts())\n"
            "\n"
            "failed = res_df[res_df['status'] == 'FAIL']\n"
            "if len(failed):\n"
            "    print('\\nFailing tests and outputs:')\n"
            "    for row in failed.itertuples(index=False):\n"
            "        print('\\n' + '=' * 80)\n"
            "        print(row.nodeid)\n"
            "        print('-' * 80)\n"
            "        print(row.stdout)\n"
            "        if row.stderr:\n"
            "            print('[stderr]')\n"
            "            print(row.stderr)\n"
            "    raise AssertionError(f'{len(failed)} tests failed')\n"
            "else:\n"
            "    print('All tests passed.')",
            "output_path = REPO_ROOT / 'outputs' / 'test_suite' / 'pytest_progress_results.csv'\n"
            "output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "res_df.to_csv(output_path, index=False)\n"
            "print('Saved detailed results to', output_path)",
        ],
    )


def notebook_09_full_experiment_suite_runner() -> nbf.NotebookNode:
    return _mk_notebook(
        title="09 Full Experiment Suite Runner",
        intro=(
            "Run the complete experiment suite (all core methods, sweeps, and report generation) "
            "from inside a notebook."
        ),
        code_cells=[
            "from pathlib import Path\n"
            "import os\n"
            "import time\n"
            "\n"
            "from cytof_archetypes.experiments import load_suite_config, run_experiment_suite\n"
            "\n"
            "CONFIG_PATH = REPO_ROOT / 'configs' / 'experiment_suite.yaml'\n"
            "assert CONFIG_PATH.exists(), f'Missing config: {CONFIG_PATH}'\n"
            "print('Using config:', CONFIG_PATH)\n"
            "\n"
            "# Optional: override output dir without editing YAML.\n"
            "# os.environ['CYTOF_SUITE_OUTPUT_DIR'] = str(REPO_ROOT / 'outputs' / 'experiment_suite')",
            "cfg = load_suite_config(CONFIG_PATH)\n"
            "env_out = os.environ.get('CYTOF_SUITE_OUTPUT_DIR')\n"
            "if env_out:\n"
            "    cfg['output_dir'] = env_out\n"
            "\n"
            "# Speed controls for quick testing:\n"
            "DOWNSAMPLE_FACTOR = 1  # e.g. 5 keeps ~20% of cells, 10 keeps ~10%\n"
            "if DOWNSAMPLE_FACTOR and DOWNSAMPLE_FACTOR > 1:\n"
            "    cfg.setdefault('dataset', {})['downsample_factor'] = int(DOWNSAMPLE_FACTOR)\n"
            "else:\n"
            "    cfg.setdefault('dataset', {}).pop('downsample_factor', None)\n"
            "    cfg.setdefault('dataset', {}).pop('downsample_fraction', None)\n"
            "\n"
            "# Verbose execution prints and progress bars:\n"
            "cfg['show_progress'] = True\n"
            "cfg['show_run_logs'] = True\n"
            "NN_TRAINING_PROGRESS = True\n"
            "NN_TRAINING_PROGRESS_LEVEL = 'epoch'  # 'epoch' or 'batch'\n"
            "cfg['show_training_progress'] = bool(NN_TRAINING_PROGRESS)\n"
            "cfg['training_progress_level'] = str(NN_TRAINING_PROGRESS_LEVEL)\n"
            "cfg['training_progress_leave'] = False\n"
            "CPU_MULTIPROCESS_WORKERS = 1  # >1 parallelizes CPU baselines (nmf/classical) across processes\n"
            "cfg['cpu_multiprocessing_workers'] = int(CPU_MULTIPROCESS_WORKERS)\n"
            "cfg['cpu_parallel_methods'] = ['nmf', 'classical_archetypes']\n"
            "\n"
            "print('Resolved output_dir:', cfg['output_dir'])\n"
            "print('Resolved default device:', cfg.get('resolved_device', cfg.get('device')))\n"
            "for _m in ['deterministic_archetypal_ae', 'probabilistic_archetypal_ae', 'ae', 'vae']:\n"
            "    print(f\"  {_m}:\", cfg.get('methods', {}).get(_m, {}).get('device'))\n"
            "print('Methods:', list(cfg.get('methods', {}).keys()))\n"
            "print('Seeds:', cfg.get('seeds', []))\n"
            "print('K sweep:', cfg.get('sweeps', {}).get('k_values', []))\n"
            "print('Latent sweep:', cfg.get('sweeps', {}).get('latent_dims', []))\n"
            "print('Downsample factor:', cfg.get('dataset', {}).get('downsample_factor', 1))\n"
            "print('show_progress:', cfg.get('show_progress'), 'show_run_logs:', cfg.get('show_run_logs'))\n"
            "print('show_training_progress:', cfg.get('show_training_progress'), 'training_progress_level:', cfg.get('training_progress_level'))\n"
            "print('cpu_multiprocessing_workers:', cfg.get('cpu_multiprocessing_workers'), 'cpu_parallel_methods:', cfg.get('cpu_parallel_methods'))",
            "RUN_FULL = True  # set to False to skip execution\n"
            "if RUN_FULL:\n"
            "    t0 = time.time()\n"
            "    out_dir = run_experiment_suite(cfg)\n"
            "    dt = time.time() - t0\n"
            "    print(f'Completed full suite in {dt/60:.2f} minutes')\n"
            "    print('Output directory:', out_dir)\n"
            "else:\n"
            "    out_dir = Path(cfg['output_dir'])\n"
            "    print('Execution skipped. Expected output directory:', out_dir)",
            "import pandas as pd\n"
            "\n"
            "out_dir = Path(cfg['output_dir'])\n"
            "tables_dir = out_dir / 'tables'\n"
            "reports_dir = out_dir / 'reports'\n"
            "plots_dir = out_dir / 'plots'\n"
            "\n"
            "required_tables = [\n"
            "    'fit_vs_complexity_summary.csv',\n"
            "    'deconvolution_quality_summary.csv',\n"
            "    'component_marker_profiles.csv',\n"
            "    'component_marker_enrichment.csv',\n"
            "    'deterministic_vs_probabilistic_summary.csv',\n"
            "    'per_class_method_metrics.csv',\n"
            "    'fit_vs_interpretability.csv',\n"
            "    'k_selection_summary.csv',\n"
            "    'class_component_means.csv',\n"
            "    'per_cell_weight_entropy.csv',\n"
            "]\n"
            "\n"
            "status = []\n"
            "for name in required_tables:\n"
            "    path = tables_dir / name\n"
            "    status.append({'table': name, 'exists': path.exists(), 'path': str(path)})\n"
            "status_df = pd.DataFrame(status)\n"
            "display(status_df)\n"
            "print('All required tables present:', bool(status_df['exists'].all()))",
            "manifest_path = reports_dir / 'artifact_manifest.json'\n"
            "if manifest_path.exists():\n"
            "    import json\n"
            "    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n"
            "    print('Manifest summary keys:', sorted(manifest.keys()))\n"
            "    print('Number of tables:', len(manifest.get('tables', [])))\n"
            "    print('Number of plots:', len(manifest.get('plots', [])))\n"
            "else:\n"
            "    print('Missing artifact manifest:', manifest_path)",
            "k_path = tables_dir / 'k_selection_summary.csv'\n"
            "if k_path.exists():\n"
            "    k_df = pd.read_csv(k_path)\n"
            "    display(k_df.sort_values(['method', 'k']).head(30))\n"
            "else:\n"
            "    print('Missing:', k_path)",
        ],
    )
