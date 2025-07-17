# IRT2 LLM Prompting

Using chatbots to solve open world knowledge graph completion. This repository is used to document the state of the research code used for the publication of the paper [Re-ranking with LLMs for Open-World Knowledge Graph Completion](https://example.com).

- [See detailed experiment results here](https://docs.google.com/spreadsheets/d/1794pSMUGz6si1_FswPXciauxJSQRjqkLJL4WH2JO_2s)
- [Download the raw experiment results and model outputs here](https://lavis.cs.hs-rm.de/storage/ilp2/ilp2-experiment-results.tgz)


## Installation

We recommend using [pyenv]([poetry](https://github.com/pyenv/pyenv)+https://python-poetry.org/) for installation.

```console
pyenv install 3.11
pyenv global 3.11
poetry install
poetry run ilp
```

You can find the source code in `./ilp` and all configurations (including legacy) in `conf/`.


## Getting Started

```console
 $ poetry run ilp

 Usage: ilp [OPTIONS] COMMAND [ARGS]...

 Use ilp from the command line.

╭─ Options ─────────────────────────────────────────────────────╮
│ --quiet  -q    suppress console output                        │
│ --debug  -d    activate debug mode (drop into pudb on error)  │
│ --help         Show this message and exit.                    │
╰───────────────────────────────────────────────────────────────╯
```

To reproduce the experiments of the paper, see the `scripts/` directory. Calls to `ilp` to run the experiments which have been conducted for the final paper results are documented in `scripts/all-experiments.sh`. If you want to have the flexibility and automation we worked with during development, you can execute the `*.fish` files. The [fish shell](https://fishshell.com/) is required to execute these scripts.  There, for the IRT2 and BLP datasets, experiments are started using `scripts/exp-irt-full.fish` and `scripts/exp-blp-full.fish`. These entry points source the outher files in the directory to configure the experiment runs. However, these are only used to properly configure the `ilp` entry point shown above like documented in `scripts/all-experiments.sh`.


## Cite

If you find our work useful, please consider giving us a cite. You can also always contact [Felix Hamann](https://github.com/kantholtz) for any comments or questions!

```
TBA
```
