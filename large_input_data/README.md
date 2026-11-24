# Pod logs of the 34 search runs

The terminal (pod) log of each of the 34 FractalNet search runs, one file per run, unmodified.
They total about 164 MB, too large for the Overleaf project, so they sit here, outside `logs/`,
and are left out of the Overleaf copy.

```
large_input_data/pod_logs/<arm>/<dataset>/seed<N>_<uuid>/<timestamp>_<dataset>_<arm>_Pod_logs.log
```

`<arm>` is `baseline`, `frozen`, `deepseek`, `mistral` or `qwen`; `<dataset>` is `cifar10` or
`cifar100`; the run folders match those under `logs/`. `analysis/common.py` finds them here.

Only the scripts that read the terminal output need them: `analysis/s14_learning_curves.py`
(which writes `data/learning_curves.csv`) and `analysis/s10_fitness_provenance.py`. Building the
documents and `make check` do not.

Verify after copying: `cd large_input_data && sha256sum -c SHA256SUMS`.
