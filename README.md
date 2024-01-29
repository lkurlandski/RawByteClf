# RawByteClf

## Preprocessing

- Ensure all raw binaries and VirusTotal reports are in the correct directory on disk.
- Run `prepare` script
- Run `label` script
- Run `encode` script
- Run `split` script

## TODO

- Encode datasets during preprocessing.
- Craft train, test, and validation sets.
- Save predictions on each epoch (for evaluating accuracy of low-resource classes)

pip install "ray[tune]"==2.6.3
pip install bayesian-optimization
pip install hyperopt

pip install ninja

## Useful

Memory analysis:
	- mprof run python {SCRIPT.py}
	- mprof plot --output={PLOT.png}

Time analysis
	- python -m cProfile -o {STATS.pstats} {SCRIPT.py}
	- gprof2dot --colour-nodes-by-selftime -f pstats {STATS.pstats} | dot -Tpng -o {PLOT.png}
