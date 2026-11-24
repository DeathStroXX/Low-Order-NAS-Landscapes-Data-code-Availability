# Reproduction archive -- see README.md.
#   make reproduce   rerun every analysis script, in dependency order
#   make sums        verify the archive against SHA256SUMS
PY ?= python3
A = analysis
.PHONY: reproduce sums
sums:
	sha256sum -c SHA256SUMS --quiet && echo "archive intact"
	cd large_input_data && sha256sum -c SHA256SUMS --quiet && echo "pod logs intact"
reproduce:
	cd $(A) && $(PY) s02_check_record.py
	cd $(A) && $(PY) s53_spectrum.py
	cd $(A) && $(PY) s59_ceiling.py
	cd $(A) && $(PY) s57_benchmarks.py
	cd $(A) && $(PY) s58_hardware.py
	cd $(A) && $(PY) s56_targets.py
	cd $(A) && $(PY) s14_learning_curves.py
	cd $(A) && $(PY) s10_fitness_provenance.py
	cd $(A) && $(PY) s11_proxy_fidelity.py
	cd $(A) && $(PY) s5_predictor_validation.py
	cd $(A) && $(PY) s72_censoring.py
	cd $(A) && $(PY) s16_surrogate.py
	cd $(A) && $(PY) s15_prescreen.py
	cd $(A) && $(PY) s27_signal_anatomy.py
	cd $(A) && $(PY) s25_transfer.py
	cd $(A) && $(PY) s35_nb201.py
	cd $(A) && $(PY) s36_regret.py
	-cd $(A) && $(PY) s43_zc_proxies.py   # needs a fetched third-party table; skipped with a message if absent
	-cd $(A) && $(PY) s44_zc_spaces.py   # needs a fetched third-party table; skipped with a message if absent
	-cd $(A) && $(PY) s45_encoding.py   # needs a fetched third-party table; skipped with a message if absent
	cd $(A) && $(PY) s17_halving.py
	cd $(A) && $(PY) s30_recipe.py
	cd $(A) && $(PY) s31_sitl.py
	cd $(A) && $(PY) s29_warmstart.py
	cd $(A) && $(PY) s73_supplement_extras.py
	cd $(A) && $(PY) s74_robustness.py
	cd $(A) && $(PY) s75_k_sensitivity.py
	cd $(A) && $(PY) s62_nb201_search.py
	-cd $(A) && $(PY) s63_extra_grids.py   # needs a fetched third-party table; skipped with a message if absent
	cd $(A) && $(PY) s68_learned_predictors.py
	cd $(A) && $(PY) s64_realization.py
	cd $(A) && $(PY) s65_monotone.py
	cd $(A) && $(PY) s66_dimension.py
	cd $(A) && $(PY) s50_decision.py
	cd $(A) && $(PY) s54_spectrum_tables.py
	cd $(A) && $(PY) s60_tables.py
	cd $(A) && $(PY) s46_tables.py
	cd $(A) && $(PY) s49_novelty.py
	cd $(A) && $(PY) s55_figures.py
	cd $(A) && $(PY) s61_figures.py
	cd $(A) && $(PY) s69_body_panels.py
	cd $(A) && $(PY) s71_merged_tables.py
