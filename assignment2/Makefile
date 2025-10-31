install:
	uv sync

download:
	uv run python -m scripts.data_download

baseline_svm:
	uv run python -m scripts.baseline_svm

# Run PCA analysis
pca:
	uv run python -m scripts.pca_analysis

# Run self-training experiments
self-training:
	uv run python -m scripts.self_training

# Run label propagation experiments
label-propagation:
	uv run python -m scripts.label_propagation

# Run both semi-supervised experiments
semi: self-training label-propagation

# Generate plots from results
plots:
	uv run python -m scripts.plot_results

# Test all scripts
test:
	@echo "Testing baseline..."
	uv run python -m scripts.baseline_svm
	@echo "Testing PCA analysis..."
	uv run python -m scripts.pca_analysis
	@echo "Testing self-training..."
	uv run python -m scripts.self_training --p 0.01
	@echo "Testing label propagation..."
	uv run python -m scripts.label_propagation --p 0.01
	@echo "All tests passed!"

# Run linting
lint:
	uv run ruff check .

# Format code
format:
	uv run ruff format .

# Clean generated files
clean:
	rm -f baseline.json
	rm -f self_training_results.json
	rm -f self_training_summary.json
	rm -f label_propagation_results.json
	rm -f label_propagation_summary.json
	rm -f semi_results.json
	rm -f semi_summary.json
	rm -rf figures/*.png
	rm -rf figures/*.gif
	rm -rf .cache/figures/*.png
	rm -rf .cache/figures/*/
	@echo "Cleaned generated files"

# Run everything in sequence
all: install download baseline_svm pca semi plots
	@echo "All experiments and plots completed!"

# Run experiments and generate plots
complete: all
