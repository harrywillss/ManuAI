"""ManūAI — New Zealand bird sound classification.

Shared, importable pipeline code. Notebooks and the future app should import from
here so that feature extraction, labels and splits have exactly one definition.

Modules
-------
config      paths and ManuAI.config handling
labels      class set (types.yaml) and eBird-code → class mapping
features    audio loading, segmentation and AST-compatible Kaldi fbank features
build_index archive metadata → data/index.csv with recording groups and a grouped split
preprocess  data/index.csv → float16 feature cache (data/features/)
data        datasets, sampler, SpecAugment and collate functions
models      AST / ViT construction with LoRA
metrics     segment- and recording-level metrics
train       training CLI (python -m manuai.train)
"""

__version__ = "2.1.0"
