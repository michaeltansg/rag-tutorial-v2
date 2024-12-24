from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(document_paths=['data/monopoly.pdf'])

dataset = EvaluationDataset(goldens=goldens)

print(dataset)