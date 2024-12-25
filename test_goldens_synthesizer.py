from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    ContextConstructionConfig,
    FiltrationConfig,
    EvolutionConfig,
)
from deepeval.synthesizer.types import Evolution
from pandas.core.frame import DataFrame
from dotenv import load_dotenv
import os

load_dotenv()

filtration_config = FiltrationConfig(
    max_quality_retries=4,
)

evolution_config = EvolutionConfig(
    num_evolutions=1,
    evolutions={
        Evolution.REASONING: 1,
        Evolution.MULTICONTEXT: 4,
        # Evolution.CONCRETIZING: 1 / 7,
        # Evolution.CONSTRAINED: 1 / 7,
        # Evolution.COMPARATIVE: 1 / 7,
        # Evolution.HYPOTHETICAL: 1 / 7,
        # Evolution.IN_BREADTH: 1 / 7,
    },
)

synthesizer = Synthesizer(filtration_config=filtration_config, evolution_config=evolution_config)


goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        "test/goldens/HR-Guide_-Policy-and-Procedure-Template.pdf",
        "test/goldens/human-rights-policy-en.pdf",
        "test/goldens/pillar-3-disclosure-dec-23-en.pdf",
    ],
    include_expected_output=True,
    context_construction_config=ContextConstructionConfig(
        embedder=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        chunk_size=800,
        chunk_overlap=80,
        max_contexts_per_document=5
    ),
)

# Access evolutions through the DataFrame
goldens_dataframe = synthesizer.to_pandas()
quality_scores: DataFrame = goldens_dataframe.head()

print(quality_scores)

questions = [golden.input for golden in goldens]

unique_questions = list(set(questions))  # Remove duplicate questions
final_questions = unique_questions[:5]  # Select the top 5 (or more if available)

print(questions)
print("================")
print(final_questions)
