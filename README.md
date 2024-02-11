# Academic Network Project
(TODO)
## ANP Core
`anp_core` serves as the backbone for handling dataset management, constructing and expanding the academic infosphere, and providing essential utilities for the Academic Network Project (ANP). It encapsulates functionalities for dataset loading, infosphere assembly, expansion, and various data processing tasks crucial for ANP analysis and exploration.


The idea is to use it in the following order:
- Parsing the Aminer dataset (imported into PYG).
- Generating the infosphere through `anp_infosphere_creation` (enabling parallel generation by dividing the infosphere into parts and utilizing multiple processes).
- Expansion of each part through `anp_infosphere_expansion_caller` (calling `anp_expansion`).
- Merging the parts via `anp_infosphere_builder`.

(This procedure will likely be revised in the future to make it entirely automated.)

## ANP NN
`anp_nn` contains models for predictive tasks using Graph Neural Networks (GNNs). Currently, it focuses on co-author prediction and a preliminary aspect of next-topic prediction within the Academic Network Project (ANP). These models leverage graph-based learning to forecast collaboration patterns among authors and anticipate future research topics within the academic sphere.

## Seedgraph backup
`seedgraph_backup` contains backup generated from `anp_infosphere_creation`
