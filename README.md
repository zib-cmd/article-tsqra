# Tensor SQRA
Accompanying code to the article "Tensor-SqRA: Modeling the Transition Rates of Interacting Molecular Systems in terms of Potential Energies" [1]

## Contents
- `julia/tsqra-example.jl`: contains the Julia implementation from Appendix B, specifically the tSqRA for arbitrary lower-order potentials and the banded computation of `apply_A`.
- `notebooks`: contains the python notebooks to generate the data from the article.

See also https://github.com/axsk/TSQRA.jl for the WIP code, also providing the application to a 9-dimensional pentane with eigenfunction computation.

## References
- [1] A. Sikorski, A. Niknejad, M. Weber, L. Donati. Tensor-SqRA: Modeling the Transition Rates of Interacting Molecular Systems in terms of Potential Energies (2023). https://arxiv.org/abs/2311.09779 