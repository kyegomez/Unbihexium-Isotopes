# Theoretical Stability Mechanisms of Unbihexium Isotopes in the Island of Stability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/10.xxxx/xxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

## 📑 Abstract

This repository contains the research paper and supplementary materials for "Theoretical Stability Mechanisms of Unbihexium Isotopes in the Island of Stability," a comprehensive theoretical investigation into the nuclear physics of Element 126 (unbihexium) and its potentially stable isotopes. The research focuses particularly on isotope $^{354}$Ubh, which some calculations predict could have a half-life on the order of 100 years if closed shell effects provide sufficient stabilization. Through rigorous mathematical modeling and analysis of competing theories about magic numbers in superheavy elements, this work contributes to our understanding of the theoretical limits of nuclear stability.

## 🌟 Key Research Objectives

1. **Explore theoretical nuclear shell configurations** that may lead to enhanced stability for superheavy elements, with specific focus on unbihexium isotopes
2. **Analyze the stability mechanisms of isotope $^{354}$Ubh** which calculations suggest could have remarkable longevity compared to other superheavy elements
3. **Evaluate competing theories about "magic numbers"** for superheavy elements and their implications for nuclear shell structure
4. **Quantify the impact of relativistic effects** on superheavy element stability
5. **Assess experimental prospects** for synthesizing unbihexium


# Experiments

Run the experiments with our script

```bash
https://github.com/kyegomez/Unbihexium-Isotopes.git
cd Unbihexium-Isotopes
pip3 install -r requirements.txt
python3 experiments.py

```

## 🔬 Theoretical Background

### The Nuclear Shell Model

The paper provides a detailed explanation of the nuclear shell model, which describes atomic nuclei as consisting of nucleons (protons and neutrons) arranged in energy levels or "shells." Originally developed by Maria Goeppert Mayer and J. Hans D. Jensen in the late 1940s, this model explains the enhanced stability observed in nuclei with certain "magic numbers" of protons or neutrons.

The mathematical foundation of the shell model is presented, including:
- The Woods-Saxon potential: $V(r) = -\frac{V_0}{1 + \exp{\frac{r-R}{a}}}$
- Spin-orbit coupling: $V_{SO}(r) = V_{SO}(r) \mathbf{l} \cdot \mathbf{s}$
- Single-particle energy levels and their arrangement

### The Island of Stability

The concept of an "island of stability" was first proposed in the 1960s, predicting a region in the chart of nuclides where superheavy elements might possess significantly longer half-lives due to nuclear shell effects. The paper explores various theoretical predictions about the location of this island, which is generally thought to center near elements with proton numbers Z = 114, 120, or 126, and neutron numbers N = 184 or 228.

Key aspects covered include:
- Historical development of the island of stability hypothesis
- Theoretical mechanisms that create stability islands
- Current experimental evidence supporting the concept

### Unbihexium: Element 126

Unbihexium (Ubh), with atomic number 126, represents a particularly interesting candidate for enhanced stability if the proton number 126 corresponds to a closed shell configuration. The paper comprehensively examines:
- Theoretical predictions for unbihexium's nuclear structure
- Potential isotopes spanning from approximately A = 290 to A = 354
- Decay modes and predicted half-lives
- Relativistic effects on unbihexium's electronic and nuclear structure

### Magic Numbers in Superheavy Elements

The conventional magic numbers in nuclear physics (2, 8, 20, 28, 50, 82, and 126 for neutrons) may not directly extrapolate to superheavy elements due to relativistic effects and increased Coulomb interactions. The paper analyzes:
- Competing predictions for proton magic numbers (Z = 114, 120, or 126)
- Evidence for neutron magic numbers at N = 184 and N = 228
- Shell structure calculations using various theoretical approaches
- The impact of nuclear deformation on magic number predictions

## 🧮 Methodology

The paper employs several sophisticated theoretical approaches to model superheavy nuclei:

### Macroscopic-Microscopic Models

These models combine a macroscopic liquid-drop component with microscopic shell corrections:

$E_{total} = E_{macro} + E_{micro}$

Where the macroscopic energy includes terms for volume, surface, Coulomb, and symmetry energy:

$E_{macro} = a_v A - a_s A^{2/3} - a_c \frac{Z^2}{A^{1/3}} - a_{sym}\frac{(N-Z)^2}{A}$

### Self-Consistent Mean-Field Methods

Both non-relativistic and relativistic variants of self-consistent mean-field methods are utilized, including:
- Hartree-Fock-Bogoliubov (HFB) calculations with Skyrme or Gogny forces
- Relativistic Mean-Field (RMF) theory based on a Lagrangian density with nucleons interacting via meson exchange

### Relativistic Effects

The paper incorporates full relativistic treatments of nuclear structure using the Dirac equation framework:

$[c\boldsymbol{\alpha} \cdot \mathbf{p} + \beta mc^2 + V(\mathbf{r})]\psi = E\psi$

### Decay Rate Calculations

Rigorous quantum mechanical calculations of alpha decay and spontaneous fission rates, including:
- Quantum tunneling approach for alpha decay
- Action integral calculations for spontaneous fission barriers
- Competition between different decay modes

## 🔍 Key Propositions

### Stability of $^{354}$Ubh

The research provides detailed calculations suggesting that if both Z = 126 and N = 228 represent strong shell closures, isotope $^{354}$Ubh could have a half-life on the order of 100 years. This represents remarkable stability for a superheavy element and would place it firmly within the theoretical island of stability.

Key factors contributing to this potential stability include:
- Double shell closure effects (if both proton and neutron numbers are "magic")
- Enhanced binding energy from shell corrections (~10-15 MeV)
- Increased fission barriers compared to neighboring nuclides

However, significant uncertainties remain, with competing models yielding different predictions about the strength of the Z = 126 shell closure.

### Competing Magic Number Theories

The research presents a systematic analysis of different theoretical predictions for magic numbers in superheavy elements:
- Z = 114 is supported by many macroscopic-microscopic calculations
- Z = 120 emerges from certain self-consistent mean-field approaches
- Z = 126 is predicted by some relativistic calculations and simple shell model extrapolations

Each prediction is analyzed in terms of the underlying theoretical assumptions and their implications for unbihexium stability.

### Relativistic Effects

The paper demonstrates that relativistic effects play a crucial role in determining the shell structure of superheavy elements through:
- Enhanced spin-orbit splitting
- Modification of the nuclear central potential
- Changes in pairing interaction strengths

These effects significantly impact predictions of magic numbers and stability regions in superheavy elements.

### Deformation and Shape Effects

While spherical shapes are typically associated with magic numbers, the research shows that nuclear deformation can also contribute to stability in superheavy elements. For unbihexium isotopes, calculations suggest:
- Potential energy minima at various deformation parameters
- Shape coexistence in certain isotopes
- Deformation-driven shell effects that may enhance stability

### Experimental Synthesis Prospects

The research evaluates several potential approaches for synthesizing unbihexium:
- Hot fusion reactions using actinide targets and heavy projectiles
- Multinucleon transfer reactions in collisions of heavy nuclei
- Symmetric or nearly symmetric fusion reactions

However, all approaches face significant challenges, with predicted production cross-sections on the order of femtobarns (10^-15 barns) or less.

## 🔮 Theoretical Uncertainties

A detailed analysis of theoretical uncertainties in the predictions is presented:

1. **Model Dependence** - Different theoretical frameworks yield varying predictions for shell closures and stability
2. **Parameter Sensitivity** - Results depend significantly on the specific parameterizations used in calculations
3. **Extrapolation Errors** - Extending models beyond the region of known nuclei introduces additional uncertainties
4. **Computational Limitations** - Approximate treatments of complex many-body correlations affect accuracy

These uncertainties are quantified through systematic comparison of different theoretical approaches and sensitivity analyses.

## 💡 Experimental Validation

The paper includes a detailed comparison of theoretical predictions with experimental data for known superheavy elements (Z=113-118):
- Half-life measurements and their agreement with different theoretical models
- Decay mode branching ratios and their implications for nuclear structure
- Trends in alpha-decay energies and their relationship to shell effects

This analysis provides a benchmark for evaluating the reliability of extrapolations to unbihexium.

## 🔬 Potential Applications

If long-lived unbihexium isotopes could be synthesized, even in minute quantities, several applications and implications are discussed:

1. **Fundamental Nuclear Physics** - Testing theoretical models at the extremes of nuclear existence
2. **Nuclear Structure Studies** - Providing unique insights into shell effects and the limits of nuclear stability
3. **Superheavy Element Chemistry** - Exploring relativistic effects on chemical properties
4. **Nucleosynthesis Understanding** - Informing models of element formation in extreme astrophysical environments

## 🔭 Future Research Directions

The paper outlines several promising avenues for future research:

### Theoretical Developments
1. Improved treatments of relativistic effects in nuclear structure models
2. More accurate calculations of pairing correlations in superheavy nuclei
3. Refined predictions of alpha-decay and spontaneous fission rates
4. Detailed exploration of potential energy surfaces, including deformation effects

### Experimental Approaches
1. Novel reaction mechanisms for synthesizing increasingly heavy elements
2. Advanced detection techniques for identifying superheavy nuclei produced in minute quantities
3. Incremental progress toward heavier elements, providing constraints on theoretical models
4. Indirect studies of shell effects in the heaviest known elements

## 📖 How to Cite This Work

If you use or reference this research, please cite as follows:

```bibtex
@article{Gomez2025Unbihexium,
  author = {Gomez, Kye and Contributors},
  title = {Theoretical Stability Mechanisms of Unbihexium Isotopes in the Island of Stability},
  journal = {Journal of Superheavy Element Research},
  url = {https://github.com/kyegomez/Unbihexium-Isotopes}
}
```

## 📚 References

The paper includes a comprehensive bibliography of over 50 references covering:
- Foundational works on nuclear shell model theory
- Historical development of the island of stability concept
- Modern theoretical approaches to superheavy element structure
- Experimental studies of the heaviest known elements
- Computational methods for predicting nuclear properties

## 📄 License

This research is released under the MIT License. See the `LICENSE` file for details.

## 👥 Contributors

- **Kye Gomez** - Principal Investigator
- Additional contributors listed in the paper

## 📩 Contact

For questions or collaborations related to this research, please open an issue on this repository or contact the principal investigator directly.
