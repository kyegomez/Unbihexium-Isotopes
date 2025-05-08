import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NuclearShellModel:
    """
    Implementation of simplified nuclear shell model calculations
    for superheavy elements with a focus on unbihexium isotopes.
    """

    # Magic numbers for protons and neutrons
    PROTON_MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]
    NEUTRON_MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126, 184, 228]

    # Constants for calculations
    VOLUME_COEFF = 15.8  # MeV
    SURFACE_COEFF = 18.3  # MeV
    COULOMB_COEFF = 0.714  # MeV
    ASYMMETRY_COEFF = 23.2  # MeV

    def __init__(self, Z, N):
        """
        Initialize the model with proton and neutron numbers.

        Args:
            Z (int): Number of protons
            N (int): Number of neutrons
        """
        self.Z = Z
        self.N = N
        self.A = Z + N

    def liquid_drop_energy(self):
        """
        Calculate binding energy using the liquid drop model.

        Returns:
            float: Binding energy in MeV
        """
        # Volume term
        E_vol = self.VOLUME_COEFF * self.A

        # Surface term
        E_surf = -self.SURFACE_COEFF * (self.A ** (2 / 3))

        # Coulomb term
        E_coul = -self.COULOMB_COEFF * (self.Z**2 / self.A ** (1 / 3))

        # Asymmetry term
        E_asym = -self.ASYMMETRY_COEFF * (
            (self.N - self.Z) ** 2 / self.A
        )

        return E_vol + E_surf + E_coul + E_asym

    def shell_correction(self, model="default"):
        """
        Calculate shell correction energy based on proximity to magic numbers.

        Args:
            model (str): Model to use for shell corrections
                "default": Simple distance-based model
                "relativistic": Includes relativistic effects

        Returns:
            float: Shell correction energy in MeV
        """
        if model == "default":
            # Simple model: shell correction is stronger near magic numbers
            proton_factor = min(
                abs(self.Z - magic)
                for magic in self.PROTON_MAGIC_NUMBERS
            )
            neutron_factor = min(
                abs(self.N - magic)
                for magic in self.NEUTRON_MAGIC_NUMBERS
            )

            # Scale factors - closer to magic numbers means stronger correction
            proton_correction = -5.0 * np.exp(-0.05 * proton_factor)
            neutron_correction = -5.0 * np.exp(-0.05 * neutron_factor)

            # Double magic gets extra stabilization
            if (
                self.Z in self.PROTON_MAGIC_NUMBERS
                and self.N in self.NEUTRON_MAGIC_NUMBERS
            ):
                return proton_correction + neutron_correction - 5.0
            else:
                return proton_correction + neutron_correction

        elif model == "relativistic":
            # More sophisticated model including relativistic effects
            # This is a simplified implementation of relativistic effects
            # In reality, this would be much more complex

            # Relativistic effects increase with Z
            rel_factor = 0.001 * self.Z**2

            proton_factor = min(
                abs(self.Z - magic) for magic in [114, 120, 126]
            )
            neutron_factor = min(
                abs(self.N - magic) for magic in [184, 228]
            )

            # Shell corrections with relativistic scaling
            proton_correction = (
                -7.0
                * np.exp(-0.04 * proton_factor)
                * (1 + rel_factor)
            )
            neutron_correction = -7.0 * np.exp(-0.04 * neutron_factor)

            # Additional correction for specific combinations (based on theoretical models)
            if self.Z == 126 and self.N == 184:  # 310Ubh
                return proton_correction + neutron_correction - 8.0
            elif self.Z == 126 and self.N == 228:  # 354Ubh
                return proton_correction + neutron_correction - 10.0
            else:
                return proton_correction + neutron_correction

        else:
            raise ValueError(f"Unknown model: {model}")

    def total_binding_energy(self, model="default"):
        """
        Calculate total binding energy including shell corrections.

        Args:
            model (str): Model to use for shell corrections

        Returns:
            float: Total binding energy in MeV
        """
        return self.liquid_drop_energy() + self.shell_correction(
            model
        )

    def alpha_decay_energy(self, model="default"):
        """
        Calculate Q-value for alpha decay.

        Args:
            model (str): Model to use for energy calculations

        Returns:
            float: Alpha decay Q-value in MeV
        """
        # Calculate parent binding energy
        parent_energy = self.total_binding_energy(model)

        # Create daughter nucleus (Z-2, N-2) after alpha decay
        daughter = NuclearShellModel(self.Z - 2, self.N - 2)
        daughter_energy = daughter.total_binding_energy(model)

        # Alpha particle binding energy (4He)
        alpha_energy = 28.3  # MeV

        # Q-value = (M_parent - M_daughter - M_alpha) * c^2
        # Using binding energies: Q = BE_daughter + BE_alpha - BE_parent
        return daughter_energy + alpha_energy - parent_energy

    def alpha_decay_half_life(self, model="default"):
        """
        Estimate alpha decay half-life using Viola-Seaborg formula.

        Args:
            model (str): Model to use for energy calculations

        Returns:
            float: Half-life in seconds
        """
        # Get alpha decay Q-value
        Q_alpha = self.alpha_decay_energy(model)

        if Q_alpha <= 0:
            return float("inf")  # Energetically forbidden

        # Viola-Seaborg parameters
        a = 1.66175
        b = -8.5166
        c = 0.20228
        d = -33.9069

        # Shell correction term
        h_n = 0
        if self.N >= 126:
            h_n = 1.5  # Enhanced stability for N >= 126

        # Calculate log(T1/2) where T1/2 is in seconds
        log_t = (
            (a * self.Z + b) * Q_alpha ** (-0.5)
            + (c * self.Z + d)
            + h_n
        )

        return 10**log_t

    def spontaneous_fission_half_life(self, model="default"):
        """
        Estimate spontaneous fission half-life.

        Args:
            model (str): Model to use for calculations

        Returns:
            float: Half-life in seconds
        """
        # This is a simplified model based on Z^2/A parameter and shell effects

        # Fissility parameter x = (Z^2/A) / (Z^2/A)_critical
        x = (self.Z**2 / self.A) / 50.883

        # Base calculation for log(T1/2) where T1/2 is in seconds
        log_t_base = 1036.5 - 8.28 * self.Z**2 / self.A

        # Shell correction factor
        shell_factor = 0

        if model == "default":
            # Simple shell correction
            if (
                self.Z in self.PROTON_MAGIC_NUMBERS
                or abs(self.Z - 126) < 5
            ):
                shell_factor += 3.0
            if (
                self.N in self.NEUTRON_MAGIC_NUMBERS
                or abs(self.N - 184) < 5
                or abs(self.N - 228) < 5
            ):
                shell_factor += 3.0

        elif model == "relativistic":
            # More detailed shell correction
            if self.Z == 126:
                shell_factor += 5.0
            elif abs(self.Z - 126) < 5:
                shell_factor += 2.0

            if self.N == 184:
                shell_factor += 5.0
            elif self.N == 228:
                shell_factor += 6.0
            elif abs(self.N - 184) < 5 or abs(self.N - 228) < 5:
                shell_factor += 2.0

            # Special case for 354Ubh (Z=126, N=228)
            if self.Z == 126 and self.N == 228:
                shell_factor += 5.0  # Additional stability

        # Apply shell correction
        log_t = log_t_base + shell_factor

        # Handle very large values to prevent overflow
        if log_t > 308:  # Maximum exponent for double precision
            return float("inf")
        elif log_t < -308:  # Minimum exponent for double precision
            return 1e-20

        # Calculate half-life with bounds checking
        try:
            half_life = 10**log_t
            return max(half_life, 1e-20)
        except OverflowError:
            return float("inf")

    def total_half_life(self, model="default"):
        """
        Calculate total half-life considering all decay modes.

        Args:
            model (str): Model to use for calculations

        Returns:
            tuple: (half-life in seconds, dominant decay mode)
        """
        t_alpha = self.alpha_decay_half_life(model)
        t_sf = self.spontaneous_fission_half_life(model)

        # Total decay constant is sum of individual decay constants
        if t_alpha == float("inf") and t_sf == float("inf"):
            return float("inf"), "stable"
        elif t_alpha == float("inf"):
            return t_sf, "sf"
        elif t_sf == float("inf"):
            return t_alpha, "α"
        else:
            lambda_total = np.log(2) / t_alpha + np.log(2) / t_sf
            t_total = np.log(2) / lambda_total

            # Determine dominant decay mode
            if t_alpha < t_sf:
                mode = "α"
            else:
                mode = "sf"

            return t_total, mode


class StabilityAnalyzer:
    """
    Class to analyze stability patterns across multiple isotopes and models.
    """

    def __init__(self):
        self.results = {}
        self.models = ["default", "relativistic"]

    def analyze_isotope_chain(
        self, Z, N_min, N_max, model="relativistic"
    ):
        """
        Analyze a chain of isotopes with varying neutron numbers.

        Args:
            Z (int): Proton number
            N_min (int): Minimum neutron number
            N_max (int): Maximum neutron number
            model (str): Model to use for calculations

        Returns:
            pandas.DataFrame: Results for each isotope
        """
        results = []

        for N in range(N_min, N_max + 1):
            nucleus = NuclearShellModel(Z, N)

            binding_energy = nucleus.total_binding_energy(model)
            binding_energy_per_nucleon = binding_energy / (Z + N)

            shell_correction = nucleus.shell_correction(model)

            alpha_q = nucleus.alpha_decay_energy(model)

            half_life, decay_mode = nucleus.total_half_life(model)

            # Convert half-life to years for very stable nuclei
            half_life_years = half_life / (365.25 * 24 * 3600)

            if half_life > 1e10:
                half_life_display = f"{half_life_years:.2e} years"
            elif half_life > 31536000:  # > 1 year
                half_life_display = f"{half_life_years:.2f} years"
            elif half_life > 86400:  # > 1 day
                half_life_display = f"{half_life/86400:.2f} days"
            elif half_life > 3600:  # > 1 hour
                half_life_display = f"{half_life/3600:.2f} hours"
            elif half_life > 60:  # > 1 minute
                half_life_display = f"{half_life/60:.2f} minutes"
            else:
                half_life_display = f"{half_life:.2e} seconds"

            results.append(
                {
                    "Z": Z,
                    "N": N,
                    "A": Z + N,
                    "Binding Energy (MeV)": binding_energy,
                    "Binding Energy per Nucleon (MeV)": binding_energy_per_nucleon,
                    "Shell Correction (MeV)": shell_correction,
                    "Alpha Q-value (MeV)": alpha_q,
                    "Half-life (s)": half_life,
                    "Half-life": half_life_display,
                    "Decay Mode": decay_mode,
                }
            )

        df = pd.DataFrame(results)
        self.results[f"Z={Z}, {model}"] = df
        return df

    def plot_half_lives(
        self, Z, N_min, N_max, models=None, log_scale=True
    ):
        """
        Plot half-lives for a chain of isotopes using different models.

        Args:
            Z (int): Proton number
            N_min (int): Minimum neutron number
            N_max (int): Maximum neutron number
            models (list): List of models to compare
            log_scale (bool): Whether to use log scale for half-life axis
        """
        if models is None:
            models = self.models

        plt.figure(figsize=(12, 8))

        for model in models:
            # Ensure we have the data
            key = f"Z={Z}, {model}"
            if key not in self.results:
                self.analyze_isotope_chain(Z, N_min, N_max, model)

            df = self.results[key]

            # Plot half-life vs neutron number
            plt.plot(
                df["N"],
                df["Half-life (s)"],
                "o-",
                label=f"Model: {model}",
            )

            # Highlight magic neutron numbers
            for magic_n in [184, 228]:
                if N_min <= magic_n <= N_max:
                    plt.axvline(
                        x=magic_n,
                        color="gray",
                        linestyle="--",
                        alpha=0.5,
                    )

        plt.xlabel("Neutron Number (N)", fontsize=14)
        plt.ylabel("Half-life (seconds)", fontsize=14)
        if log_scale:
            plt.yscale("log")
        plt.title(
            f"Predicted Half-lives for Z={Z} Isotopes", fontsize=16
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mark specific isotopes of interest
        if Z == 126 and N_min <= 184 <= N_max:
            plt.annotate(
                "$^{310}$Ubh",
                xy=(
                    184,
                    self.results[f"Z={Z}, {models[0]}"]
                    .loc[
                        self.results[f"Z={Z}, {models[0]}"]["N"]
                        == 184,
                        "Half-life (s)",
                    ]
                    .values[0],
                ),
                xytext=(
                    184 + 2,
                    self.results[f"Z={Z}, {models[0]}"]
                    .loc[
                        self.results[f"Z={Z}, {models[0]}"]["N"]
                        == 184,
                        "Half-life (s)",
                    ]
                    .values[0]
                    * 2,
                ),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3"
                ),
                fontsize=12,
            )

        if Z == 126 and N_min <= 228 <= N_max:
            plt.annotate(
                "$^{354}$Ubh",
                xy=(
                    228,
                    self.results[f"Z={Z}, {models[0]}"]
                    .loc[
                        self.results[f"Z={Z}, {models[0]}"]["N"]
                        == 228,
                        "Half-life (s)",
                    ]
                    .values[0],
                ),
                xytext=(
                    228 + 2,
                    self.results[f"Z={Z}, {models[0]}"]
                    .loc[
                        self.results[f"Z={Z}, {models[0]}"]["N"]
                        == 228,
                        "Half-life (s)",
                    ]
                    .values[0]
                    * 2,
                ),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3"
                ),
                fontsize=12,
            )

        plt.tight_layout()
        plt.show()

    def plot_shell_corrections(
        self, Z, N_min, N_max, model="relativistic"
    ):
        """
        Plot shell correction energies for a chain of isotopes.

        Args:
            Z (int): Proton number
            N_min (int): Minimum neutron number
            N_max (int): Maximum neutron number
            model (str): Model to use for calculations
        """
        # Ensure we have the data
        key = f"Z={Z}, {model}"
        if key not in self.results:
            self.analyze_isotope_chain(Z, N_min, N_max, model)

        df = self.results[key]

        plt.figure(figsize=(12, 6))
        plt.plot(
            df["N"],
            df["Shell Correction (MeV)"],
            "o-",
            color="darkblue",
        )

        # Highlight magic neutron numbers
        for magic_n in [184, 228]:
            if N_min <= magic_n <= N_max:
                plt.axvline(
                    x=magic_n, color="gray", linestyle="--", alpha=0.5
                )

        plt.xlabel("Neutron Number (N)", fontsize=14)
        plt.ylabel("Shell Correction Energy (MeV)", fontsize=14)
        plt.title(
            f"Shell Correction Energies for Z={Z} Isotopes ({model} model)",
            fontsize=16,
        )
        plt.grid(True, alpha=0.3)

        # Mark specific isotopes of interest
        if Z == 126 and N_min <= 184 <= N_max:
            plt.annotate(
                "$^{310}$Ubh",
                xy=(
                    184,
                    df.loc[
                        df["N"] == 184, "Shell Correction (MeV)"
                    ].values[0],
                ),
                xytext=(
                    184 + 5,
                    df.loc[
                        df["N"] == 184, "Shell Correction (MeV)"
                    ].values[0]
                    + 1,
                ),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3"
                ),
                fontsize=12,
            )

        if Z == 126 and N_min <= 228 <= N_max:
            plt.annotate(
                "$^{354}$Ubh",
                xy=(
                    228,
                    df.loc[
                        df["N"] == 228, "Shell Correction (MeV)"
                    ].values[0],
                ),
                xytext=(
                    228 + 5,
                    df.loc[
                        df["N"] == 228, "Shell Correction (MeV)"
                    ].values[0]
                    + 1,
                ),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3"
                ),
                fontsize=12,
            )

        plt.tight_layout()
        plt.show()

    def compare_magic_numbers(
        self, Z_values, N=184, model="relativistic"
    ):
        """
        Compare stability across different potential magic proton numbers.

        Args:
            Z_values (list): List of proton numbers to compare
            N (int): Neutron number to use
            model (str): Model to use for calculations
        """
        results = []

        for Z in Z_values:
            nucleus = NuclearShellModel(Z, N)

            binding_energy = nucleus.total_binding_energy(model)
            shell_correction = nucleus.shell_correction(model)
            half_life, decay_mode = nucleus.total_half_life(model)

            results.append(
                {
                    "Z": Z,
                    "N": N,
                    "A": Z + N,
                    "Binding Energy (MeV)": binding_energy,
                    "Shell Correction (MeV)": shell_correction,
                    "Half-life (s)": half_life,
                    "Decay Mode": decay_mode,
                }
            )

        df = pd.DataFrame(results)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Shell correction plot
        ax1.plot(
            df["Z"],
            df["Shell Correction (MeV)"],
            "o-",
            color="darkblue",
        )
        ax1.set_xlabel("Proton Number (Z)", fontsize=14)
        ax1.set_ylabel("Shell Correction Energy (MeV)", fontsize=14)
        ax1.set_title(
            f"Shell Correction Energies for N={N} Isotones",
            fontsize=16,
        )
        ax1.grid(True, alpha=0.3)

        # Highlight frequently predicted magic numbers
        for magic_z in [114, 120, 126]:
            if min(Z_values) <= magic_z <= max(Z_values):
                ax1.axvline(
                    x=magic_z, color="gray", linestyle="--", alpha=0.5
                )
                ax1.annotate(
                    f"Z={magic_z}",
                    xy=(magic_z, 0),
                    xytext=(magic_z, 1),
                    ha="center",
                    fontsize=12,
                )

        # Half-life plot
        ax2.plot(df["Z"], df["Half-life (s)"], "o-", color="darkred")
        ax2.set_xlabel("Proton Number (Z)", fontsize=14)
        ax2.set_ylabel("Half-life (seconds)", fontsize=14)
        ax2.set_title(
            f"Predicted Half-lives for N={N} Isotones", fontsize=16
        )
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Highlight frequently predicted magic numbers
        for magic_z in [114, 120, 126]:
            if min(Z_values) <= magic_z <= max(Z_values):
                ax2.axvline(
                    x=magic_z, color="gray", linestyle="--", alpha=0.5
                )

        plt.tight_layout()
        plt.show()

        return df


def test_unbihexium_stability():
    """
    Test function to analyze stability of unbihexium isotopes.
    """
    # Create analyzer object
    analyzer = StabilityAnalyzer()

    # Analyze unbihexium isotopes with neutron numbers around potential magic numbers
    print("Analyzing Unbihexium (Z=126) isotopes...")
    df = analyzer.analyze_isotope_chain(
        126, 182, 230, model="relativistic"
    )

    # Display most stable isotopes
    print("\nMost stable Unbihexium isotopes:")
    most_stable = df.sort_values(
        "Half-life (s)", ascending=False
    ).head(5)
    print(
        most_stable[
            [
                "A",
                "N",
                "Half-life",
                "Decay Mode",
                "Shell Correction (MeV)",
            ]
        ]
    )

    # Analyze isotope with N=184 (magic neutron number)
    print("\nDetailed analysis of 310Ubh (Z=126, N=184):")
    ubh310 = NuclearShellModel(126, 184)
    binding_energy = ubh310.total_binding_energy(model="relativistic")
    shell_correction = ubh310.shell_correction(model="relativistic")
    alpha_hl = ubh310.alpha_decay_half_life(model="relativistic")
    sf_hl = ubh310.spontaneous_fission_half_life(model="relativistic")
    total_hl, mode = ubh310.total_half_life(model="relativistic")

    print(f"Binding Energy: {binding_energy:.2f} MeV")
    print(f"Shell Correction: {shell_correction:.2f} MeV")
    print(
        f"Alpha Decay Half-life: {alpha_hl:.2e} seconds ({alpha_hl/(365.25*24*3600):.2e} years)"
    )
    print(
        f"Spontaneous Fission Half-life: {sf_hl:.2e} seconds ({sf_hl/(365.25*24*3600):.2e} years)"
    )
    print(
        f"Total Half-life: {total_hl:.2e} seconds ({total_hl/(365.25*24*3600):.2e} years)"
    )
    print(f"Dominant Decay Mode: {mode}")

    # Analyze isotope with N=228 (proposed magic neutron number)
    print("\nDetailed analysis of 354Ubh (Z=126, N=228):")
    ubh354 = NuclearShellModel(126, 228)
    binding_energy = ubh354.total_binding_energy(model="relativistic")
    shell_correction = ubh354.shell_correction(model="relativistic")
    alpha_hl = ubh354.alpha_decay_half_life(model="relativistic")
    sf_hl = ubh354.spontaneous_fission_half_life(model="relativistic")
    total_hl, mode = ubh354.total_half_life(model="relativistic")

    print(f"Binding Energy: {binding_energy:.2f} MeV")
    print(f"Shell Correction: {shell_correction:.2f} MeV")
    print(
        f"Alpha Decay Half-life: {alpha_hl:.2e} seconds ({alpha_hl/(365.25*24*3600):.2e} years)"
    )
    print(
        f"Spontaneous Fission Half-life: {sf_hl:.2e} seconds ({sf_hl/(365.25*24*3600):.2e} years)"
    )
    print(
        f"Total Half-life: {total_hl:.2e} seconds ({total_hl/(365.25*24*3600):.2e} years)"
    )
    print(f"Dominant Decay Mode: {mode}")

    # Plot half-lives for unbihexium isotopes
    print("\nGenerating plots...")
    analyzer.plot_half_lives(
        126, 182, 230, models=["default", "relativistic"]
    )

    # Plot shell correction energies
    analyzer.plot_shell_corrections(126, 182, 230)

    # Compare different potential magic proton numbers
    analyzer.compare_magic_numbers(
        [114, 116, 118, 120, 122, 124, 126, 128, 130], N=184
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    test_unbihexium_stability()
