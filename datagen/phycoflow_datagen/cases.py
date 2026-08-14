"""Canonical case names, notebook defaults, and metadata."""

from __future__ import annotations

from typing import Any


CASES: dict[str, dict[str, Any]] = {
    "burgers": {
        "display_name": "1D viscous Burgers",
        "folder": "1_burgers",
        "spatial_dimension": 1,
        "resolution": 512,
        "domain_length": 2.0 * 3.141592653589793,
        "dt": 0.0025,
        "burn_in_time": 0.0,
        "record_time": 2.0,
        "save_every": 4,
        "state_names": ["u"],
        "field_names": ["u"],
        "condition_names": ["viscosity"],
        "equation": "u_t + u u_x = viscosity u_xx",
    },
    "ks": {
        "display_name": "1D Kuramoto-Sivashinsky",
        "folder": "2_ks",
        "spatial_dimension": 1,
        "resolution": 256,
        "domain_length": 60.0,
        "dt": 0.05,
        "burn_in_time": 50.0,
        "record_time": 100.0,
        "save_every": 5,
        "state_names": ["u"],
        "field_names": ["u"],
        "condition_names": [
            "advection_coefficient",
            "second_order_coefficient",
            "fourth_order_coefficient",
        ],
        "equation": "u_t + u u_x + u_xx + u_xxxx = 0",
    },
    "brusselator": {
        "display_name": "2D Brusselator reaction-diffusion",
        "folder": "3_brusselator",
        "spatial_dimension": 2,
        "resolution": 192,
        "domain_length": 20.0,
        "dt": 0.01,
        "burn_in_time": 0.0,
        "record_time": 12.0,
        "save_every": 5,
        "state_names": ["u", "v"],
        "field_names": ["u", "v"],
        "condition_names": ["A", "B", "diffusivity_u", "diffusivity_v"],
        "equation": "Brusselator reaction-diffusion system",
    },
    "kolmogorov": {
        "display_name": "2D Kolmogorov flow",
        "folder": "4_navier_stokes",
        "spatial_dimension": 2,
        "resolution": 192,
        "domain_length": 2.0 * 3.141592653589793,
        "dt": 0.01,
        "burn_in_time": 20.0,
        "record_time": 10.0,
        "save_every": 5,
        "state_names": ["omega"],
        "field_names": ["u", "v", "p"],
        "condition_names": ["reynolds_number", "forcing_amplitude", "forcing_wavenumber"],
        "equation": "2D incompressible Navier-Stokes with sinusoidal Kolmogorov forcing",
    },
    "electro_thermal": {
        "display_name": "2D electro-thermal coupling",
        "folder": "5_electro_thermal",
        "spatial_dimension": 2,
        "resolution": 128,
        "domain_length": 0.128,
        "dt": 1.0,
        "burn_in_time": 0.0,
        "record_time": 1.0,
        "save_every": 1,
        "backend": "numpy",
        "device": "cpu",
        "state_names": ["E_z_real", "E_z_imag", "temperature"],
        "field_names": ["E_z_real", "E_z_imag", "temperature"],
        "field_units": ["V/m", "V/m", "K"],
        "condition_names": ["a", "b", "phi", "Sigma_Si", "kappa_alumina"],
        "condition_units": ["m", "m", "rad", "S/m", "W/(m K)"],
        "coordinate_units": "m",
        "time_units": "steady sample",
        "periodic_axes": [],
        "equation": "TE scalar Helmholtz equation bidirectionally coupled to steady heat diffusion",
    },
    "mass_transport_fluid": {
        "display_name": "2D Elder-type mass transport-fluid coupling",
        "folder": "6_mass_transport_fluid",
        "spatial_dimension": 2,
        "resolution": 128,
        "domain_length": 300.0,
        "domain_height": 150.0,
        "dt": 0.25,
        "burn_in_time": 0.0,
        "record_time": 20.0,
        "save_every": 8,
        "backend": "numpy",
        "device": "cpu",
        "state_names": ["u_x", "u_y", "concentration"],
        "field_names": ["u_x", "u_y", "concentration"],
        "field_units": ["m/s", "m/s", "mol/m^3"],
        "condition_names": ["A", "x0", "y0", "s"],
        "condition_units": ["mol/m^3", "m", "m", "m"],
        "coordinate_units": "m",
        "time_units": "years",
        "periodic_axes": [],
        "equation": "Density-coupled Darcy flow with conservative advection-diffusion transport",
    },
}


ALIASES = {
    "viscous_burgers": "burgers",
    "kuramoto_sivashinsky": "ks",
    "reaction_diffusion": "brusselator",
    "navier_stokes": "kolmogorov",
    "kolmogorov_flow": "kolmogorov",
    "electrothermal": "electro_thermal",
    "te_heat": "electro_thermal",
    "elder": "mass_transport_fluid",
    "mass_transport": "mass_transport_fluid",
}


def canonical_case_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CASES:
        raise ValueError(f"unknown case {name!r}; choose from {sorted(CASES)}")
    return normalized
