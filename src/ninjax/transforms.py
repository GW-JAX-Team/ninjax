"""Transforms that can be used on a collection of parameters (gathered in a dict) and return the transformed params

TODO: CRITICAL - This entire file is DEPRECATED for the new jim API!
      The new API uses Transform classes from jimgw.core.single_event.transforms
      Most functions here are NO LONGER USED and should be removed or marked deprecated.

      DEPRECATED functions (replaced by jim transforms or priors):
      - q_to_eta -> Use MassRatioToSymmetricMassRatioTransform from jim
      - cos_iota_to_iota -> Use SinePrior(parameter_names=["iota"]) instead
      - sin_dec_to_dec -> Use CosinePrior(parameter_names=["dec"]) instead

      POTENTIALLY USEFUL functions that may need porting to new Transform API:
      - m1_m2_to_Mc_q (if sampling in m1, m2 space)
      - spin_sphere_to_cartesian_s1/s2 (for precessing spins)
      - binary_Love (for BNS parameter sampling with EOS constraints)

      ACTION REQUIRED:
      1. Mark all deprecated functions with @deprecated decorator
      2. Test if any functions are still used (grep the codebase)
      3. Port useful functions to new jim Transform class API if needed
      4. Remove unused functions to reduce technical debt
"""

import jax.numpy as jnp
from jax.scipy.stats import norm

def q_to_eta(params: dict) -> dict:
    # TODO: DEPRECATED - Use MassRatioToSymmetricMassRatioTransform from jim instead
    params["eta"] = params["q"] / (1 + params["q"]) ** 2
    return params

def cos_iota_to_iota(params: dict):
    # TODO: DEPRECATED - Use SinePrior(parameter_names=["iota"]) instead of sampling cos_iota
    params["iota"] = jnp.arccos(params["cos_iota"])
    return params

def sin_dec_to_dec(params: dict) -> dict:
    # TODO: DEPRECATED - Use CosinePrior(parameter_names=["dec"]) instead of sampling sin_dec
    params["dec"] = jnp.arcsin(params["sin_dec"])
    return params

def m1_m2_to_Mc_q(params: dict) -> dict:
    # Default params: can sample H0 if wanted but fix c for now
    H0 = params.get("H0", 67.4)
    c = 2.998 * 10**5
    
    # Here we ensure that m2 \leq m1, and lambda_2 \geq lambda_1
    m_1_source = jnp.where(params["m_1_source"] > params["m_2_source"], params["m_1_source"], params["m_2_source"])
    m_2_source = jnp.where(params["m_1_source"] > params["m_2_source"], params["m_2_source"], params["m_1_source"])
    
    lambda_1 = jnp.where(params["lambda_2"] > params["lambda_1"], params["lambda_1"], params["lambda_2"])
    lambda_2 = jnp.where(params["lambda_2"] > params["lambda_1"], params["lambda_2"], params["lambda_1"])
    
    # Re-save
    params["m_1_source"] = m_1_source
    params["m_2_source"] = m_2_source
    params["lambda_1"] = lambda_1
    params["lambda_2"] = lambda_2
    
    # Get source-frame chirp mass and convert to detector-frame chirp mass
    M_c_source = (m_1_source * m_2_source) ** (3/5) / (m_1_source + m_2_source) ** (1/5)
    d_L = params["d_L"]
    z = H0 * d_L / c
    M_c = M_c_source * (1 + z)
    
    # Get mass ratio
    q = m_2_source / m_1_source
    
    # Add and return final params
    params["M_c"] = M_c
    params["q"] = q
    
    return params

def spin_sphere_to_cartesian_s1(params: dict) -> dict:
    s1_x = jnp.sin(params["s1_theta"]) * jnp.cos(params["s1_phi"]) * params["s1_mag"]
    s1_y = jnp.sin(params["s1_theta"]) * jnp.sin(params["s1_phi"]) * params["s1_mag"]
    s1_z = jnp.cos(params["s1_theta"]) * params["s1_mag"]
    
    params["s1_x"] = s1_x
    params["s1_y"] = s1_y
    params["s1_z"] = s1_z
    
    return params

def spin_sphere_to_cartesian_s2(params: dict) -> dict:
    s2_x = jnp.sin(params["s2_theta"]) * jnp.cos(params["s2_phi"]) * params["s2_mag"]
    s2_y = jnp.sin(params["s2_theta"]) * jnp.sin(params["s2_phi"]) * params["s2_mag"]
    s2_z = jnp.cos(params["s2_theta"]) * params["s2_mag"]
    
    params["s2_x"] = s2_x
    params["s2_y"] = s2_y
    params["s2_z"] = s2_z
    
    return params

# For overlapping signals
def q1_to_eta1(params: dict) -> dict:
    params["eta_1"] = params["q_1"] / (1 + params["q_1"]) ** 2
    return params

def q2_to_eta2(params: dict) -> dict:
    params["eta_2"] = params["q_2"] / (1 + params["q_2"]) ** 2
    return params

def cos_iota1_to_iota1(params: dict):
    params["iota_1"] = jnp.arccos(params["cos_iota_1"])
    return params

def cos_iota2_to_iota2(params: dict):
    params["iota_2"] = jnp.arccos(params["cos_iota_2"])
    return params

def sin_dec1_to_dec1(params: dict) -> dict:
    params["dec_1"] = jnp.arcsin(params["sin_dec_1"])
    return params

def sin_dec2_to_dec2(params: dict) -> dict:
    params["dec_2"] = jnp.arcsin(params["sin_dec_2"])
    return params

def t_c_dt_to_tcs(params: dict) -> dict:
    params["t_c_1"] = params["t_c"]
    params["t_c_2"] = params["t_c"] + params["dt"]
    return params


### Binary Love ###
# TODO: NEEDS TESTING - Binary Love relations for BNS tidal parameters
#       This could be useful for sampling lambda_symmetric instead of (lambda_1, lambda_2)
#       But needs to be ported to new jim Transform API with forward/backward methods
#       Consider creating BinaryLoveTransform class in jim or ninjax
BINARY_LOVE_COEFFS = {
    "n_polytropic": 0.743,

    "b11": -27.7408,
    "b12": 8.42358,
    "b21": 122.686,
    "b22": -19.7551,
    "b31": -175.496,
    "b32": 133.708,
    
    "c11": -25.5593,
    "c12": 5.58527,
    "c21": 92.0337,
    "c22": 26.8586,
    "c31": -70.247,
    "c32": -56.3076
}

BINARY_LOVE_COEFFS_ERROR = {
    "mu_1": 137.1252739,
    "mu_2": -32.8026613,
    "mu_3": 0.5168637,
    "mu_4": -11.2765281,
    "mu_5": 14.9499544,
    "mu_6": -4.6638851,
    
    "sigma_1": -0.0000739,
    "sigma_2": 0.0103778,
    "sigma_3": 0.4581717,
    "sigma_4": -0.8341913,
    "sigma_5": -201.4323962,
    "sigma_6": 273.9268276,
    "sigma_7": -71.2342246
}

def binary_love_fit(lambda_symmetric: float,
                    mass_ratio: float,
                    fit_coeffs: dict = BINARY_LOVE_COEFFS) -> float:
    """
    Computes lambda_antysymmetric from lambda_symmetric and mass_ratio. Note that this is only the fit, whereas typically the uncertainty would be marginalized over. See the CHZ paper: arXiv:1804.03221v2
    The code is copied from bilby/gw/conversion.py, changing np to jnp for JAX compatibility.
    
    Note: We take the fit coefficients as input, rather than hardcoding them, to allow for changing the fit coefficients later on.
    """
    lambda_symmetric_m1o5 = jnp.power(lambda_symmetric, -1. / 5.)
    lambda_symmetric_m2o5 = lambda_symmetric_m1o5 * lambda_symmetric_m1o5
    lambda_symmetric_m3o5 = lambda_symmetric_m2o5 * lambda_symmetric_m1o5

    q = mass_ratio
    q2 = jnp.square(mass_ratio)

    # Eqn.2 from CHZ, incorporating the dependence on mass ratio
    q_for_Fnofq = jnp.power(q, 10. / (3. - fit_coeffs["n_polytropic"]))
    Fnofq = (1. - q_for_Fnofq) / (1. + q_for_Fnofq)

    # Eqn 1 from CHZ, giving the lambda_antisymmetric_fitOnly (not yet accounting for the uncertainty in the fit)
    numerator = 1.0 + \
        (fit_coeffs["b11"] * q * lambda_symmetric_m1o5) + (fit_coeffs["b12"] * q2 * lambda_symmetric_m1o5) + \
        (fit_coeffs["b21"] * q * lambda_symmetric_m2o5) + (fit_coeffs["b22"] * q2 * lambda_symmetric_m2o5) + \
        (fit_coeffs["b31"] * q * lambda_symmetric_m3o5) + (fit_coeffs["b32"] * q2 * lambda_symmetric_m3o5)

    denominator = 1.0 + \
        (fit_coeffs["c11"] * q * lambda_symmetric_m1o5) + (fit_coeffs["c12"] * q2 * lambda_symmetric_m1o5) + \
        (fit_coeffs["c21"] * q * lambda_symmetric_m2o5) + (fit_coeffs["c22"] * q2 * lambda_symmetric_m2o5) + \
        (fit_coeffs["c31"] * q * lambda_symmetric_m3o5) + (fit_coeffs["c32"] * q2 * lambda_symmetric_m3o5)

    lambda_antisymmetric_fitOnly = Fnofq * lambda_symmetric * numerator / denominator

    return lambda_antisymmetric_fitOnly



def binary_Love(params: dict,
                error_fit_coeffs: dict = BINARY_LOVE_COEFFS_ERROR,
                penalty_lambda: float = 10_000.0) -> dict:
    """Taken from bilby/gw/conversion.py. Converts the symmetric Lambda_s to the antisymmetric Lambda_a, and returns the two values of Lambda_1 and Lambda_2 from those."""
    
    # We sample the uniform number for the error marginalization in the prior itself
    binary_love_uniform = params["binary_Love_uniform"]
    
    # Get the fit parameters
    lambda_symmetric = params["lambda_symmetric"]
    lambda_symmetric_sqrt = jnp.sqrt(lambda_symmetric)
    q = params["q"]
    q2 = jnp.square(q)
    
    # Get the fit Lambda_a value
    lambda_antisymmetric_fitOnly = binary_love_fit(lambda_symmetric, q)
    
    # Do the marginalization over the error
    lambda_antisymmetric_lambda_symmetric_meanCorr = \
        (error_fit_coeffs["mu_1"] / (lambda_symmetric * lambda_symmetric)) + \
        (error_fit_coeffs["mu_2"] / lambda_symmetric) + error_fit_coeffs["mu_3"]

    lambda_antisymmetric_lambda_symmetric_stdCorr = \
        (error_fit_coeffs["sigma_1"] * lambda_symmetric * lambda_symmetric_sqrt) + \
        (error_fit_coeffs["sigma_2"] * lambda_symmetric) + \
        (error_fit_coeffs["sigma_3"] * lambda_symmetric_sqrt) + error_fit_coeffs["sigma_4"]

    lambda_antisymmetric_mass_ratio_meanCorr = \
        (error_fit_coeffs["mu_4"] * q2) + (error_fit_coeffs["mu_5"] * q) + error_fit_coeffs["mu_6"]

    lambda_antisymmetric_mass_ratio_stdCorr = \
        (error_fit_coeffs["sigma_5"] * q2) + (error_fit_coeffs["sigma_6"] * q) + error_fit_coeffs["sigma_7"]

    lambda_antisymmetric_meanCorr = \
        (lambda_antisymmetric_lambda_symmetric_meanCorr +
            lambda_antisymmetric_mass_ratio_meanCorr) / 2.

    lambda_antisymmetric_stdCorr = \
        jnp.sqrt(jnp.square(lambda_antisymmetric_lambda_symmetric_stdCorr) +
                jnp.square(lambda_antisymmetric_mass_ratio_stdCorr))

    lambda_antisymmetric_scatter = norm.ppf(binary_love_uniform, loc=0.,
                                            scale=lambda_antisymmetric_stdCorr)

    lambda_antisymmetric = lambda_antisymmetric_fitOnly + \
        (lambda_antisymmetric_meanCorr + lambda_antisymmetric_scatter)
        
    # Clip lambda_antisymmetric to be positive
    lambda_antisymmetric = jnp.clip(lambda_antisymmetric, a_min = 0.0, a_max = lambda_symmetric)

    # Get the Lambdas, but check for their sanity and otherwise return large dummy values
    lambda_1 = lambda_symmetric - lambda_antisymmetric
    lambda_2 = lambda_symmetric + lambda_antisymmetric

    # TODO: no longer needed I guess?
    # # In case we do not agree with Lambda_1 <= Lambda_2, we give a large penalty
    # lambda_1 = jnp.where(maybe_lambda_1 > maybe_lambda_2, penalty_lambda, maybe_lambda_1)
    # lambda_2 = jnp.where(maybe_lambda_1 > maybe_lambda_2, penalty_lambda, maybe_lambda_2)

    params["lambda_1"] = lambda_1
    params["lambda_2"] = lambda_2

    return params