# Copyright 2025 Justin Philip Tuazon, Gia Mizrane Abubo

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

# interpretablefa v6.0.6
# https://pypi.org/project/interpretablefa/

import math
import time
import warnings
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity, covariance_to_correlation
import tensorflow_hub as hub
from scipy.stats import kendalltau, chi2
from scipy.optimize import shgo, minimize, NonlinearConstraint
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

ORTHOGONAL_ROTATIONS = ["priorimax", "varimax", "oblimax", "quartimax", "equamax"]
OBLIQUE_ROTATIONS = ["promax", "oblimin", "quartimin"]
POSSIBLE_ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS
OPT_SEED = 123


class PriorimaxRotator:
    """
    The class for the optimization routine of the priorimax rotation.

    Parameters
    ----------
    is_global: bool, optional
        If this is `True`, then the problem of finding the priorimax rotation will be treated as a global
        optimization problem (SHGO with COBYQA). Otherwise, local optimization (COBYQA) is used. The default
        value is `False`. Local optimization is generally faster but may produce a sub-optimal solution.
    num_starts: int, optional
        This is the number of random starts (i.e., number of local optimizations) used for finding the priorimax
        rotation, if `is_global` is `False`. The default value is 1. Note that the first start is always taken to
        be the identity matrix, so that the starts are the identity matrix and (random_starts - 1) random rotation
        matrices. This is ignored if `is_global` is `True`.
    samp_points: int, optional
        This dictates the number of sampling points used in the construction of the simplicial complex for the
        SHGO algorithm, if `is_global` is `True`. The default value is 500. The total number of sampling points used
        is (samp_points * ((T^2 + T) / 2)), where T is the number of factors. This is ignored if `is_global` is
        `False`.
    max_time: float, optional
        This is the maximum amount of time in seconds for which an optimizer will run to find the priorimax
        rotation. If `max_time` is 0 or negative, then the pre-defined orthogonal rotation (e.g., varimax,
        equamax, etc.) with the best index value is selected (i.e., the priorimax procedure is performed on the
        set of pre-defined orthogonal rotations). The default value is`300.0`.

    Attributes
    ----------
    is_global: bool
        Whether optimization is local (`False`) or global (`True`).
    num_starts: int
        The number of random starts used when `is_global` is `False`.
    samp_points: int
        The multiplier to the number of sampling points for SHGO.
    max_time: float
        The maximum amount of time in seconds for which an optimizer runs.
    """

    def __init__(self, is_global=False, num_starts=1, samp_points=500, max_time=300.0,
                 cobyqa_initial_tr=0.2, cobyqa_final_tr=1e-6):
        """
        Initializes the rotator
        """

        # Arg checks
        if not isinstance(is_global, bool):
            raise TypeError("is_global must be bool")

        try:
            num_starts = int(num_starts)
        except ValueError:
            raise TypeError("num_starts must be an int or coercible to int")
        if num_starts < 1:
            raise ValueError("num_starts must be at least 1")

        try:
            samp_points = int(samp_points)
        except ValueError:
            raise TypeError("samp_points must be an int or coercible to int")
        if samp_points < 1:
            raise ValueError("samp_points must be at least 1")

        try:
            max_time = float(max_time)
        except ValueError:
            raise TypeError("max_time must be a float or coercible to float")

        # Set values
        self.is_global = is_global
        self.max_time = max_time
        self.num_starts = num_starts
        self.samp_points = samp_points
        self.cobyqa_initial_tr = cobyqa_initial_tr
        self.cobyqa_final_tr = cobyqa_final_tr

        self._start_time = None
        self._last_best_rotation_matrix = None
        self._random_state = np.random.RandomState(OPT_SEED)

    @staticmethod
    def _get_rotation_matrix(x):
        # This gets the rotation matrix from the array x with (T^2 + T) / 2 elements, where T is the number of factors

        # Convert the input into a skew-symmetrix matrix S and signature matrix D
        # Note that len(x) = (T^2 + T) / 2
        # Simply solve the equation for T > 0 to find the number of factors
        num_of_factors = int((-1 + math.sqrt(1 + 8 * len(x))) / 2)
        skew_symmetric_matrix = np.zeros(shape=(num_of_factors, num_of_factors))
        ind = 0
        for i in range(num_of_factors):
            for j in range(i):
                skew_symmetric_matrix[i, j] = x[ind]
                skew_symmetric_matrix[j, i] = -x[ind]
                ind += 1
        diag_matrix = np.zeros(shape=(num_of_factors, num_of_factors))
        for ind_ in range(ind, len(x)):
            diag_matrix[ind_ - ind, ind_ - ind] = x[ind_]

        # The orthogonal rotation matrix is ((I - S)(I + S)^(-1))D
        identity_matrix = np.identity(num_of_factors)
        i_minus_s = identity_matrix - skew_symmetric_matrix
        i_plus_s = identity_matrix + skew_symmetric_matrix
        i_s_product = i_minus_s @ np.linalg.inv(i_plus_s)
        rotation_matrix = i_s_product @ diag_matrix

        return rotation_matrix

    def _get_rotated_loadings(self, x, unrotated_loadings):
        # This gets the rotated loadings

        rotation_matrix = self._get_rotation_matrix(x)
        loadings = unrotated_loadings @ rotation_matrix

        return loadings

    def _obj_fun(self, x, unrotated_loadings, ifa_obj, model_name):
        # The optimization problem is a minimization problem
        # Goal must be to minimize -V to maximize V

        return -self._get_v(x, unrotated_loadings, ifa_obj, model_name)

    def _get_v(self, x, unrotated_loadings, ifa_obj, model_name, model=None):
        # This gets the V-index

        # Get the prior similarities (a) and the loading similarities (b)
        if model is None:
            loadings = self._get_rotated_loadings(x, unrotated_loadings)
        else:
            loadings = model.loadings_
        num_of_vars = loadings.shape[0]
        correlations = loadings / ifa_obj._scaler
        prior = ifa_obj.models[model_name].prior_

        if prior is None:
            return None
        elif isinstance(prior, list):
            prior = ifa_obj.calculate_semantic_similarity(prior)

        a = []
        b = []
        for i in range(num_of_vars):
            for j in range(i):
                if not pd.isna(prior[i, j]):
                    a.append(prior[i, j])
                    x_1 = correlations[i, :]
                    x_2 = correlations[j, :]
                    b.append(1 - math.sqrt((1 / 2) * np.sum(((x_1 ** 2) - (x_2 ** 2)) ** 2)))

        # Compute
        n = len(a)
        tau = (1 / 2) * (kendalltau(a, b, variant="b").statistic + 1)
        a = np.array(a)
        b = np.array(b)
        theta = n * np.sum(a * b) - np.sum(a) * np.sum(b)
        theta = theta / (n * np.sum(a ** 2) - (np.sum(a)) ** 2)
        theta = (1 / math.pi) * np.arctan(theta) + 1 / 2
        v = math.sqrt(tau * theta)

        return v

    def _get_best_predefined(self, ifa_obj, model_name):
        # This gets the best rotation (in terms of the interpretability index) among the pre-defined rotations

        # Initialize values
        models = []
        indices = []
        rot_names = []
        num_man = ifa_obj.models[model_name].loadings_.shape[0]
        num_fac = ifa_obj.models[model_name].loadings_.shape[1]

        # Fit all available orthogonal rotations (except priorimax)
        for rot in np.setdiff1d(ORTHOGONAL_ROTATIONS, ["priorimax"]):
            if rot == "equamax":
                rot_kwargs = {
                    "kappa": num_fac / (2 * num_man)
                }
            else:
                rot_kwargs = None
            temp_model = FactorAnalyzer(
                n_factors=num_fac,
                rotation=rot,
                is_corr_matrix=ifa_obj.is_corr_matrix,
                rotation_kwargs=rot_kwargs
            )
            temp_model.fit(ifa_obj.data_)
            models.append(temp_model)
            rot_names.append(rot)
            indices.append(self._get_v(None, None, ifa_obj, model_name, models[-1]))

        # Return the model with the best index value
        return [models[indices.index(max(indices))], max(indices), rot_names[indices.index(max(indices))]]

    @staticmethod
    def _generate_constraint(ind):
        # This generates a constraint for the signature matrix

        def _constraint(x):
            return x[ind] ** 2 - 1

        return NonlinearConstraint(_constraint, 0, 0)

    def rotate(self, ifa_obj, model_name):
        # This implements the priorimax rotation

        # Initialize values
        none_ind = ifa_obj.calculate_v_index(model_name)
        opt_ind = -1
        pre_mod, pre_ind, pre_name = self._get_best_predefined(ifa_obj, model_name)
        unrotated_loadings = ifa_obj.models[model_name].loadings_.copy()

        # Initialize the optimizer
        num_of_factors = unrotated_loadings.shape[1]
        num_of_skew_vars = int((num_of_factors * (num_of_factors - 1)) / 2)
        num_of_diag_vars = int(num_of_factors)
        num_of_mat_vars = num_of_skew_vars + num_of_diag_vars

        # Set bound constraints for optimization
        bounds = [(-1, 1)] * num_of_mat_vars

        # Set additional constraints for optimization
        constraints = []
        for i in range(num_of_diag_vars):
            constraints.append(self._generate_constraint(num_of_skew_vars + i))

        if self.is_global:
            print(f"Performing the priorimax rotation using the global optimization algorithm, SHGO, with "
                  f"{self.samp_points * num_of_mat_vars} sampling points and with COBYQA as the local "
                  f"optimization routine...")
        else:
            print(f"Performing the priorimax rotation using the local optimization algorithm, COBYQA, with "
                  f"{self.num_starts} random start(s)...")

        # Optimize
        result = None
        run_local = not self.is_global
        self._start_time = time.time()
        local_starts = self.num_starts
        if self.max_time > 0:
            if self.is_global:
                result = shgo(
                    func=self._obj_fun,
                    bounds=bounds,
                    args=(unrotated_loadings, ifa_obj, model_name),
                    constraints=constraints,
                    callback=self._callback,
                    minimizer_kwargs={
                        "method": "COBYQA",
                        "options": {
                            "initial_tr_radius": self.cobyqa_initial_tr,
                            "final_tr_radius": self.cobyqa_final_tr
                        }
                    },
                    sampling_method="simplicial",
                    n=self.samp_points * num_of_mat_vars
                )
                if not result.success:
                    warnings.warn(f"Global optimization failed. Try increasing `samp_points` or `max_time`. "
                                  f"You can also try modifying COBYQA trust radii. Falling back to local optimization "
                                  f"with 5 random starts...", RuntimeWarning)
                    run_local = True
                    local_starts = 5
            if run_local:
                max_ind = -1
                suceeded = False
                for i in range(local_starts):
                    # Get a random start, but make sure first start is identity
                    if i > 0:
                        skew = self._random_state.uniform(-1, 1, num_of_skew_vars)
                        sig = self._random_state.choice([-1, 1], num_of_diag_vars)
                    else:
                        skew = np.zeros(num_of_skew_vars)
                        sig = np.ones(num_of_diag_vars)

                    # Optimize
                    temp_result = minimize(
                        fun=self._obj_fun,
                        x0=np.append(skew, sig),
                        bounds=bounds,
                        args=(unrotated_loadings, ifa_obj, model_name),
                        constraints=constraints,
                        method="COBYQA",
                        callback=self._callback,
                        options={
                            "initial_tr_radius": self.cobyqa_initial_tr,
                            "final_tr_radius": self.cobyqa_final_tr
                        }
                    )
                    if temp_result.success:
                        if abs(temp_result.fun) > max_ind:
                            result = temp_result
                            max_ind = abs(temp_result.fun)
                            suceeded = True
                    else:
                        warnings.warn(f"Local random start {i} failed to converge. It will be skipped.", RuntimeWarning)
                if not suceeded:
                    warnings.warn("All local random starts failed. Try increasing "
                                  "`num_starts` or `max_time`. You can also try modifying COBYQA "
                                  "trust radii.", RuntimeWarning)

        # Extract "manual" optimization results, if present
        if result is not None:
            if result.success:
                candidate_sol = result.x
            else:
                candidate_sol = self._last_best_rotation_matrix
            if candidate_sol is not None:
                ifa_obj.models[model_name].loadings_ = self._get_rotated_loadings(candidate_sol, unrotated_loadings)
                ifa_obj.models[model_name].rotation_matrix_ = self._get_rotation_matrix(candidate_sol)
                opt_ind = ifa_obj.calculate_v_index(model_name)

        # Decide on the best rotation matrix
        # 0 - unrotated loadings, 1 - a pre-defined rotation, 2 - manual priorimax rotation from optimization
        inds = [none_ind, pre_ind, opt_ind]
        best = inds.index(max(inds))
        actual_rot = None
        if best == 0:
            ifa_obj.models[model_name].loadings_ = unrotated_loadings
            ifa_obj.models[model_name].rotation_matrix_ = None
            print(f"[{model_name}] The best rotation found (priorimax) is {None}.")
        elif best == 1:
            ifa_obj.models[model_name].loadings_ = pre_mod.loadings_
            ifa_obj.models[model_name].rotation_matrix_ = pre_mod.rotation_matrix_
            actual_rot = pre_name
            print(f"[{model_name}] The best rotation found (priorimax) is pre-defined ({pre_name}).")
        elif best == 2:
            actual_rot = "priorimax"
            print(f"[{model_name}] The best rotation found (priorimax) is "
                  f"\n{ifa_obj.models[model_name].rotation_matrix_}.")

        return actual_rot

    def _callback(self, xk, *_):
        # The callback function for optimization

        # Try to recover the last "solution" if optimization fails
        self._last_best_rotation_matrix = xk

        # Terminate if optimization is taking too long
        elapsed = time.time() - self._start_time
        if elapsed > self.max_time:
            warnings.warn("Stopping optimization due to timeout based on `max_time`. "
                          "Falling back to last best result found.", RuntimeWarning)
            raise StopIteration(f"Time limit exceeded.")


class InterpretableFA:
    """
    The class for interpretable factor analysis, including priorimax rotation.

    The class:
        1) Can fit factor models by wrapping `factor_analyzer.factor_analyzer.FactorAnalyzer` from the
        factor_analyzer package
        2) Provides several indices and visualizations for assessing factor models
        3) Implements the priorimax factor rotation / procedure

    Parameters
    ----------
    data_: :obj: `pandas.core.frame.DataFrame`
        The data to be used for fitting factor models. Can be either the raw data or the correlation matrix.
    is_corr_matrix: bool, optional
        `True` if the data supplied is a correlation matrix and `False` otherwise. Defaults to `True`.
    sample_size: int, optional
        The sample size or `None`, if `is_corr_matrix` is `True`. Otherwise, this is ignored and is set to the number
        of rows in `data_`.

    Attributes
    ----------
    data_: :obj: `pandas.core.frame.DataFrame`
        The data used for fitting factor models.
    is_corr_matrix: bool
        `True` if the data is a correlation matrix and `False` otherwise.
    sample_size: int
        The sample size
    models: dict
        The dictionary containing the saved or fitted models, where the keys are the model names and the values are
        the models. Note that a model must be stored in this dictionary in order to analyze them further. Note that the
        models are `factor_analyzer.factor_analyzer.FactorAnalyzer` objects. Thus, they have the `loadings_`, `corr_`,
        `rotation_matrix_`, `structure_`, and `phi_` attributes. It was extended to have the `is_orthogonal_` and
        `prior_` attributes, as well. The `is_orthogonal_` attribute is `True` if an orthogonal rotation was applied
        to the model and `False` otherwise. The `prior_` attribute contains the prior matrix used for the priorimax
        rotation or the list of statements used to compute semantic similarities for the prior matrix. If not
        applicable, it is `None`.
    kmo: tuple
        The KMOs per item and overall KMO
    sphericity: tuple
        The test statistic and p-value of Bartlett's Test for Sphericity
    """

    use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_model = None

    def __init__(self, data_, is_corr_matrix=False, sample_size=None):
        """
        Initializes the InterpretableFA object. Note that the first time `InterpretableFA.__init__` is called with
        `prior` set to `None`, the class method `InterpretableFA.load_use_model` is run to load the Universal
        Sentence Encoder. If `prior` is not `"semantics"` or `InterpretableFA.load_use_model` has already been called
        (i.e., `InterpretableFA.use_model` is not `None`), `InterpretableFA.load_use_model` will not be called anymore.
        """

        # Initial arg checks
        if not isinstance(data_, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe")
        if data_.shape[1] != data_.select_dtypes(include=np.number).shape[1]:
            raise ValueError("all columns of the dataframe must be numeric")

        if not isinstance(is_corr_matrix, bool):
            raise TypeError("is_corr_matrix must be bool")

        # Initial values
        self.data_ = data_
        self.is_corr_matrix = is_corr_matrix
        self.sample_size = None
        self.models = {}
        self.kmo = None
        self.sphericity = None

        # In case the manifest variables are not standardized
        # Right now, they are always standardized
        self._scaler = 1

        # Set values and further arg checks
        if self.is_corr_matrix:
            if sample_size is not None:
                try:
                    self.sample_size = int(sample_size)
                except ValueError:
                    raise TypeError("the sample size must be either None, int, or coercible to int")
                if self.sample_size < 1:
                    raise ValueError("the sample size must be at least 1")

            if self.data_.shape[0] != self.data_.shape[1]:
                raise ValueError("the data correlation matrix must be a square matrix")

            for row in range(self.data_.shape[0]):
                for col in range(row + 1):
                    val = self.data_.iloc[row, col]
                    try:
                        float(val)
                    except ValueError:
                        raise TypeError("entries in the data correlation matrix must be float or coercible to float")
                    if not np.isclose(val, self.data_.iloc[col, row]):
                        raise ValueError("the data correlation matrix must be symmetric")
                    else:
                        self.data_.iloc[col, row] = val
                    if row == col:
                        if not math.isclose(val, 1):
                            raise ValueError("the diagonal entries of the data correlation matrix must be 1")
                        else:
                            val = 1
                            self.data_.iloc[row, col] = val
                    if abs(val) > 1:
                        raise ValueError("entries in the data correlation matrix must be between -1 and 1, inclusive")

            if not np.all(np.linalg.eigvals(self.data_) >= 0):
                raise ValueError("the correlation matrix must be positive semi-definite")

            self.kmo = self._get_kmo()
            self.sphericity = self._get_shpericity()
        else:
            self.kmo = calculate_kmo(self.data_)
            self.sphericity = calculate_bartlett_sphericity(self.data_)
            self.sample_size = self.data_.shape[0]

    @staticmethod
    def _corr_to_pcorr(corr_mat):
        # This gets the partial correlation matrix from the correlation matrix

        pinv = -np.linalg.pinv(corr_mat)
        np.fill_diagonal(pinv, -np.diag(pinv))
        return covariance_to_correlation(pinv)

    def _get_communalities(self, model_name):
        # This gets the communalities for the factor model

        number_of_factors = self.models[model_name].loadings_.shape[1]
        self.fit_factor_model("_for_communalities_only", number_of_factors, None)
        communalities = self.models["_for_communalities_only"].get_communalities().tolist()
        self.remove_factor_model("_for_communalities_only")

        return communalities

    def _get_kmo(self):
        # This gets the KMO if data is a correlation matrix
        # Note that this is used when the data supplied is in the form of a correlation matrix

        # Arg checks
        if not self.is_corr_matrix:
            raise ValueError("the data must be a correlation matrix")

        # Initialize values
        corr = self.data_.to_numpy(copy=True)
        pcorr = self._corr_to_pcorr(self.data_.to_numpy(copy=True))
        np.fill_diagonal(corr, 0)
        np.fill_diagonal(pcorr, 0)

        # Calculate KMOs
        pcorr = pcorr ** 2
        corr = corr ** 2
        pcorr_sum = np.sum(pcorr, axis=0)
        corr_sum = np.sum(corr, axis=0)
        kmo_items = corr_sum / (corr_sum + pcorr_sum)
        corr_sum_total = np.sum(corr)
        pcorr_sum_total = np.sum(pcorr)
        kmo_total = corr_sum_total / (corr_sum_total + pcorr_sum_total)

        return kmo_items, kmo_total

    def _get_shpericity(self):
        # This performs the Bartlett's Test for Sphericity
        # Note that this is used when the data supplied is in the form of a correlation matrix

        # Arg checks
        if not self.is_corr_matrix:
            raise ValueError("the data must be a correlation matrix")

        # Initialize values
        test_stat = None
        pval = None

        # Perform test
        if self.sample_size is not None:
            corr = self.data_.to_numpy(copy=True)
            n = self.sample_size
            p = corr.shape[0]
            corr_det = np.linalg.det(corr)
            test_stat = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
            dof = p * (p - 1) / 2
            pval = chi2.sf(test_stat, dof)

        return test_stat, pval

    @staticmethod
    def generate_grouper_prior(size, groupings):
        """
        Creates a matrix that groups indices together. The matrix can be used as a prior or soft constraints matrix.

        Parameters
        ----------
        size: int
            The total number of variables that will be partitioned (partially or completely).
        groupings: list
            The groupings for the variables or indices. It must be a list of lists, where each nested list is a list of
            positive integers. Particularly, it must be a partition of [1, 2, ..., `size`] or of a subset of
            [1, 2, ..., `size`]. For instance, `groupings` can be [[1, 2, 3], [4, 5], [6, 7, 8]]. A partial partition
            is also allowed, such as [[1, 2, 3], [6, 7, 8]].

        Returns
        ----------
        prior_matrix: :obj: `numpy.ndarray`
            The prior or soft constraints matrix that groups the variables according to the partition supplied. The
            elements are either 0, 1, or `None`.
        """

        # Arg checks
        if not isinstance(size, int):
            raise TypeError("size must be an integer")
        if size < 1:
            raise ValueError("size must be positive")

        if not isinstance(groupings, list):
            raise TypeError("groupings must be a list")

        items = [item for group in groupings for item in group]
        if not all(isinstance(item, int) for item in items):
            raise TypeError("all elements of each sublist in groupings must be an integer")
        if len(items) != len(set(items)):
            raise ValueError("the elements of groupings must be mutually exclusive")
        if not set(items) <= set(range(1, size + 1)):
            raise ValueError("groupings must partition [1, 2, ..., `size`] (or a subset of it)")

        # Construct soft constraints matrix
        prior_matrix = np.zeros(shape=(size, size), dtype=float)
        for group in groupings:
            for pair in product(group, group):
                prior_matrix[pair[0] - 1, pair[1] - 1] = 1.0
        whole_set = list(range(1, size + 1))
        not_present = [item for item in whole_set if item not in items]
        for item in not_present:
            for _ in range(size):
                prior_matrix[item - 1, _] = None
                prior_matrix[_, item - 1] = None

        return prior_matrix

    @classmethod
    def load_use_model(cls):
        """
        This loads the Universal Sentence Encoder.
        """

        # This loads the Universal Sentence Encoder from Cer et al. (2018) into memory
        cls.use_model = hub.load(cls.use_url)

    @staticmethod
    def calculate_semantic_similarity(statements):
        # This gets the semantic similarity matrix.

        if InterpretableFA.use_model is None:
            InterpretableFA.load_use_model()

        embeddings = InterpretableFA.use_model(statements)
        dots = np.inner(embeddings, embeddings)
        for i in product(range(dots.shape[0]), range(dots.shape[0])):
            # Absolute values may exceed 1 due to precision errors
            # This clips the values to [-1, 1]
            dots[i[0], i[1]] = min(max(-1, dots[i[0], i[1]]), 1)

        return 1 - (1 / math.pi) * np.arccos(dots)

    def calculate_variable_factor_correlations(self, model_name):
        """
        This calculates the correlations between each variable and each factor (i.e., Cov(X, F)). The entry at the
        ith row and jth column is the correlation between the ith variable and the jth factor.

        Parameters
        ----------
        model_name: str
            The name of the model for which the variable-factor correlations should be obtained.

        Returns
        ----------
        variable_factor_correlations: :obj: `numpy.ndarray`
            The variable-factor correlations matrix. The entry at the ith row and jth column is the correlation
            between the ith variable and the jth factor of the specified model.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Calculate correlations
        model = self.models[model_name]
        if model.is_orthogonal_:
            variable_factor_correlations = model.loadings_
        else:
            variable_factor_correlations = model.loadings_ @ model.phi_
        variable_factor_correlations = variable_factor_correlations / self._scaler

        return variable_factor_correlations

    def calculate_loading_similarity(self, model_name):
        """
        This calculates the loading similarities for each pair of components or variables for the
        specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which the loading similarities should be obtained.

        Returns
        ----------
        loading_similarity: :obj: `numpy.ndarray`
            The loading similarity matrix
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Calculate similarities
        correlations = self.calculate_variable_factor_correlations(model_name)
        num_of_vars = correlations.shape[0]
        loading_similarity = np.ones(shape=(num_of_vars, num_of_vars))
        for i in range(num_of_vars):
            for j in range(i):
                x_1 = correlations[i, :]
                x_2 = correlations[j, :]
                val = 1 - math.sqrt((1 / 2) * np.sum(((x_1 ** 2) - (x_2 ** 2)) ** 2))
                loading_similarity[i, j] = val
                loading_similarity[j, i] = val

        return loading_similarity

    def generate_multiset(self, model_name):
        """
        This generates the multiset containing the set of all ordered pairs of loading similarities and
        prior information for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which the multiset should be obtained.

        Returns
        ----------
        multiset: list or `None`
            The multiset, a list of ordered pairs (tuples). The first values are prior similarities.
            The second values are the corresponding loading similarities. If no prior was specified, it is `None`.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # The first value in the ordered pair is the prior similarity
        # The second value is the loading similarity
        num_of_vars = self.data_.shape[1]
        multiset = []
        loading_similarity = self.calculate_loading_similarity(model_name)
        prior = self.models[model_name].prior_
        if prior is None:
            return None
        elif isinstance(prior, list):
            prior = self.calculate_semantic_similarity(prior)

        for i in range(num_of_vars):
            for j in range(i):
                if not pd.isna(prior[i, j]):
                    multiset.append((prior[i, j], loading_similarity[i, j]))

        return multiset

    def calculate_tau(self, model_name):
        """
        This calculates the tau component of the V-index (interpretability index) for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which tau should be obtained.

        Returns
        ----------
        tau: float or `None`
            The tau component of the V-index for the specified model. If no prior was specified, then `None`.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Initialize values
        multiset = self.generate_multiset(model_name)
        if multiset is None:
            return None

        x = []
        y = []
        n = len(multiset)

        # Separate the set of ordered pairs into two lists and then calculate
        for i in range(n):
            x.append(multiset[i][0])
            y.append(multiset[i][1])
        tau = (1 / 2) * (kendalltau(x, y, variant="b").statistic + 1)

        return tau

    def calculate_theta(self, model_name):
        """
        This calculates the theta component of the V-index (interpretability index) for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which theta should be obtained.

        Returns
        ----------
        theta: float or `None`
            The theta component of the V-index for the specified model. If no prior was specified, then `None`.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Initialize values
        multiset = self.generate_multiset(model_name)
        if multiset is None:
            return None

        x = []
        y = []
        n = len(multiset)

        # Separate the set of ordered pairs into two lists and then calculate
        for i in range(n):
            x.append(multiset[i][0])
            y.append(multiset[i][1])

        x = np.array(x)
        y = np.array(y)
        theta = n * np.sum(x * y) - np.sum(x) * np.sum(y)
        theta = theta / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
        theta = (1 / math.pi) * np.arctan(theta) + 1 / 2

        return theta

    def calculate_v_index(self, model_name):
        """
        This calculates the V-index (interpretability index) for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which the V-index should be obtained.

        Returns
        ----------
        v_index: float or `None`
            The V-index for the specified model. If no prior was specified, then `None`.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Compute
        theta = self.calculate_theta(model_name)
        tau = self.calculate_tau(model_name)

        if theta is None or tau is None:
            return None

        v_index = math.sqrt(tau * theta)

        return v_index

    def calculate_indices(self, model_name):
        """
        This calculates several indices for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which the indices should be obtained.

        Returns
        ----------
        result: dict
            A dictionary containing the indices with the following keys:
                1) `model`: str, the model name
                2) `v_index`: float or `None`, the V-index
                3) `communalities`: :obj: `numpy.ndarray`, the communalities (the communality of the first variable is
                the first element and so on)
                4) `sphericity`: tuple, the test statistic (float) and the p-value (float), in that order, for
                Bartlett's Sphericity Test
                5) `kmo`: tuple, the KMO score per variable (:obj: `numpy.ndarray`) and the overall KMO score (float),
                in that order
                6) `sufficiency`: tuple or None, the test statistic (float), the degrees of freedom (int), and the
                p-value (float), in that order, for the sufficiency test (`None` if calculations fail)
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Compute
        v = self.calculate_v_index(model_name)
        communalities = self._get_communalities(model_name)
        sphericity = self.sphericity
        kmo = self.kmo
        result = {
            "model": model_name,
            "v_index": v,
            "communalities": communalities,
            "sphericity": sphericity,
            "kmo": kmo,
            "sufficiency": None
        }

        # Sufficiency is not supported if the data supplied is a correlation matrix
        if not self.is_corr_matrix:
            try:
                result["sufficiency"] = self.models[model_name].sufficiency(self.data_.shape[0])
            except Exception as ex:
                print(ex)

        return result

    def _check_prior(self, prior, rotation):
        result = {
            "pass": True,
            "message": "Passed.",
            "processed_prior": None
        }

        prior = prior.copy() if prior is not None else None

        if prior is None:
            if rotation == "priorimax":
                result["pass"] = False
                result["message"] = "a prior matrix is required for priorimax"
                return result
        elif isinstance(prior, np.ndarray):
            if len(prior.shape) != 2:
                result["pass"] = False
                result["message"] = "the shape of the prior matrix must be 2"
                return result

            if prior.shape[0] != self.data_.shape[1] or prior.shape[1] != self.data_.shape[1]:
                result["pass"] = False
                result["message"] = ("the number of rows (or columns) of the prior matrix must match the number of "
                                     "manifest variables (i.e., number of columns in the dataset)")
                return result

            for row in range(prior.shape[0]):
                for col in range(row + 1):
                    val = prior[row, col]

                    if not pd.isna(val):
                        try:
                            val = float(val)
                            prior[row, col] = val
                        except ValueError:
                            result["pass"] = False
                            result["message"] = "all entries in the prior matrix must be either a float or None"
                            return result

                    if pd.isna(val):
                        if pd.isna(prior[col, row]):
                            continue
                        else:
                            result["pass"] = False
                            result["message"] = "the prior matrix must be symmetric"
                            return result
                    else:
                        if pd.isna(prior[col, row]):
                            result["pass"] = False
                            result["message"] = "the prior matrix must be symmetric"
                            return result
                        else:
                            if not np.isclose(val, prior[col, row]):
                                result["pass"] = False
                                result["message"] = "the prior matrix must be symmetric"
                                return result
                            else:
                                prior[col, row] = val
        elif isinstance(prior, list):
            if len(prior) != self.data_.shape[1]:
                result["pass"] = False
                result["message"] = ("the number of statements for the prior must match the number of columns in the "
                                     "data")
                return result

            for statement in prior:
                if not isinstance(statement, str):
                    result["pass"] = False
                    result["message"] = "the statements for the prior matrix must be strings"
                    return result
        else:
            result["pass"] = False
            result["message"] = "prior must be a 2D numpy array, a list of strings, or `None`"
            return result

        result["processed_prior"] = prior

        return result

    def fit_factor_model(self, model_name, n_factors=3, rotation="priorimax", prior=None, is_global=False, num_starts=1,
                         samp_points=500, max_time=300.0, method="minres", use_smc=True, bounds=(0.005, 1),
                         impute="median", svd_method="randomized", rotation_kwargs=None):
        """
        This fits the factor model (and saves it in `self.models`). This extends
        `factor_analyzer.factor_analyzer.FactorAnalyzer` from the factor_analyzer package to include
        the priorimax rotation.

        Parameters
        ----------
        model_name: str
            The name of the model.
        n_factors: int, optional
            The `n_factors` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is `3`.
        rotation: str, optional
            The type of rotation to perform after fitting the factor model. If set to `None`, no rotation will be
            performed. Possible values include:

                a) priorimax (orthogonal rotation)
                b) varimax (orthogonal rotation)
                c) promax (oblique rotation)
                d) oblimin (oblique rotation)
                e) oblimax (orthogonal rotation)
                f) quartimin (oblique rotation)
                g) quartimax (orthogonal rotation)
                h) equamax (orthogonal rotation)

            Defaults to '"priorimax"'. Note that if `rotation` is '"priorimax"', the model is fit without
            rotation first with `factor_analyzer.factor_analyzer.FactorAnalyzer`. Then, `loadings_` and
            `rotation_matrix_` are updated with the new matrices (and these are the only attributes that are updated).
        prior: :obj: `numpy.ndarray`, list of str, or `None`,  optional
            The prior matrix that will be used for priorimax. If an array is given, then the prior matrix is used as is.
            If a list of strings are given, then the semantic similarity matrix will be used as the prior matrix.
            If `None`, then no prior matrix is used. Required only for priorimax. Default is `None`.
        is_global: bool, optional
            If this is `True`, then the problem of finding the priorimax rotation will be treated as a global
            optimization problem. Otherwise, local optimization is used. Note that this is ignored if the priorimax
            rotation is not used. The default value is `False`. This is ignored when `rotation` is not '"priorimax"'.
        num_starts: int, optional
            This is the number of random starts (i.e., number of local optimizations) used for finding the priorimax
            rotation, if `is_global` is `False`. The default value is 1. Note that the first start is always taken to
            be the identity matrix, so that the starts are the identity matrix and (random_starts - 1) random rotation
            matrices. This is ignored if `is_global` is `True`. This is ignored when `rotation` is not '"priorimax"'.
        samp_points: int, optional
            This dictates the number of sampling points used in the construction of the simplicial complex for the
            SHGO algorithm, if `is_global` is `True`. The default value is 500. The total number of sampling points used
            is (samp_points * ((T^2 + T) / 2)), where T is the number of factors. This is ignored if `is_global` is
            `False`. This is ignored when `rotation` is not '"priorimax"'.
        max_time: float, optional
            This is the maximum amount of time in seconds for which an optimizer will run to find the priorimax
            rotation. If `max_time` is 0 or negative, then the pre-defined orthogonal rotation (e.g., varimax,
            equamax, etc.) with the best index value is selected (i.e., the priorimax procedure is performed on the
            set of pre-defined orthogonal rotations). The default value is `300.0`. This is ignored when `rotation`
            is not '"priorimax"'.
        method : {'minres', 'ml', 'principal'}, optional
            The `method` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is '"minres"'.
        use_smc : bool, optional
            The `use_smc` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is `True`.
        bounds : tuple, optional
            The `bounds` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is the
            tuple `(0.005, 1)`.
        impute : {'drop', 'mean', 'median'}, optional
            The `impute` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is '"median"'.
        svd_method : {‘lapack’, ‘randomized’}
            The `svd_method` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is
            '"randomized"'.
        rotation_kwargs: optional
            The `rotation_kwargs` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is
            `None`.

        Returns
        ----------
        model: :obj: `factor_analyzer.factor_analyzer.FactorAnalyzer`
            The model fitted. Relevant attributes include `rotation_matrix_`, `loadings_`, `corr_`, `phi_`, and the
            extended attributes `is_orthogonal_` and `prior_`.
        actual_rot: str or `None`
            The actual rotation method used. If `rotation` is not "priorimax", this will be the same as
            `rotation`. If `rotation` is "priorimax", this will be "priorimax" if the optimization routine found
            a solution that is better compared to any pre-defined rotation (and the best pre-defined rotation,
            otherwise).
        """

        # Arg checks
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")
        if model_name.strip() == "" or model_name == "":
            raise ValueError("model_name must have at least one non-whitespace character")
        if model_name == "_for_communalities_only":
            raise ValueError('"_for_communalities_only" is a reserved model name')

        if rotation not in POSSIBLE_ROTATIONS and rotation is not None:
            raise ValueError(f"rotation must be one of: {POSSIBLE_ROTATIONS}")

        check_prior = self._check_prior(prior, rotation)
        if not check_prior["pass"]:
            raise ValueError(check_prior["message"])
        else:
            prior = check_prior["processed_prior"]

        if not isinstance(is_global, bool):
            raise TypeError("is_global must be bool")

        try:
            num_starts = int(num_starts)
        except ValueError:
            raise TypeError("num_starts must be an int or coercible to int")
        if num_starts < 1:
            raise ValueError("num_starts must be at least 1")

        try:
            samp_points = int(samp_points)
        except ValueError:
            raise TypeError("samp_points must be an int or coercible to int")
        if samp_points < 1:
            raise ValueError("samp_points must be at least 1")

        try:
            max_time = float(max_time)
        except ValueError:
            raise TypeError("max_time must be a float or coercible to float")

        # Fit the factor model
        priorimax_rotator = None
        if rotation == "priorimax":
            rotation = None
            priorimax_rotator = PriorimaxRotator(is_global, num_starts, samp_points, max_time)
        elif rotation == "equamax" and rotation_kwargs is None:
            rotation_kwargs = {
                "kappa": n_factors / (2 * self.data_.shape[1])
            }
        fa = FactorAnalyzer(n_factors, rotation, method, use_smc, self.is_corr_matrix, bounds,
                            impute, svd_method, rotation_kwargs)
        fa.fit(self.data_)
        self.models[model_name] = fa
        self.models[model_name].is_orthogonal_ = rotation is None or rotation in ORTHOGONAL_ROTATIONS
        self.models[model_name].prior_ = prior
        if priorimax_rotator is not None:
            actual_rot = priorimax_rotator.rotate(self, model_name)
        else:
            actual_rot = rotation
        model = self.models[model_name]

        return model, actual_rot

    def summarize_model(self, model_name, loadings_and_scores=True):
        """
        This returns the variable-factor correlations matrix, the indices, the loadings, and
        the estimated factor scores for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to be summarized.
        loadings_and_scores: bool
            Whether to include the loadings and the scores. Defaults to `True`.

        Returns
        ----------
        summary: dict
            A dictionary containing the result of `self.calculate_indices(model_name)` and the additional
            keys:
                1) `variable_factor_correlations`: the variable-factor correlations matrix
                2) `loadings`: the loading matrix
                3) `scores`: the estimated factor scores
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Calculate indices
        summary = self.calculate_indices(model_name)
        variable_factor = self.calculate_variable_factor_correlations(model_name)
        loadings = self.models[model_name].loadings_
        scores = self.models[model_name].transform(self.data_) if not self.is_corr_matrix else None
        summary["variable_factor_correlations"] = variable_factor
        if loadings_and_scores is True:
            summary["loadings"] = loadings
            summary["scores"] = scores

        return summary

    def analyze_model(self, model_name, sorted_=True):
        """
        This provides information about the model such as the variable-factor correlations, the communalities, and
        the Kaiser-Meyer-Olkin Sampling Adequacy statistic.

        Parameters
        ----------
        model_name: str
            The name of the model to be analyzed.
        sorted_: bool
            Whether the variables should be sorted according to their largest correlations or not. Defaults to `True`.

        Returns
        ----------
        analysis: :obj: `pandas.core.frame.DataFrame`
            A pandas DataFrame that contains the variables, variable-factor correlations, communalities, and
            per-variable KMO scores
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Calculate values
        model = self.models[model_name]
        analysis = {
            "variable": self.data_.columns.values.tolist()
        }
        corr_mat = self.calculate_variable_factor_correlations(model_name)
        for i in range(corr_mat.shape[1]):
            analysis["factor_" + str(i + 1)] = corr_mat[:, i].tolist()
        analysis["communality"] = self._get_communalities(model_name)
        analysis["kmo_msa"] = self.kmo[0].tolist()
        analysis = pd.DataFrame(analysis)
        if sorted_:
            temp = analysis.drop(["variable", "communality", "kmo_msa"], axis=1)
            analysis = analysis.reindex(temp.max(1).sort_values(ascending=False).index)

        return analysis

    def remove_factor_model(self, model_name):
        """
        This removes the specified factor model from `self.models`.

        Parameters
        ----------
        model_name: str
            The model to be removed.

        Returns
        ----------
        None
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        del self.models[model_name]

    def select_factor_model(self, models="all"):
        """
        This performs the 'priorimax' procedure (model selection based on the V-index).
        Note that this is not the rotation method.

        Parameters
        ----------
        models: '"all"' or list
            If the value is '"all"', then the procedure is performed on`self.models`. The value can also be a
            list of the model names and if so, the procedure will be performed on the specified models. Defaults to
            '"all"'.

        Returns
        ----------
        results: list
            A list of the summaries (i.e., from `self.summarize_model`) of the models, sorted from the highest
            index value to the lowest index value. The selected model is the first element of the list.
        """

        # Arg checks
        if models == "all":
            models = self.models
        elif not (isinstance(models, list) and set(models).issubset(self.models.keys())):
            raise ValueError(f"models is invalid, models must be either 'all' "
                             f"or a list that is a subset of {list(self.models.keys())}")
        else:
            raise TypeError("models must be either a list or 'all'")
        if len(models) == 0:
            raise ValueError("list cannot have a length of 0")

        # Find the best model
        results = []
        for model in models:
            results.append(self.summarize_model(model))
        results = sorted(results, key=lambda x: x["v_index"])

        return results

    def interp_plot(self, model_name, title=None, w=10, h=6):
        """
        This generates the interpretability plot with LOWESS curve for the specified model.

        Parameters
        ----------
        model_name: str
            The name of the model for which the V-index should be visualized.
        w: float, optional
            The width of the figure in inches.
        h: float, optional
            The height of the figure in inches.
        title: str, optional
            The title of the plot. If `None`, a default title will be used.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Get prior and loading similarities
        multiset = self.generate_multiset(model_name)
        if multiset is None:
            return None

        x = [item[0] for item in multiset]
        y = [item[1] for item in multiset]
        df = pd.DataFrame({
            'Prior Information': x,
            'Loading Similarity': y
        })

        # Create plot
        plt.figure(figsize=(w, h))
        sns.regplot(data=df, x='Prior Information', y='Loading Similarity', lowess=True,
                    scatter_kws={'alpha': 0.5, 'edgecolor': 'w'}, line_kws={'color': 'red'})
        plt.title(title or f'Interpretability Plot with LOWESS Curve - Model: {model_name}')
        plt.xlabel('Semantic Similarity' if isinstance(self.models[model_name].prior_, list) else 'Prior Similarity')
        plt.ylabel('Loading Similarity')
        plt.grid(True)
        plt.show()

    def var_factor_corr_plot(self, model_name, sorted_=True, title=None, w=10, h=8, cmap=None):
        """
        This generates a heatmap of the variable-factor correlations with a user-defined colormap or a default colormap.

        Parameters
        ----------
        model_name: str
            The name of the model for which the variable-factor correlations should be visualized.
        sorted_: bool
            Whether the variables should be sorted according to their largest correlations or not.
        title: str, optional
            The title of the heatmap plot. If `None`, a default title will be used.
        w: float, optional
            The width of the figure in inches.
        h: float, optional
            The height of the figure in inches.
        cmap: str or :obj:`matplotlib.colors.Colormap`, optional
            The colormap to use for the heatmap. Can be a string for predefined colormaps or an
            obj:`matplotlib.colors.Colormap`. If `None`, a default colormap will be used.
        """

        # Arg checks
        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        # Get correlations
        corr_mat = self.calculate_variable_factor_correlations(model_name)

        df_corr = pd.DataFrame(corr_mat, columns=[f'Factor {i + 1}' for i in range(corr_mat.shape[1])],
                               index=self.data_.columns)
        if sorted_:
            sorted_index = df_corr.abs().max(axis=1).sort_values(ascending=False).index
            df_corr = df_corr.loc[sorted_index]

        # Get colors
        if cmap is None:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'default_cmap',
                ['darkred', 'red', 'orange', 'white', 'lightblue', 'blue', 'darkblue'],
                N=1024
            )
        elif isinstance(cmap, str):
            if cmap not in plt.colormaps():
                raise ValueError(f"Predefined colormap '{cmap}' is not available.")

        # Create plot
        plt.figure(figsize=(w, h))
        sns.heatmap(df_corr, cmap=cmap, center=0, annot=True, fmt='.2f', linewidths=0.5)
        plt.title(title or f'Variable-Factor Correlation Heatmap - Model: {model_name}')
        plt.show()
