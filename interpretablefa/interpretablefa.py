# Copyright 2024 Justin Philip Tuazon, Gia Mizrane Abubo

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

# interpretablefa v2.0.0

import math
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity, covariance_to_correlation
import tensorflow_hub as hub
from scipy.stats import kendalltau, chi2
import nlopt
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ORTHOGONAL_ROTATIONS = ["priorimax", "varimax", "oblimax", "quartimax", "equamax"]
OBLIQUE_ROTATIONS = ["promax", "oblimin", "quartimin"]
POSSIBLE_ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS


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
    prior: :obj: `numpy.ndarray` or `None`
        If `prior` is `None`, then the prior is generated using pairwise semantic similarities from the Universal
        Sentence Encoder. If `prior` is of class `numpy.ndarray`, it must be a 2D array (i.e., its shape must be an
        ordered pair).
    questions: list of str, optional
        The questions associated with each variable. It is assumed that the order of the questions correspond to the
        order of the columns in `data_`. For example, the first element in `questions` correspond to the first column
        of `data_`. If `prior` is not `None`, this is ignored.
    is_corr_matrix: bool, optional
        `True` if the data supplied is a correlation matrix and `False` otherwise. Defaults to `True`. Note that if
        the correlation matrix is supplied, the KMO values, sphericity test result, and estimated factor scores are not
        provided.
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
    prior: :obj: `numpy.ndarray` or `None`
        The prior used for calculating the interpretability index and for the performing priorimax rotation.
    models: dict
        The dictionary containing the saved or fitted models, where the keys are the model names and the values are
        the models. Note that a model must be stored in this dictionary in order to analyze them further.
    questions: list of str
        The list of questions used for calculating semantic similarities.
    embeddings: list or `None`
        The embeddings of the questions, used for calculating semantic similarities.
    kmo: tuple
        The KMOs per item and overall KMO
    sphericity: tuple
        The test statistic and p-value of Bartlett's Test for Sphericity
    orthogonal: dict
        The dictionary containing information on whether a model is orthogonal or not. They keys are the model names
        and each value is either `True` (if the model is orthogonal) or `False`.
    """

    use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_model = None

    def __init__(self, data_, prior, questions, is_corr_matrix=False, sample_size=None):
        """
        Initializes the InterpretableFA object. Note that the first time `InterpretableFA.__init__` is called with
        `prior` set to `None`, the class method `InterpretableFA.load_use_model` is run to load the Universal
        Sentence Encoder. If `prior` is not `None` or `InterpretableFA.load_use_model` has already been called (i.e.,
        `InterpretableFA.use_model` is not `None`), `InterpretableFA.load_use_model` will not be called anymore.
        """

        if not isinstance(data_, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe")
        if not isinstance(is_corr_matrix, bool):
            raise TypeError("is_corr_matrix must be Boolean")
        self.data_ = data_
        self.is_corr_matrix = is_corr_matrix
        self.sample_size = None
        self.models = {}
        self.orthogonal = {}
        self.embeddings = None
        self._scaler = 1
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
        if prior is None:
            if not (bool(questions) and isinstance(questions, list) and
                    all(isinstance(question, str) for question in questions)):
                raise TypeError("questions must be a list of strings")
            if data_.shape[1] != len(questions):
                raise ValueError("the length of questions must match the number of columns in data")
            self.questions = questions
            if InterpretableFA.use_model is None:
                InterpretableFA.load_use_model()
            self.prior = self._calculate_semantic_similarity()
        elif isinstance(prior, np.ndarray):
            self.questions = []
            if len(prior.shape) != 2:
                raise ValueError("the shape of prior must be 2")
            if prior.shape[0] != data_.shape[1]:
                raise ValueError("the number of rows and of columns in prior must match the number of columns in data")
            for row in range(prior.shape[0]):
                for col in range(row + 1):
                    val = prior[row, col]
                    if val is None:
                        if prior[col, row] is None:
                            continue
                        else:
                            raise ValueError("prior must be a symmetric matrix (2D numpy array)")
                    else:
                        if not np.isclose(val, prior[col, row]):
                            raise ValueError("prior must be a symmetric matrix (2D numpy array)")
                        else:
                            prior[col, row] = val
                    try:
                        float(val)
                    except ValueError:
                        raise TypeError("values in prior must either be a float, coercible to float, or None")
            self.prior = prior
        else:
            raise TypeError("prior must be a 2D numpy array or None")

    @staticmethod
    def _corr_to_pcorr(corr_mat):
        # This gets the partial correlation matrix from the correlation matrix

        pinv = -np.linalg.pinv(corr_mat)
        np.fill_diagonal(pinv, -np.diag(pinv))
        return covariance_to_correlation(pinv)

    def _get_kmo(self):
        # This gets the KMO if data is a correlation matrix

        if not self.is_corr_matrix:
            raise ValueError("the data must be a correlation matrix")
        corr = self.data_.to_numpy(copy=True)
        pcorr = self._corr_to_pcorr(self.data_.to_numpy(copy=True))
        np.fill_diagonal(corr, 0)
        np.fill_diagonal(pcorr, 0)
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

        if not self.is_corr_matrix:
            raise ValueError("the data must be a correlation matrix")
        test_stat = None
        pval = None
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
        prior_matrix = np.zeros(shape=(size, size), dtype=object)
        for group in groupings:
            for pair in product(group, group):
                prior_matrix[pair[0] - 1, pair[1] - 1] = 1
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

        cls.use_model = hub.load(cls.use_url)

    def _calculate_semantic_similarity(self):
        # This gets the semantic similarity matrix.

        if self.embeddings is None and len(self.questions) > 0:
            self.embeddings = InterpretableFA.use_model(self.questions)
        dots = np.inner(self.embeddings, self.embeddings)
        for i in product(range(dots.shape[0]), range(dots.shape[0])):
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        model = self.models[model_name]
        if self.orthogonal[model_name]:
            variable_factor_correlations = model.loadings_
        else:
            variable_factor_correlations = np.matmul(model.loadings_, model.phi_)
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
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
        multiset: list
            The multiset, a list of ordered pairs (tuples).
        """

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        num_of_vars = self.data_.shape[1]
        multiset = []
        loading_similarity = self.calculate_loading_similarity(model_name)
        prior = self.prior
        for i in range(num_of_vars):
            for j in range(i):
                if prior[i, j] is not None:
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
        tau: float
            The tau component of the V-index for the specified model.
        """

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        multiset = self.generate_multiset(model_name)
        x = []
        y = []
        n = len(multiset)
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
        theta: float
            The theta component of the V-index for the specified model.
        """

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        multiset = self.generate_multiset(model_name)
        x = []
        y = []
        n = len(multiset)
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
        v_index: float
            The V-index for the specified model.
        """

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        multiset = self.generate_multiset(model_name)
        x = []
        y = []
        n = len(multiset)
        for i in range(n):
            x.append(multiset[i][0])
            y.append(multiset[i][1])
        tau = (1 / 2) * (kendalltau(x, y, variant="b").statistic + 1)
        x = np.array(x)
        y = np.array(y)
        theta = n * np.sum(x * y) - np.sum(x) * np.sum(y)
        theta = theta / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
        theta = (1 / math.pi) * np.arctan(theta) + 1 / 2
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
                2) `v_index`: float, the V-index
                3) `communalities`: :obj: `numpy.ndarray`, the communalities (the communality of the first variable is
                the first element and so on)
                4) `sphericity`: tuple, the test statistic (float) and the p-value (float), in that order, for
                Bartlett's Sphericity Test
                5) `kmo`: tuple, the KMO score per variable (:obj: `numpy.ndarray`) and the overall KMO score (float),
                in that order
                6) `sufficiency`: tuple or None, the test statistic (float), the degrees of freedom (int), and the
                p-value (float), in that order, for the sufficiency test (`None` if calculations fail)
        """

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        v = self.calculate_v_index(model_name)
        communalities = self.models[model_name].get_communalities().tolist()
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
        if not self.is_corr_matrix:
            try:
                result["sufficiency"] = self.models[model_name].sufficiency(self.data_.shape[0])
            except Exception as ex:
                print(ex)
        return result

    @staticmethod
    def _get_rotation_matrix(x):
        # This gets the rotation matrix from the array x with (T^2 + T) / 2 elements, where T is the number of factors

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
        identity_matrix = np.identity(num_of_factors)
        i_minus_s = identity_matrix - skew_symmetric_matrix
        i_plus_s = identity_matrix + skew_symmetric_matrix
        i_s_product = np.matmul(i_minus_s, np.linalg.inv(i_plus_s))
        rotation_matrix = np.matmul(i_s_product, diag_matrix)
        return rotation_matrix

    def _get_rotated_loadings(self, x, unrotated_loadings):
        # This gets the rotated loadings

        rotation_matrix = self._get_rotation_matrix(x)
        loadings = np.matmul(unrotated_loadings, rotation_matrix)
        return loadings

    def _get_v(self, x, grad, unrotated_loadings, model=None):
        # This gets the V-index

        if model is None:
            loadings = self._get_rotated_loadings(x, unrotated_loadings)
        else:
            loadings = model.loadings_
        num_of_vars = loadings.shape[0]
        correlations = loadings / self._scaler
        prior = self.prior
        a = []
        b = []
        for i in range(num_of_vars):
            for j in range(i):
                if prior[i, j] is not None:
                    a.append(prior[i, j])
                    x_1 = correlations[i, :]
                    x_2 = correlations[j, :]
                    b.append(1 - math.sqrt((1 / 2) * np.sum(((x_1 ** 2) - (x_2 ** 2)) ** 2)))
        n = len(a)
        tau = (1 / 2) * (kendalltau(a, b, variant="b").statistic + 1)
        a = np.array(a)
        b = np.array(b)
        theta = n * np.sum(a * b) - np.sum(a) * np.sum(b)
        theta = theta / (n * np.sum(a ** 2) - (np.sum(a)) ** 2)
        theta = (1 / math.pi) * np.arctan(theta) + 1 / 2
        v = math.sqrt(tau * theta)
        return v

    @staticmethod
    def _generate_constraint(ind):
        # This generates a constraint for the signature matrix

        def _constraint(x, grad):
            return x[ind] ** 2 - 1

        return _constraint

    def _get_best_predefined(self, num_factors):
        # This gets the best rotation (in terms of the interpretability index) among the pre-defined rotations

        models = []
        indices = []
        rot_names = []
        for rot in np.setdiff1d(ORTHOGONAL_ROTATIONS, ["priorimax"]):
            temp_model = FactorAnalyzer(n_factors=num_factors, rotation=rot, is_corr_matrix=self.is_corr_matrix)
            temp_model.fit(self.data_)
            models.append(temp_model)
            rot_names.append(rot)
            indices.append(self._get_v(None, None, None, models[-1]))
        return [models[indices.index(max(indices))], max(indices), rot_names[indices.index(max(indices))]]

    def _rotate_factors(self, model_name, max_time=300.0, opt_seed=123):
        # This implements the priorimax rotation

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        try:
            max_time = float(max_time)
        except ValueError:
            raise TypeError("max_time must be a float or coercible to float")
        try:
            opt_seed = int(opt_seed)
        except ValueError:
            raise TypeError("opt_seed must be an int or coercible to int")
        none_ind = self.calculate_v_index(model_name)

        opt_ind = -1
        pre_mod, pre_ind, pre_name = self._get_best_predefined(self.models[model_name].loadings_.shape[1])
        unrotated_loadings = self.models[model_name].loadings_.copy()
        if max_time > 0:
            num_of_factors = unrotated_loadings.shape[1]
            num_of_skew_vars = int((num_of_factors * (num_of_factors - 1)) / 2)
            num_of_diag_vars = int(num_of_factors)
            num_of_mat_vars = num_of_skew_vars + num_of_diag_vars
            nlopt.srand(opt_seed)
            opt = nlopt.opt(nlopt.GN_ISRES, int(num_of_mat_vars))
            opt.set_maxtime(max_time)
            opt.set_population(200 * num_of_mat_vars)
            opt.set_lower_bounds(np.array([-1] * num_of_mat_vars))
            opt.set_upper_bounds(np.array([1] * num_of_mat_vars))
            opt.set_max_objective(lambda x, grad: self._get_v(x, grad, unrotated_loadings))
            for i in range(num_of_diag_vars):
                opt.add_equality_constraint(self._generate_constraint(num_of_skew_vars + i))
            results = opt.optimize(np.append(np.zeros(num_of_skew_vars), np.ones(num_of_diag_vars)))
            rotation_matrix = self._get_rotation_matrix(results)
            rotated_loadings = self._get_rotated_loadings(results, unrotated_loadings)
            self.models[model_name].loadings_ = rotated_loadings
            self.models[model_name].rotation_matrix_ = rotation_matrix
            opt_ind = self.calculate_v_index(model_name)
        inds = [none_ind, pre_ind, opt_ind]
        best = inds.index(max(inds))
        if best == 0:
            self.models[model_name].loadings_ = unrotated_loadings
            self.models[model_name].rotation_matrix_ = None
            print(f"[{model_name}] The best rotation found (priorimax) is {None}.")
        elif best == 1:
            self.models[model_name].loadings_ = pre_mod.loadings_
            self.models[model_name].rotation_matrix_ = pre_mod.rotation_matrix_
            print(f"[{model_name}] The best rotation found (priorimax) is pre-defined ({pre_name}).")
        elif best == 2:
            print(f"[{model_name}] The best rotation found (priorimax) is "
                  f"\n{self.models[model_name].rotation_matrix_}.")

    def fit_factor_model(self, model_name, n_factors=3, rotation="priorimax", max_time=300.0, opt_seed=1,
                         method="minres", use_smc=True, bounds=(0.005, 1), impute="median", svd_method="randomized",
                         rotation_kwargs=None):
        """
        This fits the factor model (and saves it in `self.models`). This extends
        `factor_analyzer.factor_analyzer.FactorAnalyzer` from the factor_analyzer package to include
        the priorimax rotation.

        Parameters
        ----------
        model_name: str
            The name of the model.
        n_factors: int, optional
            The `n_factors` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is 3.
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

            Defaults to 'priorimax'. Note that if `rotation` is 'priorimax', the model is fit without
            rotation first with `factor_analyzer.factor_analyzer.FactorAnalyzer`. Then, `loadings_` and
            `rotation_matrix_` are updated with the new matrices (and these are the only attributes that are updated).
        max_time: float, optional
            If `rotation` is either 'priorimax', this is the maximum time in seconds for which the
            optimizer will run to find the rotation matrix. If `max_time` is 0 or negative, then the pre-defined
            orthogonal rotation (e.g., varimax, equamax, etc.) with the best index value is selected (i.e., the
            priorimax procedure is performed on the set of pre-defined orthogonal rotations). The
            default value is 300.0 (i.e., 5 minutes). This is ignored when `rotation` is not 'priorimax'.
        opt_seed: int, optional
            This is the seed used for the optimizer (ISRES). The default value is 1.
        method : {'minres', 'ml', 'principal'}, optional
            The `method` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is 'minres'.
        use_smc : bool, optional
            The `use_smc` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is `True`.
        bounds : tuple, optional
            The `bounds` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is the
            tuple `(0.005, 1)`.
        impute : {'drop', 'mean', 'median'}, optional
            The `impute` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is 'median'.
        svd_method : {‘lapack’, ‘randomized’}
            The `svd_method` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is
            'randomized'.
        rotation_kwargs: optional
            The `rotation_kwargs` supplied to `factor_analyzer.factor_analyzer.FactorAnalyzer`. The default value is
            `None`.

        Returns
        ----------
        model: :obj: `factor_analyzer.factor_analyzer.FactorAnalyzer`
            The model fitted.
        """

        if rotation not in POSSIBLE_ROTATIONS and rotation is not None:
            raise ValueError(f"rotation must be one of: {POSSIBLE_ROTATIONS}")
        try:
            max_time = float(max_time)
        except ValueError:
            raise TypeError("max_time must be a float or coercible to float")
        self.orthogonal[model_name] = rotation is None or rotation in ORTHOGONAL_ROTATIONS
        if rotation == "priorimax":
            special_rotation = rotation
            rotation = None
        else:
            special_rotation = "none"
        fa = FactorAnalyzer(n_factors, rotation, method, use_smc, self.is_corr_matrix, bounds,
                            impute, svd_method, rotation_kwargs)
        fa.fit(self.data_)
        self.models[model_name] = fa
        if special_rotation != "none":
            self._rotate_factors(model_name, max_time, opt_seed)
        model = self.models[model_name]
        return model

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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        model = self.models[model_name]
        analysis = {
            "variable": self.data_.columns.values.tolist()
        }
        corr_mat = self.calculate_variable_factor_correlations(model_name)
        for i in range(corr_mat.shape[1]):
            analysis["factor_" + str(i + 1)] = corr_mat[:, i].tolist()
        analysis["communality"] = model.get_communalities().tolist()
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")
        del self.models[model_name]
        del self.orthogonal[model_name]

    def select_factor_model(self, models="all"):
        """
        This performs the 'priorimax' procedure (model selection based on the V-index).
        Note that this is not the rotation method.

        Parameters
        ----------
        models: 'all' or list
            If the value is 'all', then the procedure is performed on`self.models`. The value can also be a
            list of the model names and if so, the procedure will be performed on the specified models. Defaults to
            'all'.

        Returns
        ----------
        results: list
            A list of the summaries (i.e., from `self.summarize_model`) of the models, sorted from the highest
            index value to the lowest index value. The selected model is the first element of the list.
        """

        if models == "all":
            models = self.models
        elif not (isinstance(models, list) and set(models).issubset(self.models.keys())):
            raise ValueError(f"models is invalid, models must be either 'all' "
                             f"or a list that is a subset of {list(self.models.keys())}")
        else:
            raise TypeError("models must be either a list or 'all'")
        results = []
        if len(models) == 0:
            raise ValueError("list cannot have a length of 0")
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        multiset = self.generate_multiset(model_name)
        x = [item[0] for item in multiset]
        y = [item[1] for item in multiset]

        df = pd.DataFrame({
            'Prior Information': x,
            'Loading Similarity': y
        })

        plt.figure(figsize=(w, h))
        sns.regplot(data=df, x='Prior Information', y='Loading Similarity', lowess=True,
                    scatter_kws={'alpha': 0.5, 'edgecolor': 'w'}, line_kws={'color': 'red'})
        plt.title(title or f'Interpretability Plot with LOWESS Curve - Model: {model_name}')
        plt.xlabel('Prior Similarity' if self.embeddings is None else 'Semantic Similarity')
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

        if model_name not in self.models.keys():
            raise KeyError(f"model not found, model_name must be one of {list(self.models.keys())}")

        corr_mat = self.calculate_variable_factor_correlations(model_name)

        df_corr = pd.DataFrame(corr_mat, columns=[f'Factor {i + 1}' for i in range(corr_mat.shape[1])],
                               index=self.data_.columns)

        if sorted_:
            sorted_index = df_corr.abs().max(axis=1).sort_values(ascending=False).index
            df_corr = df_corr.loc[sorted_index]

        if cmap is None:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'default_cmap', ['darkred', 'red', 'orange', 'white', 'lightblue', 'blue', 'darkblue'],
                N=1024
            )
        elif isinstance(cmap, str):
            # Use a predefined colormap if the input is a string
            if cmap not in plt.colormaps():
                raise ValueError(f"Predefined colormap '{cmap}' is not available.")
            cmap = plt.get_cmap(cmap)

        plt.figure(figsize=(w, h))
        sns.heatmap(df_corr, cmap=cmap, center=0, annot=True, fmt='.2f', linewidths=0.5)
        plt.title(title or f'Variable-Factor Correlation Heatmap - Model: {model_name}')
        plt.show()
