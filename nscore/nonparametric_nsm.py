"""Sequential method based on Safe Any-time Valid Inference (SAVI)

This module defines the sequential test for arbitrary partial credit observations based on
a novel nonnegative martingale construction which maintains E[M_t] = 1 w.p. 1 for all t.
"""
from typing import Union

import numpy as np
from scipy.stats import beta, dirichlet

from statistical_comparison_core import (
    Decision,
    Hypothesis,
    MirroredTestMixin,
    SequentialTestBase,
    TestResult,
)


class ContinuousNsmTest(SequentialTestBase):
    """Nonnegative supermartingale (NSM) test for continuous outcomes, discretized solely to optimize lambda.

    This class defines a novel exact nonnegative supermartingale (NSM) test for general bounded
    evaluation performance measures. This test was developed by D. Snyder, A. Badithela, H. Nishimura,
    and additional collaborators from the University of Pennsylvania, Princeton University, and the
    Toyota Research Institute (TRI).

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        c: Partial credit evaluation score vector.
        lambda_parameter: Key parameter of test structure. Is fit adaptively online; always in [0., 1.)
    """

    def __init__(
        self,
        alternative: Hypothesis,
        alpha: float,
        c: np.array,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the test object.

        Args:
            alternative: Specification of the alternative hypothesis.
            alpha: Significance level of the test.
            c: Vector of evaluation score outcomes bins. Shape is (K, )
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        self.alternative = alternative
        self._alpha = alpha

        # Assign general martingale cutoff as function of self.alpha
        self._cutoff = 1.0 / alpha

        self.c = c
        self.K = self.c.shape[0]

        # Assign dummy lambda_parameter
        self.lambda_parameter = 0.0
        self._store_lambda_params = None

        # Beta posterior for Bernoulli parameter P0.
        self._dirichlet_posterior_0 = None
        # Beta posterior for Bernoulli parameter P1.
        self._dirichlet_posterior_1 = None

        # Bayesian estimate of the Dirichlet alpha parameter and implied mean performance.
        self._estimated_p_0 = None
        self._estimated_mu_0 = None

        # Bayesian estimate of the Dirichlet alpha parameter and implied mean performance.
        self._estimated_p_1 = None
        self._estimated_mu_1 = None

        # Time state for decision information.
        self._t = None

        # Martingale value
        self._martingale = None

        self._p_value_less = None
        self._p_value_more = None
        self._less_is_alternative = None
        self._less_is_null = None
        self._p_value = None

        self.reset(verbose)

    @property
    def alpha(self):
        return self._alpha

    def _estimate_parameters(self, verbose: bool) -> None:
        """
        Method to extract and store the Bayesian posterior mean for the associated
        data-dependent beta distribution.

        Args:
            verbose: If True, print the outputs to stdout.
        """
        if verbose:
            print(("  Estimate the Bernoulli parameters from the Beta " "posteriors:"))
        self._estimated_p_0 = self._dirichlet_posterior_0.mean()
        self._estimated_mu_0 = np.dot(self.c, self._estimated_p_0)

        self._estimated_p_1 = self._dirichlet_posterior_1.mean()
        self._estimated_mu_1 = np.dot(self.c, self._estimated_p_1)
        if verbose:
            print(
                "    Estimated mean performances: "
                f"({self._estimated_mu_0:.5f}, {self._estimated_mu_1:.5f})"
            )

    def _f_of_lambda(self, Pbar, delta_P, lambda_estimate):
        f_of_lambda = 0.0
        if self.alternative is Hypothesis.P0LessThanP1:
            for i in range(self.K - 1):
                for j in range(i + 1, self.K):
                    tmp0 = np.abs(delta_P[i, j]) * np.log(
                        1.0
                        + lambda_estimate
                        * np.sign(delta_P[i, j])
                        * (self.c[j] - self.c[i])
                    )
                    tmp1 = Pbar[i, j] * np.log(
                        1.0
                        - lambda_estimate
                        * lambda_estimate
                        * (self.c[j] - self.c[i])
                        * (self.c[j] - self.c[i])
                    )
                    f_of_lambda += tmp0
                    f_of_lambda += tmp1
        else:
            for i in range(self.K - 1):
                for j in range(i + 1, self.K):
                    tmp0 = np.abs(delta_P[i, j]) * np.log(
                        1.0
                        - lambda_estimate
                        * np.sign(delta_P[i, j])
                        * (self.c[j] - self.c[i])
                    )
                    tmp1 = Pbar[i, j] * np.log(
                        1.0
                        - lambda_estimate
                        * lambda_estimate
                        * (self.c[j] - self.c[i])
                        * (self.c[j] - self.c[i])
                    )
                    f_of_lambda += tmp0
                    f_of_lambda += tmp1

        return f_of_lambda

    def _compute_optimal_lambda_long(self, verbose: bool) -> None:
        """
        Generalized method to compute the value lambda_opt which approximately maximizes
        the expected martingale growth rate (under the assumption that the Bayesian posterior
        is the true data-generating process). The optimism (treating Bayesian posterior as truth)
        is allowed in this instance, as Type-1 Error control is guaranteed for all posterior
        beliefs due to the structure of the test and constraints on lambda being in [0, 1).

        This involves maximizing a function (self._f_of_lambda()) which computes the expected
        growth rate of the martingale as a function of the beliefs about p0 and p1 [which are
        the probabilities of each of the K outcomes under the null and alternative, respectively].
        Note that in this terminology, the mean of the null and alternative are respectively
        np.dot(c, p0) and np.dot(c, p1).

        In practice, lambda is univariate and we are able to approximately solve this optimization
        via direct discretization. This is not the most efficient method in theory, but is reliable
        and takes a fixed number of calls to the function evaluation (making it a good, predictable
        method until further accelerations can be reliably implemented).

        Args:
            verbose: If True, print the outputs to stdout.
        """
        doneFlag = False
        if self.alternative is Hypothesis.P0LessThanP1:
            if self._estimated_mu_1 <= self._estimated_mu_0:
                # Current evidence suggests we are likely to shrink the martingale,
                # so set lambda to 0 to avoid this.
                self.lambda_parameter = 0.0
                doneFlag = True
        else:  # self.alternative is Hypothesis.P0MoreThanP1
            if self._estimated_mu_0 <= self._estimated_mu_1:
                # Current evidence suggests we are likely to shrink the martingale,
                # so set lambda to 0 to avoid this.
                self.lambda_parameter = 0.0
                doneFlag = True

        if doneFlag:
            pass
        else:
            # Evidence suggests we can grow in expectation

            # Compute intermediate quantities Pbar, delta_P
            p0_hat = (self._dirichlet_posterior_0.mean()).reshape(-1, 1)
            p1_hat = (self._dirichlet_posterior_1.mean()).reshape(-1, 1)

            # Estimated joint distribution of outcomes P = np.matmul(p0, p1^T)
            Phat = np.matmul(p0_hat, np.transpose(p1_hat))
            if verbose:
                print("Current Phat: ")
                print(Phat)

            # Decompose Phat into symmetric (hysteresis) and antisymmetric (signal) components.
            Pbar = np.zeros((self.K, self.K))
            delta_P = np.zeros((self.K, self.K))

            for i in range(self.K - 1):
                for j in range(i + 1, self.K):
                    # Hysteresis term
                    Pbar[i, j] = np.minimum(Phat[i, j], Phat[j, i])
                    # Signal term
                    delta_P[i, j] = Phat[i, j] - Phat[j, i]

            if verbose:
                print("Current Pbar: ")
                print(Pbar)
                print()
                print("Current delta_P: ")
                print(delta_P)

            # Set of possible lambda values
            LAMBDA_VALS = (np.arange(100) + 0.5) / 100
            F_OF_LAMBDA = np.zeros(LAMBDA_VALS.shape[0])

            # Compute the function value for all lambda in LAMBDA_VALS
            for i in range(LAMBDA_VALS.shape[0]):
                lambda_val = LAMBDA_VALS[i]
                f_lambda = self._f_of_lambda(Pbar, delta_P, lambda_val)
                F_OF_LAMBDA[i] = f_lambda

            # Take the arg-maximum
            critical_idx = np.argmax(F_OF_LAMBDA)
            try:
                # Choose the largest lambda to break ties
                self.lambda_parameter = LAMBDA_VALS[critical_idx[-1]]
            except:
                self.lambda_parameter = LAMBDA_VALS[critical_idx]

        # Store the parameter so that we can debug things later if needed. 
        self._store_lambda_params.append(self.lambda_parameter)

        # If verbose, print to stdout. 
        if verbose:
            print(
                "    Estimated optimal lambda: "
                f"{self.lambda_parameter:.5f}"
            )

    def step(
        self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Updates the test state for single new pair of continuous data.

        Args:
            datum_0: Progress datum from the first source (baseline robot policy).
            datum_1: Progress datum from the second source (test / novel robot policy).
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.

        Raise:
            ValueError: If the input data take values not in [0., 1].
            ValueError: If the input has length greater than 1
        """
        
        # Check that data is appropriately bounded in [0, 1] and has size 1
        if isinstance(datum_0, bool):
            pass
        elif (isinstance(datum_0, int) or isinstance(datum_0, float)):
            if (datum_0 < 0. or datum_0 > 1):
                raise ValueError("Invalid datum_0; int / float is outside acceptable range [0, 1]")
        elif isinstance(datum_0, list):
            if len(datum_0) > 1 or datum_0[0] < 0. or datum_0[0] > 1.:
                raise ValueError("Invalid input for datum_0")
            datum_0 = datum_0[0]
        elif isinstance(datum_0, np.ndarray):
            if (datum_0.shape[0]) > 1 or datum_0[0] < 0. or datum_0[0] > 1.:
                raise ValueError("Invalid input for datum_0")
            datum_0 = datum_0[0]
        else:
            try: 
                datum_0 = float(datum_0)
                if (datum_0 < 0. or datum_0 > 1):
                    raise ValueError("Invalid datum_0; int / float is outside acceptable range [0, 1]")
            except:
                raise TypeError("Unacceptable type for datum_0")
        
        
        if isinstance(datum_1, bool):
            pass
        elif (isinstance(datum_1, int) or isinstance(datum_1, float)):
            if (datum_1 < 0. or datum_1 > 1):
                raise ValueError("Invalid datum_1; int / float is outside acceptable range [0, 1]")
        elif isinstance(datum_1, list):
            if len(datum_1) > 1 or datum_1[0] < 0. or datum_1[0] > 1.:
                raise ValueError("Invalid input for datum_1")
            datum_1 = datum_1[0]
        elif isinstance(datum_1, np.ndarray):
            if (datum_1.shape[0]) > 1 or datum_1[0] < 0. or datum_1[0] > 1.:
                raise ValueError("Invalid input for datum_1")
            datum_1 = datum_1[0]
        else:
            try: 
                datum_1 = float(datum_1)
                if (datum_1 < 0. or datum_1 > 1):
                    raise ValueError("Invalid datum_1; int / float is outside acceptable range [0, 1]")
            except:
                raise TypeError("Unacceptable type for datum_1")

        # Special case: accept Bernoulli data in boolean form via reformatting.
        # Map boolean to scores {0, 1}, corresponding to datum values {0, K-1}
        if isinstance(datum_0, bool):
            if datum_0:
                datum_0 = 1.0
                discrete_datum_0 = self.K - 1
            else:
                datum_0 = 0.0
                discrete_datum_0 = 0
        else:
            discrete_datum_0 = int(
                np.searchsorted(self.c, datum_0 + 1e-9, side="right") - 1
            )
            discrete_datum_0 = min(max(discrete_datum_0, 0), self.K - 1)

        if isinstance(datum_1, bool):
            if datum_1:
                datum_1 = 1.0
                discrete_datum_1 = self.K - 1
            else:
                datum_1 = 0
                discrete_datum_1 = 0
        else:
            discrete_datum_1 = int(
                np.searchsorted(self.c, datum_1 + 1e-9, side="right") - 1
            )
            discrete_datum_1 = min(max(discrete_datum_1, 0), self.K - 1)

        # Henceforth: datum_0 and datum_1 are in {0, 1, ..., self.K-1}.
        # Print to stdout if verbose is True.
        if verbose:
            print(
                (
                    "Update the NSM process given new "
                    f"datum_0 == {datum_0:.3f} and datum_1 == {datum_1:.3f}."
                    f""
                )
            )

        # Increment the time index
        self._t += 1

        # Construct the martingale multiplicative increment
        if self.alternative is Hypothesis.P0LessThanP1:
            self._less_is_alternative = True
            self._less_is_null = False
            martingale_multiplier = 1.0 + (self.lambda_parameter * (datum_1 - datum_0))
            self._martingale *= martingale_multiplier
            self._p_value_less = np.minimum(self._p_value_less, 1.0 / self._martingale)
        else:
            self._less_is_alternative = False
            self._less_is_null = True
            martingale_multiplier = 1.0 + (self.lambda_parameter * (datum_0 - datum_1))
            self._martingale *= martingale_multiplier
            self._p_value_more = np.minimum(self._p_value_more, 1.0 / self._martingale)
        self._p_value = np.minimum(self._p_value_less, self._p_value_more)

        # P0 <= P1 | Test for Alternative: P0 < P1 | Test for Null: P0 > P1
        # P0 > P1 | Test for Alternative: P0 > P1 | Test for Null: P0 < P1


        # Construct test result decision and info
        # If less is alternative and pvalue_less < 1.0/self.cutoff, then accept alternative.
        # elif less is null and pvalue_less < 1.0/self.cutoff, then accept alternative.
        # Otherwise, fail to decide.

        if self._less_is_alternative and self._p_value_less <= 1.0 / self._cutoff:
            decision = Decision.AcceptAlternative
        elif self._less_is_null and self._p_value_more <= 1.0 / self._cutoff:
            decision = Decision.AcceptAlternative
        else:
            decision = Decision.FailToDecide

        info = {
            "Time": self._t,
            "P-Value": np.minimum(self._p_value_less, self._p_value_more),
        }

        # Finally, update Dirichlet posteriors and estimates of the multinoulli parameters.
        self._alpha_0[discrete_datum_0] += 1
        self._dirichlet_posterior_0 = dirichlet(alpha=self._alpha_0)

        self._alpha_1[discrete_datum_1] += 1
        self._dirichlet_posterior_1 = dirichlet(alpha=self._alpha_1)

        # Use the posteriors to estimate the empirical means
        self._estimate_parameters(verbose)

        # Use posteriors and empirical means to compute optimal lambda for next evaluation draw
        self._compute_optimal_lambda_long(verbose)

        result = TestResult(decision, info)

        return result

    def reset(self, verbose: bool = False) -> None:
        """
        Reset the partial credit NSM Test process.

        Args:
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        # 1️⃣  Initialise the martingale and p‑value
        self._martingale = 1.0
        self._p_value_less = 1.0
        self._p_value_more = 1.0
        self._p_value = 1.0

        # 2️⃣  Reset the time counter
        self._t = int(0)

        # 3️⃣  Clear the history of λ values (used for debugging)
        try:
            del self._store_lambda_params
        except:
            pass

        self._store_lambda_params = []

        # 4️⃣  Initialise Dirichlet prior counts for both streams
        self._alpha_0 = np.ones(self.K) * 2.0 / self.K
        self._alpha_1 = np.ones(self.K) * 2.0 / self.K

        # 5️⃣  Build the corresponding Dirichlet posterior objects
        self._dirichlet_posterior_0 = dirichlet(alpha=self._alpha_0)
        self._dirichlet_posterior_1 = dirichlet(alpha=self._alpha_1)

        # For alternative P1 > P0
        # 6️⃣  Print a short description of the hypothesis (if verbose)
        if self.alternative is Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")
        else:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")

        # Estimate parameters and choose lambda(t=0)
        # 7️⃣  Compute the initial posterior means and the first λ value
        self._estimate_parameters(verbose)
        self._compute_optimal_lambda_long(verbose)

class MirroredContinuousNsmTest(MirroredTestMixin, SequentialTestBase):
    """
    Mirrored NSM Test for comparing means of discrete, bounded random variables.

    This class defines the mirrored version of the NSM Test, which has the ability to
    accept the Null Hypothesis in addition to reject it. Otherwise it is the same as
    PartialCreditNsmTest.

    The significance level alpha controls the following two errors simultaneously: (1)
    probability of wrongly accepting the alternative when the null is true, and (2)
    probability of wrongly accepting the null when the alternative is true. Note that
    Bonferroni correction is not needed since the null hypothesis for one test is the
    alternative for the other; that is, the subtests reject on disjoint subsets of the
    filtration.

    Attributes:
        alternative: Specification of the alternative hypothesis.
        alpha: Significance level of the test.
        martingale: Current martingale value.
        p_value: Current p-value.
    """

    _base_class = ContinuousNsmTest
    _wrapper_owned_attributes = frozenset(
        {"_p_value", "_p_type", "_p_value_less", "_p_value_more"}
    )

    def __init__(self, alternative: Hypothesis, *args, **kwargs) -> None:
        super().__init__(alternative, *args, **kwargs)
        self._p_value_less = 1.0
        self._p_value_more = 1.0
        self._p_value = 1.0
        self._p_type = "N/A"

    def step(self,
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        """Runs the mirrored test of a single pair of partial credit data.
        This updates the constituent one-sided tests. The design of the one-sided
        tests precludes simultaneous rejection, ensuring that the test result will
        be consistent with the constituent one-sided tests and the data. 

        Args:
            datum_0: partial credit datum from the first source.
            datum_1: partial credit datum from the second source.
            verbose (optional): If True, print the outputs to stdout. Defaults to False.

        Returns:
            TestResult: Result of the hypothesis test.
        """
        if verbose:
            print("Test for Alternative:")
        result_for_alternative = self._test_for_alternative.step(
            datum_0, datum_1, verbose
        )
        if verbose:
            print("Test for Null:")
        result_for_null = self._test_for_null.step(datum_0, datum_1, verbose)

        info = {
            "result_for_alternative": result_for_alternative,
            "result_for_null": result_for_null,
        }

        if (not result_for_alternative.decision == Decision.FailToDecide) and (
            result_for_null.decision == Decision.FailToDecide
        ):
            decision = Decision.AcceptAlternative
        elif (not result_for_null.decision == Decision.FailToDecide) and (
            result_for_alternative.decision == Decision.FailToDecide
        ):
            decision = Decision.AcceptNull
        else:
            decision = Decision.FailToDecide

        alt_pv = result_for_alternative.info["P-Value"]
        null_pv = result_for_null.info["P-Value"]
        # Each child updates only the directional p-value matching its alternative.
        self._p_value_less = min(
            self._test_for_alternative._p_value_less,
            self._test_for_null._p_value_less,
        )
        self._p_value_more = min(
            self._test_for_alternative._p_value_more,
            self._test_for_null._p_value_more,
        )
        self._p_value = self.compose_wrapper_p_value(
            self._p_value_less,
            self._p_value_more,
        )
        if alt_pv <= null_pv:
            self._p_type = "alternative"
        else:
            self._p_type = "null"

        result = TestResult(decision, info)

        return result

    def reset(self, verbose: bool = False) -> None:
        """Resets the Mirrored NSM test. This amounts to instantiating 
        two one-sided tests and running the reset method on each. The verbose
        option is passed through to the subcomponent one-sided tests. 

        Args:
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        if verbose:
            print("Test for Alternative:")
        self._test_for_alternative.reset(verbose)
        if verbose:
            print("Test for Null:")
        self._test_for_null.reset(verbose)
        self._p_value_less = 1.0
        self._p_value_more = 1.0
        self._p_value = 1.0
        self._p_type = "N/A"
