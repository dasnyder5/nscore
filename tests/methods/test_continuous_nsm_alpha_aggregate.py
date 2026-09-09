"""Tests for ContinuousNsmTest alpha state and mirrored NSM p-value metadata."""

import numpy as np
import pytest

from statistical_comparison_core import Decision, Hypothesis
from nscore.nonparametric_nsm import ContinuousNsmTest, MirroredContinuousNsmTest


C_BINARY = np.array([0.0, 1.0])


# ---------------------------------------------------------------------------
# ContinuousNsmTest alpha state
# ---------------------------------------------------------------------------

class TestAlphaProperty:
    def test_alpha_is_read_only(self):
        t = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY)
        assert t.alpha == 0.05
        assert t._cutoff == pytest.approx(1.0 / 0.05)

        with pytest.raises(AttributeError):
            t.alpha = 0.01

        assert t.alpha == 0.05
        assert t._cutoff == pytest.approx(1.0 / 0.05)

    def test_rejected_alpha_change_does_not_reset_history(self):
        t = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY)
        # Feed a few data points to build history.
        t.step(0, 1)
        t.step(0, 1)
        martingale_before = t._martingale
        p_value_before = t._p_value
        time_before = t._t
        lambda_len_before = len(t._store_lambda_params)

        with pytest.raises(AttributeError):
            t.alpha = 0.01

        assert t._martingale == martingale_before
        assert t._p_value == p_value_before
        assert t._t == time_before
        assert len(t._store_lambda_params) == lambda_len_before


# ---------------------------------------------------------------------------
# MirroredContinuousNsmTest alpha immutability
# ---------------------------------------------------------------------------

class TestMirroredAlphaFanout:
    def test_alpha_assignment_is_rejected_by_children(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        assert m._test_for_alternative.alpha == 0.05
        assert m._test_for_null.alpha == 0.05
        assert m._test_for_alternative._cutoff == pytest.approx(20.0)
        assert m._test_for_null._cutoff == pytest.approx(20.0)

        with pytest.raises(AttributeError):
            m.alpha = 0.01

        assert m._test_for_alternative.alpha == 0.05
        assert m._test_for_null.alpha == 0.05
        assert m._test_for_alternative._cutoff == pytest.approx(20.0)
        assert m._test_for_null._cutoff == pytest.approx(20.0)

    def test_rejected_alpha_change_does_not_reset_child_or_wrapper_history(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        m.step(0, 1)
        m.step(0, 1)

        alt_mart = m._test_for_alternative._martingale
        alt_pv = m._test_for_alternative._p_value
        alt_t = m._test_for_alternative._t
        null_mart = m._test_for_null._martingale
        null_pv = m._test_for_null._p_value
        null_t = m._test_for_null._t
        wrapper_pv = m._p_value
        wrapper_pt = m._p_type

        with pytest.raises(AttributeError):
            m.alpha = 0.01

        assert m._test_for_alternative._martingale == alt_mart
        assert m._test_for_alternative._p_value == alt_pv
        assert m._test_for_alternative._t == alt_t
        assert m._test_for_null._martingale == null_mart
        assert m._test_for_null._p_value == null_pv
        assert m._test_for_null._t == null_t
        assert m._p_value == wrapper_pv
        assert m._p_type == wrapper_pt


# ---------------------------------------------------------------------------
# MirroredContinuousNsmTest wrapper aggregate metadata
# ---------------------------------------------------------------------------

class TestWrapperAggregate:
    def test_wrapper_p_value_is_min_of_children(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        # Feed data that should favour the alternative side.
        m.step(0, 1)
        alt_pv = m._test_for_alternative._p_value
        null_pv = m._test_for_null._p_value
        assert m._p_value == pytest.approx(min(alt_pv, null_pv))
        assert m._p_value == pytest.approx(min(m._p_value_less, m._p_value_more))

    def test_wrapper_p_type_alternative_on_tie(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        # At construction both children have _p_value=1.0.
        # lambda_parameter starts at 0 for both children, so stepping with
        # any data produces martingale multiplier = 1.0, keeping both child
        # _p_value == 1.0 -- a deterministic tie.
        m.step(0.5, 0.5)
        assert m._test_for_alternative._p_value == pytest.approx(
            m._test_for_null._p_value
        )
        assert m._p_type == "alternative"

    def test_wrapper_aggregate_does_not_corrupt_child_state(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        # Step with data favouring alternative to diverge child p-values.
        m.step(0, 1)
        m.step(0, 1)
        m.step(0, 1)
        alt_pv = m._test_for_alternative._p_value
        null_pv = m._test_for_null._p_value
        wrapper_pv = m._p_value
        # Children should have diverged: alt p-value drops, null stays at 1.0.
        assert alt_pv < null_pv
        # Wrapper aggregate is the min, which equals alt_pv.
        assert wrapper_pv == pytest.approx(alt_pv)
        # If wrapper assignment had fanned out, null child would also be alt_pv.
        assert m._test_for_null._p_value == pytest.approx(null_pv)
        assert m._test_for_null._p_value != wrapper_pv
        # Children should not have _p_type at all.
        assert not hasattr(m._test_for_alternative, "_p_type")
        assert not hasattr(m._test_for_null, "_p_type")

    def test_wrapper_tracks_directional_p_values(self):
        less = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        for _ in range(3):
            less.step(0, 1)

        assert less._p_value_less == pytest.approx(
            less._test_for_alternative._p_value_less
        )
        assert less._p_value_more == pytest.approx(less._test_for_null._p_value_more)
        assert less._p_value == pytest.approx(min(less._p_value_less, less._p_value_more))

        more = MirroredContinuousNsmTest(
            Hypothesis.P0MoreThanP1, alpha=0.05, c=C_BINARY
        )
        for _ in range(3):
            more.step(1, 0)

        assert more._p_value_less == pytest.approx(more._test_for_null._p_value_less)
        assert more._p_value_more == pytest.approx(
            more._test_for_alternative._p_value_more
        )
        assert more._p_value == pytest.approx(min(more._p_value_less, more._p_value_more))

    def test_p_value_initialized_at_construction(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        assert m._p_value == 1.0
        assert m._p_value_less == 1.0
        assert m._p_value_more == 1.0
        assert m._p_type == "N/A"

    def test_ranking_p_value_matches_declared_alpha_rejection_invariant(self):
        rng = np.random.default_rng(0)
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="ranking",
        )

        for step in range(1, 26):
            result = m.step(
                rng.binomial(1, 0.2),
                rng.binomial(1, 0.8),
            )

            wrapper_p_rejects = bool(m._p_value <= m.alpha)
            test_rejects = result.decision is not Decision.FailToDecide
            assert wrapper_p_rejects == test_rejects, (
                f"step {step}: wrapper _p_value <= alpha disagrees with "
                f"decision {result.decision}"
            )

    def test_comparison_mode_p_value_remains_plain_directional_min(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="comparison",
        )

        for _ in range(3):
            m.step(0, 1)

        raw_min = min(m._p_value_less, m._p_value_more)
        assert m._p_value == pytest.approx(raw_min)

    def test_ranking_no_ties_p_value_remains_plain_directional_min(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="ranking_no_ties",
        )

        for _ in range(3):
            m.step(0, 1)

        raw_min = min(m._p_value_less, m._p_value_more)
        assert m._p_value == pytest.approx(raw_min)

    def test_ranking_transforms_only_aggregate_p_value(self):
        seq_0 = [0, 0, 0]
        seq_1 = [1, 1, 1]
        comparison = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="comparison",
        )
        ranking = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="ranking",
        )

        for datum_0, datum_1 in zip(seq_0, seq_1):
            comparison.step(datum_0, datum_1)
            ranking.step(datum_0, datum_1)

        raw_min = min(ranking._p_value_less, ranking._p_value_more)
        assert ranking._p_value_less == pytest.approx(comparison._p_value_less)
        assert ranking._p_value_more == pytest.approx(comparison._p_value_more)
        assert comparison._p_value == pytest.approx(raw_min)
        assert ranking._p_value == pytest.approx(min(1.0, 2.0 * raw_min))

    def test_p_type_is_unchanged_by_ranking_transform(self):
        seq_0 = [0, 0, 1, 0, 0]
        seq_1 = [1, 1, 1, 0, 1]
        comparison = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="comparison",
        )
        ranking = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1,
            alpha=0.05,
            c=C_BINARY,
            inference_mode="ranking",
        )

        for datum_0, datum_1 in zip(seq_0, seq_1):
            comparison.step(datum_0, datum_1)
            ranking.step(datum_0, datum_1)

        assert ranking._p_type == comparison._p_type


# ---------------------------------------------------------------------------
# reset() restores wrapper and child state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restores_aggregate(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        m.step(0, 1)
        m.step(0, 1)
        assert m._p_value < 1.0

        m.reset()
        assert m._p_value == 1.0
        assert m._p_value_less == 1.0
        assert m._p_value_more == 1.0
        assert m._p_type == "N/A"
        assert m._test_for_alternative._martingale == 1.0
        assert m._test_for_null._martingale == 1.0
        assert m._test_for_alternative._p_value == 1.0
        assert m._test_for_null._p_value == 1.0


# ---------------------------------------------------------------------------
# Existing step / run_on_sequence expectations
# ---------------------------------------------------------------------------

class TestExistingBehavior:
    def test_step_returns_expected_info_shape(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        result = m.step(0, 1)
        assert result.decision in (
            Decision.AcceptAlternative,
            Decision.AcceptNull,
            Decision.FailToDecide,
        )
        assert "result_for_alternative" in result.info
        assert "result_for_null" in result.info

    def test_run_on_sequence_binary(self):
        m = MirroredContinuousNsmTest(
            Hypothesis.P0LessThanP1, alpha=0.05, c=C_BINARY
        )
        seq_0 = [0] * 30
        seq_1 = [1] * 30
        result = m.run_on_sequence(seq_0, seq_1)
        assert result.decision == Decision.AcceptAlternative


# ---------------------------------------------------------------------------
# ContinuousNsmTest score-grid handling
# ---------------------------------------------------------------------------

class TestScoreGrid:
    def test_nonuniform_score_grid_is_used_for_discretization(self):
        c = np.array([0.0, 0.2, 0.8, 1.0])
        test = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=c)

        prior_alpha_0 = test._alpha_0.copy()
        prior_alpha_1 = test._alpha_1.copy()
        test.step(0.79, 0.81)

        assert test._alpha_0[1] == pytest.approx(prior_alpha_0[1] + 1.0)
        assert test._alpha_0[2] == pytest.approx(prior_alpha_0[2])
        assert test._alpha_1[2] == pytest.approx(prior_alpha_1[2] + 1.0)

    def test_datum_below_grid_floor_bins_to_zero(self):
        """A datum under c[0] clamps to the lowest bin instead of underflowing.

        The previous downward linear scan had no floor at zero, so it walked
        the bin index past the start of the array and raised IndexError.
        """
        c = np.array([0.2, 0.5, 1.0])
        test = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=c)

        prior_alpha_0 = test._alpha_0.copy()
        prior_alpha_1 = test._alpha_1.copy()
        test.step(0.1, 0.6)

        assert test._alpha_0[0] == pytest.approx(prior_alpha_0[0] + 1.0)
        assert test._alpha_1[1] == pytest.approx(prior_alpha_1[1] + 1.0)

    def test_datum_outside_unit_interval_is_rejected_before_binning(self):
        """Out-of-range data is validated out, so binning never sees it.

        This is why the underflow above is only reachable through a score grid
        whose first entry is above zero, not through out-of-range data.
        """
        c = np.array([0.0, 0.5, 1.0])
        test = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=c)

        with pytest.raises(ValueError):
            test.step(-0.05, 0.75)

    def test_grid_boundary_values_bin_to_their_own_index(self):
        """Exact grid points bin to themselves, including the endpoints."""
        c = np.array([0.0, 0.2, 0.8, 1.0])

        for datum, expected_bin in ((0.0, 0), (0.2, 1), (0.8, 2), (1.0, 3)):
            test = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=0.05, c=c)
            prior_alpha_0 = test._alpha_0.copy()
            test.step(datum, 0.5)

            assert test._alpha_0[expected_bin] == pytest.approx(
                prior_alpha_0[expected_bin] + 1.0
            ), f"datum {datum} did not bin to index {expected_bin}"
