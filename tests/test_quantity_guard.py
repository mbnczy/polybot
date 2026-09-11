"""
Two over/under markets can only nest if they count the same thing.

The first pair to clear every execution gate on the live book was three corners
"implying" three goals. These tests pin that it is refused, and that the real
nests the one-week window is full of — a team's goals under the match's, the
first half under the full match, a ladder of corner counts — still pass.
"""

from __future__ import annotations

import pytest

from strategy.quantity_guard import is_over_under, same_quantity, statistics

M = "Al Faisaly Saudi Club vs. Al Ittihad Saudi Club"


@pytest.mark.parametrize("narrow,broad", [
    # the live case: corners under goals
    (f"{M}: Al Faisaly Saudi Club O/U 2.5 Corners", f"{M}: Al Faisaly Saudi Club O/U 2.5"),
    ("Djokovic vs. Alcaraz: Total Sets O/U 3.5", "Djokovic vs. Alcaraz: Total Games O/U 3.5"),
    ("Arsenal vs. Spurs: O/U 4.5 Cards", "Arsenal vs. Spurs: O/U 4.5 Corners"),
    ("Arsenal vs. Spurs: Arsenal O/U 3.5 Shots", "Arsenal vs. Spurs: Arsenal O/U 3.5"),
])
def test_different_statistics_do_not_nest(narrow, broad):
    assert same_quantity(narrow, broad) is False


@pytest.mark.parametrize("narrow,broad", [
    # a team's goals under the match's goals
    (f"{M}: Al Faisaly Saudi Club O/U 0.5", f"{M}: O/U 0.5"),
    # the first half under the full match
    ("Imisli FK vs. Neftchi Baku PFC: 1st Half O/U 0.5", "Imisli FK vs. Neftchi Baku PFC: O/U 0.5"),
    # a ladder of corner counts
    ("Tanzania vs. Canada: O/U 9.5 Total Corners", "Tanzania vs. Canada: O/U 7.5 Total Corners"),
    # one set's games under the match's games
    ("Cotorcea vs. Spivacov: 1st Set Games O/U 6.5", "Cotorcea vs. Spivacov: Total Games O/U 6.5"),
    # goals named explicitly, and not
    ("Arsenal vs. Spurs: O/U 2.5 Goals", "Arsenal vs. Spurs: O/U 1.5"),
])
def test_the_same_statistic_nests(narrow, broad):
    assert same_quantity(narrow, broad) is True


def test_a_margin_is_not_a_statistic():
    """"Win by 5+ points" under "win" is a genuine nest — not an over/under pair."""
    assert same_quantity("Will the Lakers win by 5+ points?", "Will the Lakers win?") is None


def test_a_team_name_is_not_a_statistic():
    """Red Star is a club, not red cards."""
    assert same_quantity("Red Star vs. Partizan: Red Star O/U 0.5",
                         "Red Star vs. Partizan: O/U 0.5") is True


def test_the_default_statistic_is_empty():
    assert statistics("Arsenal vs. Spurs: O/U 2.5") == frozenset()
    assert statistics("Arsenal vs. Spurs: O/U 2.5 Corners") == {"corners"}


def test_over_under_detection():
    assert is_over_under("X vs. Y: O/U 2.5")
    assert is_over_under("Total points over/under 220.5")
    assert not is_over_under("Will the Lakers win?")
