import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pep_cleaning import POSITIONAL_TITLE_RE  # noqa: E402


def _is_title(name: str) -> bool:
    # Same call pep_cleaning.run() uses to drop a row / alias.
    return POSITIONAL_TITLE_RE.match(name) is not None


def test_bishop_title_after_honorific_is_filtered():
    """'Rt Rev.' before 'Bishop of' slipped past the old start-anchored regex."""
    assert _is_title("The Rt Rev. the Lord Bishop of St. Albans")


def test_archbishop_title_is_filtered():
    assert _is_title("The Most Rev. and Rt Hon. the Lord Archbishop of Canterbury DBE")


def test_normal_name_is_kept():
    assert not _is_title("John Smith")


def test_rt_rev_real_person_is_kept():
    """Decision: 'rt rev' is not a title pattern. Lord Harries is a real
    person (a life peer), so his row must survive the filter."""
    assert not _is_title("The Rt Rev. the Lord Harries of Pentregarth")


if __name__ == "__main__":
    test_bishop_title_after_honorific_is_filtered()
    test_archbishop_title_is_filtered()
    test_normal_name_is_kept()
    test_rt_rev_real_person_is_kept()
    print("All pep_cleaning tests passed.")
