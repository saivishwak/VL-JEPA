from llm_jepa.evaluation.metrics import exact_match, gsm8k_match, startswith_match


def test_exact_match_normalizes_whitespace():
    assert exact_match("a  b", "a b")


def test_startswith_match():
    assert startswith_match("Good movie because", "Good")


def test_gsm8k_match_extracts_final_answer():
    assert gsm8k_match("work\n#### 42", "gold\n#### 42")
