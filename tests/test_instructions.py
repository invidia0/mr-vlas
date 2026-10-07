from vlacourse.data.instructions import RUNNING_EXAMPLE, instruction_corpus


def test_corpus_deterministic_and_contains_running_example():
    a, b = instruction_corpus(200, seed=1), instruction_corpus(200, seed=1)
    assert a == b and len(a) == 200 and a[0] == RUNNING_EXAMPLE
    assert all(s[0].isupper() and s.endswith(".") for s in a)
