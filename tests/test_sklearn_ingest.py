import numpy as np

from edgeforge.ingest import sklearn_ingest


class _NotAModel:
    """Module-level (not local) so it's actually picklable -- a local class isn't."""


def test_tree_ingest_structure_and_footprint(tree_clf_path):
    result = sklearn_ingest.ingest(tree_clf_path)
    ir = result.ir
    assert ir.kind == "classical"
    assert ir.task == "classification"
    assert ir.input_spec.shape == (4,)
    assert ir.output_spec.size == 3  # 3 iris classes
    assert ir.nodes[0].op == "tree"
    # Tree data must be real weight tensors (not raw attrs), or footprint would silently read as 0 --
    # this is a regression test for that exact bug.
    assert ir.weight_bytes() > 0


def test_tree_reference_fn_matches_predict_proba(tree_clf_path):
    import pickle

    with open(tree_clf_path, "rb") as f:
        model = pickle.load(f)
    result = sklearn_ingest.ingest(tree_clf_path)
    rng = np.random.default_rng(0)
    samples = result.input_sampler(rng, 10)
    for s in samples:
        cls, out = result.reference_fn(s)
        proba = model.predict_proba(s.reshape(1, -1))[0]
        assert cls == int(np.argmax(proba))
        np.testing.assert_allclose(out, proba, atol=1e-6)


def test_binary_logistic_regression_expands_to_two_classes(logreg_bin_path):
    result = sklearn_ingest.ingest(logreg_bin_path)
    ir = result.ir
    assert ir.output_spec.size == 2  # not 1 -- see sklearn_ingest's _binary_expand
    w = ir.tensors[ir.nodes[0].attrs["weight"]]
    assert w.shape == (2, 4)
    np.testing.assert_array_equal(w.data[0], np.zeros(4))  # class-0 row is the fixed zero logit


def test_mlp_hidden_relu_is_in_place(mlp_clf_path):
    result = sklearn_ingest.ingest(mlp_clf_path)
    ir = result.ir
    act_nodes = [n for n in ir.nodes if n.op == "activation"]
    assert act_nodes
    for n in act_nodes:
        assert n.inputs == n.outputs  # in-place, per the RAM-saving design in sklearn_ingest


def test_unsupported_model_type_raises(tmp_path):
    import pickle

    from edgeforge.errors import UnsupportedModelError

    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump(_NotAModel(), f)
    try:
        sklearn_ingest.ingest(path)
        assert False, "expected UnsupportedModelError"
    except UnsupportedModelError as e:
        assert "unsupported estimator type" in str(e)
