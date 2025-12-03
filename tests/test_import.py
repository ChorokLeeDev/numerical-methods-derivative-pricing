def test_quantlib_import():
    import importlib
    ql = importlib.import_module('QuantLib')
    assert ql is not None
