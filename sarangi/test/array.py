import pytest
import numpy as np
import sarangi.util


def test_flat_to_structured():
    data = np.arange(3).astype(np.float32)
    fields = ['a', 'b']
    s = sarangi.util.flat_to_structured(data[np.newaxis, :], fields=fields, dims=[2, 1])
    assert s.shape == (1,)
    assert list(s.dtype.names) == fields


def test_round_trip_flat():
    data = np.arange(3).astype(np.float32)[np.newaxis, :]
    fields = ['a', 'b']
    s = sarangi.util.flat_to_structured(data, fields=fields, dims=[2, 1])
    f = sarangi.util.structured_to_flat(s)
    assert np.allclose(data, f)
    assert f.shape == (1, 3)


def test_fat_scalar_to_flat():
   dtype = np.dtype([('a', np.float32, 2), ('b', np.float32, (1,))])
   s = np.zeros((1,), dtype=dtype)
   s['a'][:] = [1, 2]
   s['b'] = 3
   f = sarangi.util.structured_to_flat(s)
   # print('f', f)
   # print('f.shape', f.shape)
   # print('f.dtype', f.dtype)
   assert np.allclose(f, np.array([[1., 2., 3.]], dtype=np.float64))


def test_flat_scalar_to_dict():
    import numbers
    # fat scalar
    dtype = np.dtype([('a', np.float32, 2), ('b', np.float32, (1,))])
    s = np.zeros((1,), dtype=dtype)
    d = sarangi.util.structured_to_dict(s)
    assert isinstance(d['a'], list)
    assert isinstance(d['b'], numbers.Number)
    # lean scalar
    dtype = np.dtype([('a', np.float32, 2), ('b', np.float32)])
    s = np.zeros((1,), dtype=dtype)
    d = sarangi.util.structured_to_dict(s)
    assert isinstance(d['a'], list)
    assert isinstance(d['b'], numbers.Number)
    # keep fat scalar
    dtype = np.dtype([('a', np.float32, 2), ('b', np.float32, (1,))])
    s = np.zeros((1,), dtype=dtype)
    d = sarangi.util.structured_to_dict(s, keep_fat_scalars=True)
    assert isinstance(d['a'], list)
    assert isinstance(d['b'], list)


def test_round_trip_to_dict():
    d = {'a': 1, 'b': [2, 3]}
    s = sarangi.util.dict_to_structured(d)
    d2 = sarangi.util.structured_to_dict(s)
    for k, v in d2.items():
        assert v == d[k]
    for k, v in d.items():
        assert v == d2[k]





#def test_round_trip_to_dict():
#    d = {'a': 1, 'b': [[2, 3]]}
#    s = sarangi.util.dict_to_structured(d)
