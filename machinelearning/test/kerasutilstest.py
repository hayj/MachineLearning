x1 = xVal
y1 = yVal
x2 = iteratorToArray(asap.getTokensOnlyValidationInfiniteBatcher(), steps=asap.getValidationBatchsCount())
y2 = iteratorToArray(asap.getLabelOnlyValidationInfiniteBatcher(), steps=asap.getValidationBatchsCount())
for i in range(len(x1)):
    if i % 100 == 0:
        print("--------a")
        print(x1[i])
        print(x2[i])
        print("--------b")
        print(y1[i])
        print(y2[i])
    assert np.array_equal(x1[i], x2[i])
    assert np.array_equal(y1[i], y2[i])
    assert x1[i][1] == x2[i][1]
    assert y1[i][1] == y2[i][1]
assert np.array_equal(x1, x2)
assert np.array_equal(y1, y2)
assert not np.array_equal(x1[2], x2[4])
assert not np.array_equal(y1[2], y2[4])