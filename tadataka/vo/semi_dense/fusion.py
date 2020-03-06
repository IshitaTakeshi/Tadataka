def fusion(d1, d2, var1, var2):
    V = (var1 + var2)
    d = (d1 * var2 + d2 * var1) / V
    var = (var1 * var2) / V
    return d, var
