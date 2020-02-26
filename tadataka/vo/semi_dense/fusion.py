def fusion(d1, d2, var1, var2):
    d = (d1 * var2 + d2 * var1) / (var1 + var2)
    var = (var1 * var2) / (var1 + var2)
    return d, var
