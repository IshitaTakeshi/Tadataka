from tadataka.vo.semi_dense.hypothesis import HypothesisMap


def fusion_(mu1, mu2, var1, var2):
    V = (var1 + var2)
    mu = (mu1 * var2 + mu2 * var1) / V
    var = (var1 * var2) / V
    return mu, var


def fusion(hypothesis1: HypothesisMap, hypothesis2: HypothesisMap):
    inv_depth_map, variance_map = fusion_(
        hypothesis1.inv_depth_map, hypothesis2.inv_depth_map,
        hypothesis1.variance_map, hypothesis2.variance_map
    )
    return HypothesisMap(inv_depth_map, variance_map)
