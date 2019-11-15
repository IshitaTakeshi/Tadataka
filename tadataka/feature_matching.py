from autograd import numpy as np


def reduce_redundancy(rows, cols, data):
    def reduce_along(rows, column):
        # get unique elements in 'column'
        for u in np.unique(data[cols==column]):
            mask = np.logical_and(cols == column, data == u)
            rows_ = rows[mask]
            # overwrite the row indices
            # (equivalent to move elements to the same row as rows_[0])
            for r in rows_[1:]:
                rows[rows==r] = rows_[0]
        return rows

    n_cols = np.max(cols) + 1

    for j in range(0, n_cols):
        rows = reduce_along(rows, j)
    return rows, cols, data


def to_matrix(rows, cols, data):
    n_rows = np.max(rows) + 1
    n_cols = np.max(cols) + 1

    matrix = np.full((n_rows, n_cols), np.nan)
    matrix[rows, cols] = data

    # remove rows where all elements are nan
    mask = ~np.all(np.isnan(matrix), axis=1)
    return matrix[mask]


class MatchMatrixGenerator(object):
    # the 'add' method of this class takes three arguments:
    # viewpoint1, viewpoint2, match matrix.
    # Match marix is a matrix of 2 columns, where each column contains
    # the indices of matching correspondence.
    # For example, the arguments below means 0th and 1st keypoint of the 0th view
    # is matching to the 0th and 2nd keypoint in the 1st view
    #
    # viewpoint1, viewpoint2 = 0, 1
    # matches = [[0, 0],
    #            [1, 2]]

    # Say 'add' is called 4 times with the argumens below

    # 0, 1
    # [[0, 0],
    #  [1, 2]]

    # 0, 2
    # [[1, 0],
    #  [3, 1]]

    # 1, 2
    # [[1, 2]]

    # 1, 3
    # [[2, 1],
    #  [1, 2]]

    # We formulate a redundant matrix from the given arguments
    #
    # redundant representation:
    #       0    1    2    3     # viewpoint index
    #   0  [0    0    nan  nan]
    #   1  [1    2    nan  nan]
    #   2  [1    nan  0    nan]
    #   3  [3    nan  1    nan]
    #   4  [nan  1    2    nan]
    #   5  [nan  2    nan  1  ]
    #   6  [nan  1    nan  2  ]

    # The 1st row in the redundant matrix means that
    # the 1st point in the 0th view and
    # the 2nd point in the 1st view are the projections of the same 3D point.
    # In the same way, the 2nd rows means that
    # the 1st point in the 0th view and
    # the 0th point in the 2nd view are the projections of the same 3D point.
    # Furthermore, the 5th row indicates that
    # the 2nd point in the 1st view and
    # the 1st point in the 3rd view are the projections of the same 3D point.
    # What we can find from above is that
    #
    # * 1st point in the 0th view
    # * 2nd point in the 1st view
    # * 0th point in the 2nd view
    # * 1st point in the 3rd view
    #
    # are the projections of the same 3D point.
    # Therefore, we can reduce the redundant matrix into the compact form
    # like below

    # compact representation:
    #       0    1    2    3     # viewpoint index
    #   0  [0    0    nan  nan]
    #   1  [1    2    0    1  ]
    #   2  [3    nan  1    nan]
    #   3  [nan  1    2    2  ]

    # As you can see, the 1st, 2nd, and 5th row in the redundant form
    # are compressed into the 1st row in the compact representation.
    # The 4th row and 6th row in the redundant form are combined into
    # the 3rd row of the compact form in the same manner.
    # The compact matrix is transposed before returning.

    def __init__(self):
        self.rows = []
        self.cols = []
        self.data = []

        self.n_rows = 0

    def add(self, viewpoint1, viewpoint2, matches):
        n_matches = matches.shape[0]

        rows = np.repeat(np.arange(self.n_rows, self.n_rows + n_matches), 2)
        cols = np.tile([viewpoint1, viewpoint2], n_matches)
        data = matches.flatten()

        self.rows += rows.tolist()
        self.cols += cols.tolist()
        self.data += data.tolist()

        self.n_rows += n_matches

    def matrix(self):
        rows = np.array(self.rows)
        cols = np.array(self.cols)
        data = np.array(self.data)
        rows, cols, data = reduce_redundancy(rows, cols, data)
        return to_matrix(rows, cols, data).T
