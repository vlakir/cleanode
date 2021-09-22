from typing import List

# A draft for the future realization of a adoptation according:
# [Everhart2] Everhart, E. 1985, An Efficient Integrator that uses Gauss-Radau Spacings,
#             The Dynamics of Comets: Their Origin and Evolution


def binomial_coeffs(order: int) -> List[int]:
    """
    Return binomial coefficients of the defined order
    :param order: order
    :type order: int
    :return: binomial coefficients
    :rtype: List[int]
    """
    def binc(bcs, n, k):
        if k > n:
            return 0
        if k > n // 2:
            k = n - k
        if k == 0:
            return 1
        if k == 1:
            return n
        while len(bcs) < n - 3:
            for i in range(len(bcs), n - 3):
                r = []
                for j in range(2, i // 2 + 3):
                    r.append(binc(bcs, i + 3, j - 1) + binc(bcs, i + 3, j))
                bcs.append(r)
        r = bcs[n - 4]
        if len(r) < k - 1:
            for i in range(len(r), k - 1):
                r.append(binc(bcs, n - 1, k - 1) + binc(bcs, n - 1, k))
        return bcs[n - 4][k - 2]

    result = []
    temp = []
    for m in range(order + 1):
        result.append(binc(temp, order, m))

    return result


if __name__ == '__main__':
    a_old = [1, 1, 1, 1]
    a_new = [1, 1, 1, 1]
    dt_old = 5
    dt_new = 5
    q = dt_new / dt_old

    b_c = []
    for i in range(1, len(a_old) + 1):
        row = []
        for j in range(1, len(a_old) + 1):
            if j >= i:
                row.append(binomial_coeffs(j)[i])
            else:
                row.append(0)

        b_c.append(row)

    for i in range(len(a_old)):
        a_new[i] = 0
        for j in range(len(a_old)):
            a_new[i] += b_c[i][j] * a_old[j]
        a_new[i] *= q ** (i + 1)

    print(a_old)
    print(a_new)
