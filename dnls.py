import numpy as np
from scipy.integrate import complex_ode


def central_amplitude(time, L, M=101, aperiodicity=0):
    """Return the amplitude at the central site of the M-site lattice at the
    given time.  The initial condition is amplitude 1 at the central site,
    zero at all other sites.

    """
    integrator = complex_ode(dnls_rhs(M, L, aperiodicity))

    central_site_index = (M - 1)/2
    ic = np.zeros(shape=(M,))
    ic[central_site_index] = 1
    integrator.set_initial_value(ic)

    y = integrator.integrate(time)
    return np.abs(y[central_site_index])


def dnls_rhs(M, L, aperiodicity=0):
    """Return a function that evaluates the right-hand-side of the DNLS.
    (This can then be passed to an ODE solver.)

    Parameters
    ----------
    M : int
        Number of sites.
    L : float
        Nonlinearity parameter.

    Returns
    -------
    f : function
        A function that evaluates the RHS of the DNLS.

    """
    if aperiodicity == 0:
        hoppings = (  np.diag(np.ones(shape=(M - 1,)), -1)
                    + np.diag(np.ones(shape=(M - 1,)),  1) )
    else:
        fh = fibonacci_hoppings(M - 1, aperiodicity)
        hoppings = np.diag(fh, -1) + np.diag(fh, 1)

    def f(t, y):
        return -1j*L*np.abs(y)**2*y + 0.5j*np.dot(hoppings, y)

    return f


def fibonacci_hoppings(length, p):
    """Return an array of fibonacci hoppings for aperiodicity parameter `p`.

    """
    a, b = ab_values(length, p)
    return np.array(fibonacci(length, a, b))


def ab_values(length, p):
    """Return the lengths of the hoppings for the fibonacci chain, chosen so
    that the mean hopping approaches 1 as length increases, with `b`
    lengthening and `a` shortening monotonically.

    """
    # TODO: Is it possible to choose a and b values to set the mean exactly to
    #       1 for every choice of length and p, while preserving the
    #       monotonicity of a and b with p?
    golden_ratio = (1 + np.sqrt(5))/2
    a = 1/(1 + (2 - golden_ratio)*p)
    b = a*(1 + p)
    return (a, b)


def fibonacci(length, a, b):
    """Return the beginning of the fibonacci chain with elements a and b.

    Parameters
    ----------
    length : int
        The length of the chain requested.
    a : object
        The a element of the chain.
    b : object
        The b element of the chain.

    Returns
    -------
    chain : tuple
        A tuple of the requested length, with objects a and b alternating in
        the aperiodic fibonacci pattern.

    """
    if length == 1:
        return (a,)
    if length == 2:
        return (a, b)

    first = (a,)
    second = (a, b)
    while True:
        next = second + first
        if len(next) >= length:
            return next[0:length]
        first = second
        second = next
