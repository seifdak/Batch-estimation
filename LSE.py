import numpy as np 
import scipy
import scipy.io 
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, spdiags, csr_matrix, linalg, coo_matrix


dataset = "/Users/zif/Desktop/RES/dataset1.mat"
FREQ = 10.
TS = 1. / FREQ 
GROUND_TRUTH_DATA = "ground_truth.csv"
#Select value of delta = 1, 10, 100, 1000 (1 corresponds to the whole dataset)
delta = 1

def function(delta):

    """ 

    Load data from dataset and into arrays

    """
    data = scipy.io.loadmat((dataset))
    x_true = data["x_true"]      # A 12709×1 array containing the true position, xk, of the robot (one-dimensional, along the rail) [m]
    t = data["t"]                # A 12709×1 array containing the data timestamps [s]
    v = data["v"]                # A 12709×1 array containing the speed
    r = data["r"]                # The range, rk, between the robot and the cylinder’s center as measured by the laser
    l = data["l"]                # The scalar true position, xc, of the cylinder’s center 
    r_var = data["r_var"][0][0]  # Range readings variance
    v_var = data["v_var"][0][0]  # Speed readings variance


    """ 

    preprocess and parametrize data into sets with param delta to only solve for a set
    Use all Odermetry data
    Use laser rangefinder at timestamps

    """
    u = v * TS
    y = l - r
    pos_var = v_var*TS**2

    # setting

    step = delta

    # select timestamp set

    time_sets = [at for at in y[::step]] 
    time_sets = csr_matrix(time_sets)
    t = time_sets

    # Cumulative Sum

    id = int(12709 / delta)
    sum = np.zeros(id)
    for set in range(id):
        inn = delta * set
        out = delta * (set + 1)
        cumsum = np.cumsum(u[inn:out])
        sum[set] = float(cumsum[-1])

    rest = 12709 % delta
    if rest != 0:
        cumsum_norest = u[(12709-rest):12709]
        estimate = np.concatenate((sum,cumsum_norest[-1]))
        estimate = np.array(estimate)
        u = csr_matrix(estimate).T  
    else:
        u = csr_matrix(sum).T

    # select measurments set

    laser_sets = [yb for yb in y[::step]]
    laser_sets = csr_matrix(laser_sets)
    y = laser_sets

    position_sets = [a for a in x_true[::step]]
    np.savetxt("ground_truth.csv", position_sets, delimiter=",")
    position_sets = csr_matrix(position_sets)
    x_true = position_sets.todense()

    #preapre batch matrices without initial state knowledge 

    A = scipy.sparse.diags([[1] * len(u.toarray())], [0],  format="csc").todense() 

    A_inv = A
    for i in range(len(A)-1):
        A_inv[i + 1, i] = - A[i , i]
    A_inv = csr_matrix(A_inv)
    C = scipy.sparse.diags([[1] * len(u.toarray())], [0], format="csc")
    H = np.vstack((A_inv.toarray(), C.toarray()))
    H = np.delete(H, 0, 0)
    H = csr_matrix(H)
    z = np.vstack((u.toarray(), y.toarray()))
    z = np.delete(z, 0, 0)
    z = csr_matrix(z)
    Q = scipy.sparse.diags( [[1. / (pos_var * delta)] * len(u.toarray())], [0], format="csc")
    R= scipy.sparse.diags( [[1. / r_var] * len(u.toarray())], [0] , format="csc")
    W_inv = scipy.sparse.block_diag((Q, R)).todense()
    W_inv = np.delete(W_inv, 0, 0)
    W_inv = np.delete(W_inv, 0, 1)
    W_inv = csr_matrix(W_inv)


    right = H.T @ W_inv @ z
    left = H.T @ W_inv @ H
    x = scipy.sparse.linalg.spsolve(left, right)

    # Uncertainty Envelope

    diag = np.diag(np.linalg.inv(left.todense()))
    upper_bound = 3 * np.sqrt(diag)
    lower_bound = -3 * np.sqrt(diag) 

    """
    ----------------Plotting------------------------------ 
    """  
    soll_x = np.genfromtxt(GROUND_TRUTH_DATA, delimiter=",")


    plt.figure()
    plt.title("Robot's position")
    plt.ylabel("X Position in meters")
    plt.plot(soll_x, label="Position Ground Truth")
    plt.plot(x, label="Estimated Position")
    plt.legend()

    plt.figure()
    plt.title("Position Residuals.")
    plt.ylabel("X Position error")
    plt.plot(soll_x - x, label="Error")
    plt.plot(upper_bound, label="Upper bound")
    plt.plot(lower_bound, label="Lower bound")
    plt.legend()

    plt.figure()
    plt.title("Histogram")
    plt.hist(soll_x - x, bins='auto')
    plt.legend()

    plt.show()
if __name__ == '__main__':
   print("Least square Batch estimation with Delta =", delta, "subsets")
   results = function(delta)