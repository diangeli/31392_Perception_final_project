import numpy as np

def initialize_kalman():
    dt = 0.1

    # The initial state (6x1).
    x = np.array([[0], # Position along the x-axis
                  [0], # Velocity along the x-axis
                  [0], # acceleration along the x-axis
                  [0], # Velocity along the y-axis
                  [0], # Position along the y-axis
                  [0], # acceleration along the y-axis
                  [0], # Position along the z-axis
                  [0], # Velocity along the z-axis
                  [0]]) # acceleration along the z-axis])

    # The initial uncertainty (6x6).
    P = np.identity(9)*100

    # The external motion (6x1).
    u = np.array([[0], 
                  [0], 
                  [0], 
                  [0], 
                  [0], 
                  [0], 
                  [0], 
                  [0],
                  [0]])

    # The transition matrix (6x6). 
    F = np.array([[1,dt,(dt**2)/2,0,0,0,0,0,0],
                  [0,1,dt,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0],
                  [0,0,0,1,dt,(dt**2)/2,0,0,0],
                  [0,0,0,0,1,dt,0,0,0],
                  [0,0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,1,dt,(dt**2)/2],
                  [0,0,0,0,0,0,0,1,dt],
                  [0,0,0,0,0,0,0,0,1]])

    # The observation matrix (2x6).
    H = np.array([[1,0,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,1,0,0]])

    # The measurement uncertainty.
    R = [[1],
        [1],
        [1]]
    
    return x, P, u, F, H, R

def update(x, P, Z, H, R):
    y = Z - np.dot(H,x)
    S = np.linalg.multi_dot([H,P,H.transpose()]) + R
    K = np.linalg.multi_dot([P,H.transpose(),np.linalg.pinv(S)])
    x_new = x + np.dot(K,y)
    p_new = np.dot(np.identity(len(P)), P)
    return np.asarray(x_new), np.asarray(p_new)
    
def predict(x, P, F, u):
    x_new = np.dot(F,x) + u
    p_new = np.linalg.multi_dot([F,P,F.transpose()])
    return np.asarray(x_new), np.asarray(p_new)



