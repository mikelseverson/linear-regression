from numpy import *


# We want a measurement to minimize,
# Every time we draw an error we have a "error" value we are trying to minimize
# This is called "Gradiant Decent"
def compute_error_for_line_given_points(b, m, points):
    #initialize 0
    totalError = 0
    #each data point
    for i in range(0, len(points)):
        x = points [i, 0]
        y = points [i, 1]
        #get difference, square it, add to totalError
        totalError += (y - (m * x + b)) **2

    #get the average
    return totalError / float(len(points))

def gradiant_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    # gradiant descent
    for i in range(num_iterations):
            # update b and m with a new more accurate b and m
            # this is done by preforming a "gradient step"
            b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

# This is where the greatest of the magic will happen
def step_gradient(b_current, m_current, points, learningRate):

    # Starting point for gradient
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction in respect to b and m
        # computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))


    # Update our b and m values using our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def run():
    points = genfromtxt('data.csv', delimiter=',')

    # Hyperparameters
    learning_rate = 0.0001
    num_iterations = 1000

    # y = mx + b
    initial_m = 0
    initial_b = 0

    # Train our model
    print 'starting gradiant descent at b = {0}, m = {1}, error {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b, m] = gradiant_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print 'ending point at b = {1}, m = {2}, error {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run();
