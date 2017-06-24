from scipy.stats import multivariate_normal

# limited number of frames without associated detection
survive = 5

# Initial sample positions are drawn from a Normal distribution centered at 
# the detection center (zone along the image borders for sequences).
# Initial size corresponds to the detection size, 
# and the motion direction is set to be orthogonal to the closest image border.


# Termination : if the tracker is not associated to a detection 
# after the number of frames in the input, it is automatically terminated.


class Particle(object):
    """ Represents a particle
            x: the x-coordinate
            y: the y-coordinate
            w: the particle weight
            l: the particle life
    """

    def __init__(self,x=0.0,y=0.0,w=1.0,l=0):
        self.x = x
        self.y = y
        self.w = w
        self.l = l


# spread the n particles in the shape (defined by a center) from a Normal distribution centered at the detection center
def repartition(x_center,y_center,number_particules):
	mean = [x_center, y_center]
	cov = [[100, 0], [0, 100]]
	x, y = np.random.multivariate_normal(mean, cov, number_particules).T
	plt.plot(x, y, 'x')
	plt.axis('equal')
	plt.show()