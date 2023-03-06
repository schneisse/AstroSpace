import numpy as np

class PointCloud:
    def __init__(self, R=2, Npoints=1000, center = [0, 0, 0]):
        # self.cloud = set() if cloud is None else cloud
        self.cloud = None
        self.radius = R
        self.Npoints = Npoints
        self.center = center
    
    def Cube(self):
        if self.cloud is None:
            self.cloud = self.radius*(np.random.rand(self.Npoints,3) - 0.5)
            return self.cloud
    
    def Sphere(self, non_uniform = False, uni=2):

        def inner_points(R, center, uni=1):
            if uni == 1:

                while True:
                    x = np.random.random()*2 - 1
                    y = np.random.random()*2 - 1
                    z = np.random.random()*2 - 1
                    h, k, l = center
                    
                    if (x-h)**2 + (y-k)**2 + (z-l)**2 <= R**2:
                        return np.array([x, y, z])
            else:
                while True:
                    x = np.random.exponential()*uni - 1
                    y = np.random.exponential()*uni - 1
                    z = np.random.exponential()*uni - 1
                    h, k, l = center
                    
                    if (x-h)**2 + (y-k)**2 + (z-l)**2 <= R**2:
                        return np.array([x, y, z])

        if self.cloud is None and non_uniform==False:
            cloud = []
            for n in range(self.Npoints):
                cloud.append(inner_points(self.radius, self.center))
            self.cloud = np.array(cloud)
            return self.cloud 
           
        elif self.cloud is None and non_uniform==True:
            cloud = []
            for n in range(self.Npoints):
                cloud.append(inner_points(self.radius, self.center, uni=uni))
            self.cloud = np.array(cloud)
            return self.cloud
    
    def Sphere_surface(self):
        def sample_spherical(npoints, ndim=3):
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0)
            return vec
        
        xi, yi, zi = sample_spherical(self.Npoints)
        r = self.radius
        self.cloud = np.array(list((zip(r*xi, r*yi, r*zi))))
        return self.cloud

    def CustomCloud(self, my_cloud):
        self.cloud = my_cloud
        return self.cloud
        
    # def Voronoi_cell():
    #     pass
