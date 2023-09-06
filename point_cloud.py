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
            cloud = self.radius*(np.random.rand(self.Npoints,3) - 0.5)
            self.cloud = np.array([np.array(i) + np.array(self.center) for i in cloud])
            return self.cloud
    
    def Sphere(self):

        def inner_points(R, center, uni=1):
            c = True
            while c==True:
                h, k, l = center
                x = np.random.uniform(h-R, h+R)
                y = np.random.uniform(k-R, k+R)
                z = np.random.uniform(l-R, l+R)
                
                if (x-h)**2 + (y-k)**2 + (z-l)**2 <= R**2:
                    c = False
                    return np.array([x, y, z])

        cloud = []
        for n in range(self.Npoints):
            cloud.append(inner_points(self.radius, self.center))
    
        self.cloud = np.array(cloud)
        return self.cloud 
           
    def Sphere_surface(self):
        def sample_spherical(npoints, ndim=3):
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0)
            return vec
        
        xi, yi, zi = sample_spherical(self.Npoints)
        r = self.radius
        cloud = np.array(list((zip(r*xi, r*yi, r*zi))))
        self.cloud = np.array([np.array(i) + np.array(self.center) for i in cloud])
        return self.cloud

    def CustomCloud(self, my_cloud):
        self.cloud = my_cloud
        return self.cloud
        
    # def Voronoi_cell():
    #     pass
