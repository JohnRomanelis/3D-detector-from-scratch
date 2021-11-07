import open3d as o3d 
import numpy as np 

from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence


class Drawable(metaclass=ABCMeta):

    """
        Parent Class for every drawable object,
        like pointcloud or bbox.
        

        Every subclass should implement the _drawable method.
        This method should return an open3d object that can 
        be drawn using the draw_geometries method. 
        _drawable method should also be marked as @property

        Use of this class:
        This class is used alongside the Scene class (see bollow)
        to keep the open3d as the backend and dont directly call 
        open3d methods in other classes. (Simpler Code!)

    """

    @property
    def drawable(self):
        return self._drawable

    @abstractmethod
    def _drawable(self):
        pass

    def __str__(self):
        return "Drawable"

    def __repr__(self):
        return "Drawable"

class BoundingBox(Drawable):

    def __init__(self, color=None):
        # Creating an open3d lineset centered at the origin
        self._box = o3d.geometry.TriangleMesh.create_box().translate((-0.5,-0.5,-0.5))
        
        # TIP: could use different color based on difficulty or accuracy score
        if color is not None:
            self._color = color
        else:
            self._color = (0.,0.,0.)


    def scale(self, scaling_vec):
        # if the scaling is the same across all axis
        if isinstance(scaling_vec, float):
            # use the open3d scale
            self._box.scale(scaling_vec, center=(0.,0.,0.))
        
        else:
            if isinstance(scaling_vec, (list, tuple)):
                scaling_vec = np.array(scaling_vec)
            
            scaling_matrix = np.diag(scaling_vec)
            
            
            mesh_vertices = np.asarray(self._box.vertices)
            mesh_vertices = np.matmul(mesh_vertices, scaling_matrix)

            self._box.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    def rotate(self, theta, axis='z'):
        # rotate around axis by angle theta
        rotation_vec = np.array([0., 0., 0.])


        if axis == 'x':
            rotation_vec[0] = -theta
        elif axis == 'y':
            rotation_vec[1] = -theta
        else:
            rotation_vec[2] = -theta

       
        R = self._box.get_rotation_matrix_from_xyz(rotation_vec)
        self._box.rotate(R, center=(0, 0, 0))

    def translate(self, translation_vec):
        # translate mesh to location
        self._box.translate(translation_vec)

    def __str__(self):
        return "Bounding Box"
    
    def __repr__(self):
        return "Bounding Box"

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def _drawable(self):
        lineset = o3d.geometry.LineSet()
        lineset = lineset.create_from_triangle_mesh(self._box)
        lineset.paint_uniform_color(self._color)
        return lineset

class KittiBoundingBox(BoundingBox):
    """
        Creating a Bounding Box based on the Kitti arguments
        
    Args: 
         - center : (x, y, z) the center of the bounding box
         - dims: (l, h, w) the dimensions of the bounding box
         - rotation_y: rotation around the up_vector

         TODO: take as input the value of the direction bin


    """

    def __init__(self, center, dims, rotation_y, color=None):
        super().__init__(color=color)

        self.scale(dims)
        self.rotate(rotation_y)
        self.translate(center)

    def __str__(self):
        return "KittiBoundingBox"

    def __repr__(self):
        return "KittiBoundingBox"

class PointCloud(Drawable):

    """
    Args:
        - points: numpy.array with shape [N, 3]


    """
    def __init__(self, points, color = None):

        assert points.shape[1] == 3

        self._pc = o3d.geometry.PointCloud()
        self._pc.points = o3d.utility.Vector3dVector(points)

        if color is not None:
            self._pc.paint_uniform_color(color)


    #@color.setter
    def color(self, c):
        self._pc.paint_uniform_color(c)

    @property
    def _drawable(self):
        return self._pc

class Scene(MutableSequence):
    """ 
        Basically a list Containing Drawable objects.
        
        Basic methods:
            - add item: add a drawable item
            - draw: display all the drawable objects
    """

    def __init__(self, *args):
        super(Scene, self).__init__()

        # types of objects that this list may contain
        self.oktypes = Drawable
        # creating a list to store the data
        self._list = list()
        self.extend(list(args))


    def _check(self, v):
        if not isinstance(v, self.oktypes):
            raise TypeError

    def __len__(self): return len(self._list)

    def __getitem__(self, i): return self._list[i]

    def __delitem__(self, i): del self._list[i]

    def __setitem__(self, i, v):
        self._check(v)
        self._list[i] = v

    def insert(self, i, v):
        self._check(v)
        self._list.insert(i, v)

    def __str__(self):
        return str(self._list)

    def draw(self):
        render_list = [geometry.drawable for geometry in self._list]
        o3d.visualization.draw_geometries(render_list)

class KittiScene(Scene):
    """
        Instead of the simple append, insert etc operations
        of the simple Scene class, in this class we can 
        add more complex structures such as a bounding box 
        or a pointcloud represented as numpy arrays
    """

    def add_bbox(self, center, dims, rotation_y, color=None):
        new_bbox = KittiBoundingBox(center, dims, rotation_y, color)
        self._list.append(new_bbox)


    def add_lidar_pointcloud(self, pointcloud, color=None):
        pc = PointCloud(pointcloud, color)
        self._list.append(pc)

    def add_multiple_boxes(self, boxes_np, color=None, code_size=7):
        assert code_size == 7
        boxes_np = boxes_np.reshape(-1, code_size)

        if color is not None:
            color = color.reshape(-1, 3)
            if len(color) == 1:
                # using the same color for all points
                color = color.repeat(len(boxes_np), axis=0)
            print(len(color), "  ", len(boxes_np))

            assert len(boxes_np) == len(color)

            bboxes = [KittiBoundingBox(bbox[:3], bbox[3:6], bbox[6], c) 
                                                for bbox, c in zip(list(boxes_np), color)]
        else:
            bboxes = [KittiBoundingBox(bbox[:3], bbox[3:6], bbox[6]) 
                                                for bbox in list(boxes_np)]
        self._list.extend(bboxes)

##-------------------------------------------------------------##
## Following functions have been merged into BoundingBox class ##
##-------------------------------------------------------------##

def scale(mesh, scaling_vec):
    """
    Args:
         - mesh: the mesh to be scaled, 
                **the mesh should be centered at the origin

         - scaling_vec: the scaling to apply per axis
        
    
    Returns:
         - the scaled mesh
    """

    # if the scaling is the same across all axis
    if isinstance(scaling_vec, float):
        # use the open3d scale
        mesh.scale(scaling_vec, center=(0.,0.,0.))
    
    else:
        if isinstance(scaling_vec, (list, tuple)):
            scaling_vec = np.array(scaling_vec)
        
        scaling_matrix = np.diag(scaling_vec)
        
        
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_vertices = np.matmul(mesh_vertices, scaling_matrix)

        print(mesh_vertices)
        mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        mesh.compute_vertex_normals()

def mesh_to_lineset(mesh):

    lineset = o3d.geometry.LineSet()
    lineset = lineset.create_from_triangle_mesh(mesh)
    lineset.paint_uniform_color((0.0,0.0,0.0))
    return lineset