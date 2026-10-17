
"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — GEOMETRIC OBSTACLE SHAPES
===============================================================================

PURPOSE
-------
This file defines the different geometric shapes that can be used as obstacles
in a three-dimensional (3D) environment for UAV (Unmanned Aerial Vehicle)
path planning.

The map generator can use these shapes to create an environment containing
solid objects, such as irregular polyhedra, cylinders, and ellipsoids, as well
as gates that the UAV may need to navigate through.

Each obstacle stores information such as:
    - Its centre position in 3D space.
    - Its dimensions or boundary limits.
    - Its orientation, where applicable.
    - Its velocity, for use in dynamic environments.
    - An optional point cloud associated with the obstacle.

The most important method in each obstacle class is is_inside().
It checks whether a given 3D point lies inside the obstacle's solid region.
A path-planning or collision-checking algorithm can use this result to decide
whether a UAV position or candidate path point collides with an obstacle.

OBSTACLE TYPES
--------------
1. Polyhedron : An irregular 3D solid made from flat polygonal faces.
2. Cylinder   : A vertical cylinder with a circular cross-section.
3. Ellipsoid  : A 3D oval-shaped solid that can be rotated.
4. CircleGate : A circular/elliptical frame with an opening in the middle.
5. RectGate   : A rectangular frame with an opening in the middle.

ROTATION
--------
The euler_to_rot() function converts three Euler angles (roll, pitch, yaw)
into a 3x3 rotation matrix. This matrix is used to orient objects in 3D space.

The implementation uses quaternions internally to combine the rotations.

LIBRARIES
---------
NumPy:
    Used for vectors, matrices, coordinate transformations, and calculations.

ConvexHull from SciPy:
    Used to construct the outer convex shape around randomly generated points
    when creating a Polyhedron.

IMPORTANT NOTE
--------------
This file defines the obstacle geometry and its point-containment checks.
It does not itself plan the UAV's path or move obstacles over time.
Other parts of the environment must use the geometry and update positions
when simulating movement.

===============================================================================
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def euler_to_rot(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles into a 3D rotation matrix.

    The three input angles represent rotations around the X, Y, and Z axes:
        r = roll  : rotation around X
        p = pitch : rotation around Y
        y = yaw   : rotation around Z

    All angles must be expressed in radians.

    Quaternions are used internally to combine the three rotations.
    The resulting rotation matrix can rotate a 3D vector or describe an
    object's orientation.

    Args:
        euler:
            NumPy array containing [roll, pitch, yaw].

    Returns:
        A 3x3 NumPy rotation matrix.

    Example:
        euler_to_rot(np.array([0.0, 0.0, 0.0]))

        Returns the identity matrix, meaning there is no rotation.
    """

    # Extract the rotation angles around the X, Y, and Z axes.
    r, p, y = euler

    # Construct the quaternion for rotation around the X-axis.
    # Quaternion components are stored in the order [w, x, y, z].
    # w is the scalar component; x, y, and z describe the vector component.
    qx = np.array([
        np.cos(r / 2),
        np.sin(r / 2),
        0.0,
        0.0
    ])

    # Construct the quaternion for rotation around the Y-axis.
    qy = np.array([
        np.cos(p / 2),
        0.0,
        np.sin(p / 2),
        0.0
    ])

    # Construct the quaternion for rotation around the Z-axis.
    qz = np.array([
        np.cos(y / 2),
        0.0,
        0.0,
        np.sin(y / 2)
    ])

    def qmul(a, b):
        """
        Multiply two quaternions.

        Quaternion multiplication combines rotations. It is not the same as
        multiplying corresponding array elements.

        Args:
            a: First quaternion [w, x, y, z].
            b: Second quaternion [w, x, y, z].

        Returns:
            The product quaternion [w, x, y, z].
        """

        # Separate the four components of each quaternion.
        aw, ax, ay, az = a
        bw, bx, by, bz = b

        # Calculate the quaternion product.
        # The result represents the combined rotations.
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    # Combine the rotations in the order qz * (qy * qx).
    # Under the usual column-vector convention, this corresponds to applying
    # the X rotation first, then Y, and finally Z.
    q = qmul(qz, qmul(qy, qx))

    # Extract the combined quaternion components.
    w, x, y_, z = q

    # Convert the quaternion into a 3x3 rotation matrix.
    # This formula assumes q is a unit quaternion.
    # The matrix can be used to rotate vectors or orient a 3D object.
    return np.array([
        [
            1 - 2 * (y_ * y_ + z * z),
            2 * (x * y_ - z * w),
            2 * (x * z + y_ * w)
        ],
        [
            2 * (x * y_ + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y_ * z - x * w)
        ],
        [
            2 * (x * z - y_ * w),
            2 * (y_ * z + x * w),
            1 - 2 * (x * x + y_ * y_)
        ]
    ])


class Polyhedron:
    """
    Represent an irregular 3D solid with flat faces.

    A polyhedron can be created from randomly generated vertices. SciPy's
    ConvexHull algorithm connects those vertices to form a convex solid.

    Each face is represented by:
        - A point lying on the face.
        - An outward-facing normal vector perpendicular to the face.

    The face equations are stored in the form:

        A @ point <= b0

    For a correctly oriented convex polyhedron, a point satisfies all these
    inequalities when it lies inside the solid, allowing a small tolerance.

    Attributes:
        h_poly: Face points and normals, stored as a (6, number_of_faces) array.
        A: Matrix containing one face normal per row.
        b0: Boundary constant for each face.
        bd: Stored boundary dimensions along X, Y, and Z.
        cpt: Centre position of the polyhedron.
        v: Velocity vector.
        cloud: Optional point-cloud data.
    """

    def __init__(self):
        """
        Initialise an empty polyhedron and its associated properties.

        Geometry is populated later by random_init().
        """

        # Each column will contain six values:
        # [point_x, point_y, point_z, normal_x, normal_y, normal_z].
        # None means that the face geometry has not yet been created.
        self.h_poly = None

        # A will store the face normals, with shape (number_of_faces, 3).
        self.A = None

        # b0 will store one boundary constant for each face.
        self.b0 = None

        # Stored half-extents/bounds along X, Y, and Z.
        # Initially, all three values are zero.
        self.bd = np.zeros(3)

        # Centre of the polyhedron: [x, y, z].
        self.cpt = np.zeros(3)

        # Velocity vector: [vx, vy, vz].
        self.v = np.zeros(3)

        # Empty list reserved for optional point-cloud data.
        self.cloud = []

    def _build_planes(self):
        """
        Build the mathematical plane equations for all polyhedron faces.

        Each face is represented by:

            normal . point = boundary_constant

        The resulting arrays allow the inside test to check:

            A @ point - b0 <= 0

        The method stores the face normals in A and the boundary constants
        in b0. It does not return a value.
        """

        # Number of faces is the number of columns in h_poly.
        faces = self.h_poly.shape[1]

        # Allocate space for one 3D normal vector per face.
        self.A = np.zeros((faces, 3))

        # Allocate one boundary constant per face.
        self.b0 = np.zeros(faces)

        # Process every face separately.
        for i in range(faces):

            # First three values of this column give a point on the face.
            p = self.h_poly[:3, i]

            # Last three values give the face's normal vector.
            n = self.h_poly[3:, i]

            # Store the normal as a row of the matrix A.
            self.A[i] = n

            # Calculate the plane's boundary constant using the dot product.
            # For points on this face, n dot p equals b0[i].
            self.b0[i] = n.dot(p)

    def is_inside(self, pt: np.ndarray, epsilon: float = 1e-6) -> bool:
        """
        Check whether a point lies inside or on the polyhedron's boundary.

        Args:
            pt:
                3D point [x, y, z] to test.

            epsilon:
                Small numerical tolerance for floating-point calculations.
                It allows points extremely close to a face to count as inside.

        Returns:
            True if the point satisfies all face inequalities within tolerance.
            False if it lies outside at least one face.

        Note:
            If the face equations have not been initialised, this method
            returns True. This is the current implementation's default
            behavior for an uninitialised polyhedron.
        """

        # If the face equations do not exist, treat the point as inside.
        if self.A is None or self.A.shape[0] == 0:
            return True

        # For each face, calculate:
        #     normal dot point - boundary_constant
        #
        # An interior point should give a value <= 0 for every face.
        # Taking the maximum checks the most violated face inequality.
        return float(np.max(self.A @ pt - self.b0)) <= epsilon

    def random_init(
        self,
        cpt: np.ndarray,
        bound: np.ndarray,
        rng: np.random.Generator
    ):
        """
        Generate a random convex polyhedron around a given centre.

        The method:
            1. Chooses a random number of vertices.
            2. Generates vertices within the supplied X, Y, and Z bounds.
            3. Uses ConvexHull to construct the outer convex shape.
            4. Calculates a normal vector for each triangular face.
            5. Attempts to orient the normals outward.
            6. Builds the plane equations used for collision checks.

        Args:
            cpt:
                Centre around which the vertices are generated, [x, y, z].

            bound:
                Maximum random displacement from the centre along each axis:
                [bound_x, bound_y, bound_z].

            rng:
                NumPy random-number generator used to generate reproducible
                random values when configured with a fixed seed.

        Returns:
            None. Updates the polyhedron's stored geometry and properties.
        """

        # Select a random integer number of vertices between 4 and 35.
        # ceil() converts the sampled floating-point number to an integer.
        n_verts = int(np.ceil(rng.uniform(4, 35)))

        # Generate random X, Y, and Z coordinates around the centre.
        # Each coordinate varies independently within its specified bounds.
        verts = np.column_stack([
            cpt[0] + rng.uniform(-bound[0], bound[0], n_verts),
            cpt[1] + rng.uniform(-bound[1], bound[1], n_verts),
            cpt[2] + rng.uniform(-bound[2], bound[2], n_verts),
        ])

        # Construct the convex hull surrounding the generated vertices.
        # The hull identifies the outer surface of the random point cloud.
        hull = ConvexHull(verts)

        # Each entry in hull.simplices identifies the vertices of one
        # triangular face of the hull.
        faces = hull.simplices

        # Allocate six values per face:
        # three coordinates for a point on the face and three for its normal.
        h_poly = np.zeros((6, len(faces)))

        # Calculate the geometry of each triangular face.
        for i, tri in enumerate(faces):

            # Retrieve the three vertices of this triangular face.
            p0, p1, p2 = verts[tri]

            # Create two edge vectors along the triangle.
            edge0 = p1 - p0
            edge1 = p2 - p1

            # The cross product of two edge vectors is perpendicular
            # to the face, so it provides a candidate normal vector.
            normal = np.cross(edge0, edge1)

            # Calculate the length of the normal vector.
            nrm = np.linalg.norm(normal)

            # Handle a degenerate or nearly degenerate face.
            if nrm < 1e-12:
                normal = np.array([0.0, 0.0, 1.0])
            else:
                # Normalise the vector so that its length is one.
                normal = normal / nrm

            # Attempt to orient the normal outward relative to the supplied
            # centre. The sign test determines which direction it points.
            if normal.dot(p1 - cpt) < 0:
                normal = -normal

            # Store a point on the face and its normal in the same column.
            h_poly[:3, i] = p1
            h_poly[3:, i] = normal

        # Save the generated face data on the object.
        self.h_poly = h_poly

        # Convert the face points and normals into plane equations.
        self._build_planes()

        # Save copies of the dimensions and centre supplied to this method.
        self.bd = bound.copy()
        self.cpt = cpt.copy()

        # Clear any previously stored point-cloud data.
        self.cloud = []

    def get_bd(self):
        """
        Return a copy of the polyhedron's stored boundary dimensions.

        Returns:
            NumPy array [bound_x, bound_y, bound_z].
        """

        # Return a copy so that changing the returned array does not directly
        # change the array stored inside this object.
        return self.bd.copy()

    def get_center(self):
        """
        Return a copy of the polyhedron's stored centre coordinates.

        Returns:
            NumPy array [x, y, z].
        """
        return self.cpt.copy()

    def set_center(self, cpt):
        """
        Update the stored centre coordinates.

        Args:
            cpt: New centre position [x, y, z].

        Note:
            This changes the stored centre only. It does not translate the
            existing face geometry or rebuild A and b0.
        """
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        """
        Return a copy of the stored velocity vector.

        Returns:
            NumPy array [vx, vy, vz].
        """
        return self.v.copy()

    def set_vel(self, vel):
        """
        Set the stored velocity vector.

        Args:
            vel: Velocity [vx, vy, vz], typically in distance units per second.

        Note:
            This stores the velocity. It does not move the polyhedron by itself.
        """
        self.v = np.asarray(vel, dtype=float)


class Cylinder:
    """
    Represent a vertical cylinder in 3D space.

    The cylinder is aligned with the global Z-axis. Its circular cross-section
    lies in the XY plane, and its height extends above and below its centre.

    Attributes:
        cpt: Centre position [x, y, z].
        r: Radius of the circular cross-section.
        h: Full height of the cylinder.
        bd: Stored bounds [r, r, h].
        v: Velocity vector [vx, vy, vz].
        cloud: Optional point-cloud data.
    """

    def __init__(self, cpt: np.ndarray, r: float, h: float):
        """
        Initialise a cylinder with a centre, radius, and height.

        Args:
            cpt: Centre coordinates [x, y, z].
            r: Radius of the cylinder.
            h: Full height of the cylinder.
        """

        # Store the centre as a floating-point NumPy array.
        self.cpt = np.asarray(cpt, dtype=float)

        # Store the radius and height as floating-point values.
        self.r = float(r)
        self.h = float(h)

        # Store the cylinder's radial and vertical dimensions.
        # Note that the third value is the full height, not the half-height.
        self.bd = np.array([r, r, h], dtype=float)

        # Initialise the velocity to zero.
        self.v = np.zeros(3)

        # Empty list reserved for optional point-cloud data.
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        """
        Check whether a point lies inside the cylinder.

        Args:
            pt: 3D point [x, y, z] to test.

        Returns:
            True if the point lies within both the cylinder's height
            and its circular cross-section; otherwise False.
        """

        # Check the vertical distance from the centre.
        # The cylinder extends half its height in either direction.
        if abs(pt[2] - self.cpt[2]) >= 0.5 * self.h:
            return False

        # Calculate the horizontal distance between the point and the
        # cylinder's central axis using its X and Y coordinates.
        # If this distance is no greater than the radius, the point lies
        # inside the circular cross-section.
        return float(np.linalg.norm(pt[:2] - self.cpt[:2])) <= self.r

    def get_bd(self):
        """Return a copy of the stored dimensions [r, r, h]."""
        return self.bd.copy()

    def get_center(self):
        """Return a copy of the cylinder's centre coordinates."""
        return self.cpt.copy()

    def set_center(self, cpt):
        """
        Update the cylinder's centre position.

        Args:
            cpt: New centre coordinates [x, y, z].
        """
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        """Return a copy of the cylinder's velocity vector."""
        return self.v.copy()

    def set_vel(self, vel):
        """
        Set the cylinder's velocity vector.

        Args:
            vel: Velocity [vx, vy, vz].
        """
        self.v = np.asarray(vel, dtype=float)


class Ellipsoid:
    """
    Represent a rotatable ellipsoid in 3D space.

    An ellipsoid is a 3D oval-shaped solid. Its size can differ along its
    principal axes, and its orientation can be changed using Euler angles.

    The implementation constructs a matrix E from the rotation matrix R and
    the supplied bounds:

        E = R @ diag(bound) @ R.T

    A point is inside when:

        norm(E_inverse @ (point - centre)) <= 1

    Under this formulation, the entries in bound act as the ellipsoid's
    semi-axis lengths.

    Attributes:
        E: Matrix describing the ellipsoid's orientation and dimensions.
        Einv: Inverse of E, used by the point-containment test.
        d: Centre coordinates.
        bd: Stored semi-axis lengths.
        v: Velocity vector.
        cloud: Optional point-cloud data.
    """

    def __init__(self):
        """Create an ellipsoid object with default values."""

        # Start with identity matrices.
        # These are replaced with the actual ellipsoid matrices by init().
        self.E = np.eye(3)
        self.Einv = np.eye(3)

        # Centre position of the ellipsoid.
        self.d = np.zeros(3)

        # Stored dimensions along the ellipsoid's principal axes.
        self.bd = np.zeros(3)

        # Initialise the velocity to zero.
        self.v = np.zeros(3)

        # Empty list reserved for optional point-cloud data.
        self.cloud = []

    def init(self, cpt: np.ndarray, bound: np.ndarray, euler: np.ndarray):
        """
        Configure the ellipsoid's position, size, and orientation.

        Args:
            cpt:
                Centre coordinates [x, y, z].

            bound:
                Three semi-axis lengths along the ellipsoid's principal axes.

            euler:
                Rotation angles [roll, pitch, yaw] in radians.

        Returns:
            None. Updates the object's matrices and properties.
        """

        # Convert the Euler angles into a 3D rotation matrix.
        R = euler_to_rot(np.asarray(euler, dtype=float))

        # Construct a diagonal matrix from the three semi-axis lengths.
        coeff = np.diag(bound)

        # Rotate the diagonal dimension matrix into the required orientation.
        # The resulting matrix describes the rotated ellipsoid.
        self.E = R @ coeff @ R.T

        # Calculate the inverse matrix, which is needed for the inside test.
        self.Einv = np.linalg.inv(self.E)

        # Store the centre and dimensions.
        self.d = np.asarray(cpt, dtype=float)
        self.bd = np.asarray(bound, dtype=float)

        # Clear any previously stored point-cloud data.
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        """
        Check whether a point lies inside or on the ellipsoid.

        Args:
            pt: 3D point [x, y, z] to test.

        Returns:
            True if the point satisfies the ellipsoid's containment equation;
            otherwise False.
        """

        # Subtract the centre to express the point relative to the ellipsoid.
        relative_pt = pt - self.d

        # Transform the relative point using the inverse ellipsoid matrix.
        # The resulting vector's length is <= 1 for points inside the ellipsoid.
        return float(np.linalg.norm(self.Einv @ relative_pt)) <= 1.0

    def get_bd(self):
        """Return a copy of the ellipsoid's stored semi-axis lengths."""
        return self.bd.copy()

    def get_center(self):
        """Return a copy of the ellipsoid's centre coordinates."""
        return self.d.copy()

    def set_center(self, cpt):
        """
        Update the ellipsoid's centre.

        Args:
            cpt: New centre coordinates [x, y, z].
        """
        self.d = np.asarray(cpt, dtype=float)

    def get_vel(self):
        """Return a copy of the ellipsoid's velocity vector."""
        return self.v.copy()

    def set_vel(self, vel):
        """Set the ellipsoid's velocity vector [vx, vy, vz]."""
        self.v = np.asarray(vel, dtype=float)


class CircleGate:
    """
    Represent a circular or elliptical gate frame in 3D space.

    The gate has an outer elliptical boundary and a smaller opening.
    The material between the outer boundary and the opening is treated as
    solid, so the UAV should avoid colliding with it.

    The gate is rotated around the global Z-axis by theta.

    Coordinate convention:
        Local X: Direction through the gate's thickness.
        Local Y: One direction across the gate.
        Local Z: The other direction across the gate.

    The rect array is used as:
        rect[0]: Half-thickness along local X.
        rect[1]: Outer semi-axis along local Y.
        rect[2]: Outer semi-axis along local Z.

    The inner opening is approximated by subtracting rect[0] from the two
    outer semi-axes.

    Attributes:
        cpt: Gate centre.
        rect: Gate dimensions.
        theta: Rotation angle in radians.
        R: Rotation matrix around Z.
        bd: Stored approximate boundary dimensions.
        v: Velocity vector.
        cloud: Optional point-cloud data.
    """

    def __init__(self, cpt: np.ndarray, rect: np.ndarray, theta: float):
        """
        Initialise a circular/elliptical gate.

        Args:
            cpt: Gate centre [x, y, z].
            rect: Dimensions [half_thickness, outer_radius_y, outer_radius_z].
            theta: Rotation around the Z-axis, in radians.
        """

        # Store the centre and dimensions as floating-point arrays.
        self.cpt = np.asarray(cpt, dtype=float)
        self.rect = np.asarray(rect, dtype=float)

        # Store the gate's rotation angle.
        self.theta = float(theta)

        # Calculate cosine and sine of the rotation angle.
        c, s = np.cos(theta), np.sin(theta)

        # Create a rotation matrix that rotates points around the Z-axis.
        self.R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Rotate the dimension vector to estimate the gate's spatial bounds.
        # bd stores approximate dimensions for use by the map generator.
        pt = self.R @ self.rect
        self.bd = np.array([
            self.rect[0] + abs(pt[0]),
            self.rect[0] + abs(pt[1]),
            self.rect[2]
        ])

        # Initialise the velocity to zero.
        self.v = np.zeros(3)

        # Empty list reserved for optional point-cloud data.
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        """
        Check whether a point lies inside the solid gate frame.

        Args:
            pt: 3D point [x, y, z] to test.

        Returns:
            True if the point lies inside the outer gate but outside its
            opening, meaning it intersects the solid frame.

            False if the point is outside the gate or inside the opening.
        """

        # Move the point into the gate's local coordinate system.
        # Subtracting cpt makes the gate centre the local origin.
        # R.T reverses the gate's rotation.
        diff = self.R.T @ (pt - self.cpt)

        # Reject points outside the gate's thickness.
        if abs(diff[0]) >= self.rect[0]:
            return False

        # Check the outer elliptical boundary in the local Y-Z plane.
        # For an ellipse, y^2/a^2 + z^2/b^2 < 1 means inside.
        # Points on or outside the outer boundary are not inside the frame.
        if (
            (diff[1] ** 2) / (self.rect[1] ** 2)
            + (diff[2] ** 2) / (self.rect[2] ** 2)
            >= 1.0
        ):
            return False

        # Estimate the semi-axes of the inner opening.
        # The opening is smaller than the outer ellipse.
        a = self.rect[1] - self.rect[0]
        b = self.rect[2] - self.rect[0]

        # If either inner semi-axis is non-positive, there is no valid
        # opening under this implementation, so points in the outer region
        # are treated as points in the solid frame.
        if a <= 0 or b <= 0:
            return True

        # Check whether the point lies inside the inner elliptical opening.
        # If it does, the point is passing through the hole, not hitting
        # the solid frame.
        if (
            (diff[1] ** 2) / (a * a)
            + (diff[2] ** 2) / (b * b)
            <= 1.0
        ):
            return False

        # The point is inside the outer ellipse but outside the opening.
        # Therefore, it lies within the solid gate frame.
        return True

    def get_bd(self):
        """Return a copy of the stored approximate boundary dimensions."""
        return self.bd.copy()

    def get_center(self):
        """Return a copy of the gate's centre coordinates."""
        return self.cpt.copy()

    def get_rect(self):
        """Return a copy of the gate's dimension array."""
        return self.rect.copy()

    def get_rot(self):
        """Return a copy of the gate's 3x3 rotation matrix."""
        return self.R.copy()

    def set_center(self, cpt):
        """
        Update the gate's centre position.

        Args:
            cpt: New centre coordinates [x, y, z].
        """
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        """Return a copy of the gate's velocity vector."""
        return self.v.copy()

    def set_vel(self, vel):
        """Set the gate's velocity vector [vx, vy, vz]."""
        self.v = np.asarray(vel, dtype=float)


class RectGate:
    """
    Represent a rectangular gate frame in 3D space.

    The gate consists of a rectangular outer boundary and a smaller rectangular
    opening. The solid frame around the opening is treated as an obstacle.

    The gate is rotated around the global Z-axis by theta.

    Coordinate convention:
        Local X: Direction through the gate's thickness.
        Local Y: Horizontal direction across the gate.
        Local Z: Vertical direction across the gate.

    The rect array is used as:
        rect[0]: Half-thickness along local X.
        rect[1]: Outer half-width along local Y.
        rect[2]: Outer half-height along local Z.

    The opening is defined by reducing the outer width and height by the
    thickness value rect[0].

    Attributes:
        cpt: Gate centre.
        rect: Gate dimensions.
        theta: Rotation angle in radians.
        R: Rotation matrix around Z.
        bd: Stored approximate boundary dimensions.
        v: Velocity vector.
        cloud: Optional point-cloud data.
    """

    def __init__(self, cpt: np.ndarray, rect: np.ndarray, theta: float):
        """
        Initialise a rectangular gate.

        Args:
            cpt: Gate centre [x, y, z].
            rect: Dimensions [half_thickness, outer_half_width, outer_half_height].
            theta: Rotation around the Z-axis, in radians.
        """

        # Store the centre and dimensions.
        self.cpt = np.asarray(cpt, dtype=float)
        self.rect = np.asarray(rect, dtype=float)

        # Store the gate's rotation angle.
        self.theta = float(theta)

        # Calculate cosine and sine of the rotation angle.
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrix for rotating around the global Z-axis.
        self.R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Store approximate boundary dimensions.
        # These are the values currently used by this implementation.
        self.bd = np.array([
            self.rect[1],
            self.rect[1],
            self.rect[2]
        ])

        # Initialise the velocity to zero.
        self.v = np.zeros(3)

        # Empty list reserved for optional point-cloud data.
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        """
        Check whether a point lies inside the solid rectangular gate frame.

        Args:
            pt: 3D point [x, y, z] to test.

        Returns:
            True if the point is inside the outer rectangle but outside
            the central opening, meaning it lies in the solid frame.

            False if the point is outside the gate or inside its opening.
        """

        # Convert the point from world coordinates to local gate coordinates.
        # Subtract the centre, then undo the gate's rotation using R.T.
        diff = self.R.T @ (pt - self.cpt)

        # Reject points outside the outer rectangular dimensions.
        # The first check is for thickness, the second for width,
        # and the third for height.
        if (
            abs(diff[2]) >= self.rect[2]
            or abs(diff[1]) >= self.rect[1]
            or abs(diff[0]) >= self.rect[0]
        ):
            return False

        # Check whether the point lies inside the smaller rectangular opening.
        # The opening's half-height and half-width are reduced by the
        # gate's thickness value rect[0].
        if (
            abs(diff[2]) <= self.rect[2] - self.rect[0]
            and abs(diff[1]) <= self.rect[1] - self.rect[0]
        ):
            # The point is passing through the opening, so it is not
            # considered to collide with the solid frame.
            return False

        # The point is inside the outer rectangle and outside the opening.
        # It therefore lies within the solid gate frame.
        return True

    def get_bd(self):
        """Return a copy of the stored approximate boundary dimensions."""
        return self.bd.copy()

    def get_center(self):
        """Return a copy of the gate's centre coordinates."""
        return self.cpt.copy()

    def get_rect(self):
        """Return a copy of the gate's dimension array."""
        return self.rect.copy()

    def get_rot(self):
        """Return a copy of the gate's 3x3 rotation matrix."""
        return self.R.copy()

    def set_center(self, cpt):
        """
        Update the gate's centre position.

        Args:
            cpt: New centre coordinates [x, y, z].
        """
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        """Return a copy of the gate's velocity vector."""
        return self.v.copy()

    def set_vel(self, vel):
        """Set the gate's velocity vector [vx, vy, vz]."""
        self.v = np.asarray(vel, dtype=float)
