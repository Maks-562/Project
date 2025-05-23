// Define rectangle vertices
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};

// Define rectangle lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define rectangle surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Define mesh size (average size of elements)
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.1;



Physical Surface("domain") = {20};  // Main domain
Physical Curve("boundary") = {1};  // Outer boundary
Physical Curve("holes") = {10, 11, 12};  // Circular holes

// Generate 2D mesh
Mesh 2;

// Output mesh format
Save "mesh.msh";