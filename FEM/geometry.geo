SetFactory("OpenCASCADE");

// Define main domain (square)
L = 1.0;  // Side length of the square
Rectangle(1) = {0, 0, L, L}; // Add extra '0' to avoid parameter errors

// Define circular inclusions
r1 = 0.15; r2 = 0.25; r3 = 0.1;
c1 = newv; Circle(c1) = {0.3, 0.7, 0, r1};
c2 = newv; Circle(c2) = {0.7, 0.7, 0, r2};
c3 = newv; Circle(c3) = {0.5, 0.4, 0, r3};

// Define a curve loop for holes
Curve Loop(10) = {c1};
Curve Loop(11) = {c2};
Curve Loop(12) = {c3};

// Define surface and holes
Plane Surface(20) = {1, 10, 11, 12}; // Ensure all loops are included


Physical Surface("domain") = {20};  // Main domain
Physical Curve("boundary") = {1};  // Outer boundary
Physical Curve("holes") = {10, 11, 12};  // Circular holes


// Mesh refinement
MeshSize {c1, c2, c3} = 0.01;
MeshSize {1} = 0.05;




Mesh 2;
Save "mesh.msh";
