//////////////////////////////////////////////////////////////////////
// S U N F I S H   R A Y   T R A C E R ///////////////////////////////
//////////////////////////////////////////////////////////////////////
// Copyright (C) 2018 Jonathan Metzgar                              //
//////////////////////////////////////////////////////////////////////


const float FX_DEGREES_TO_RADIANS = 0.01745329;
const float FX_RADIANS_TO_DEGREES = 57.2957795;

const int SOLVER_MAX_ITERATIONS = 10;
const float SOLVER_MAX_ERROR = 0.01;

const float EPSILON = 1e-6;
const vec3 XEPSILON = vec3(EPSILON, 0.0, 0.0);
const vec3 YEPSILON = vec3(0.0, EPSILON, 0.0);
const vec3 ZEPSILON = vec3(0.0, 0.0, EPSILON);
const int MAX_PATH_DEPTH = 10;
const int MAX_HITABLES = 16;
const int MAX_LIGHTS = 4;

const int RAY_CAST = 0;
const int RAY_TRACE = 1;
const int PATH_TRACE = 2;
const int RenderMode = 0;

const int HITABLE_SPHERE = 0;
const int HITABLE_CYLINDER = 1;
const int HITABLE_PLANE = 2;
const int HITABLE_CONE = 3;
const int HITABLE_DISK = 4;
const int HITABLE_TORUS = 5;
const int HITABLE_BOX = 6;
const int HITABLE_XYRECT = 7;
const int HITABLE_XZRECT = 8;
const int HITABLE_YZRECT = 9;
const int HITABLE_SQUADRIC = 10;
const int HITABLE_MESH = 11;

const int LIGHT_POINT = 0;
const int LIGHT_DIRECTION = 1;
const int LIGHT_SPHERE = 2;

const int MATERIAL_DIFFUSE = 0;
const int MATERIAL_SPECULAR = 1;
const int MATERIAL_DIELECTRIC = 2;
const int MATERIAL_EMISSION = 3;

const int SKY_SHIRLEY = 0;
const int SKY_CUBEMAP = 1;
const int SKY_CUBEMAPBLUR = 2;
const int SKY_DAWN = 3;
const int SKY_NONE = 4;
const int iSkyMode = SKY_DAWN;


const vec3 Left = vec3(-1.0, 0.0, 0.0);
const vec3 Right = vec3(1.0, 0.0, 0.0);
const vec3 Up = vec3(0.0, 1.0, 0.0);
const vec3 Down = vec3(0.0, -1.0, 0.0);
const vec3 Forward = vec3(0.0, 0.0, 1.0);
const vec3 Backward = vec3(0.0, 0.0, -1.0);
const vec3 One = vec3(1.0, 1.0, 0.0);
const vec3 Zero = vec3(0.0, 0.0, 0.0);
const vec3 OneHalf = vec3(0.5, 0.5, 0.5);
const vec3 OneThird = vec3(1.0/3.0, 1.0/3.0, 1.0/3.0);
const vec3 OneFourth = vec3(0.25, 0.25, 0.25);
const vec3 OneFifth = vec3(1.0/5.0, 1.0/5.0, 1.0/5.0);
const vec3 TwoThirds = vec3(2.0/3.0, 2.0/3.0, 2.0/3.0);
const vec3 TwoFifths = vec3(2.0/5.0, 2.0/5.0, 2.0/5.0);
const vec3 ThreeFourths = vec3(0.75, 0.75, 0.75);
const vec3 ThreeFifths = vec3(3.0/5.0, 3.0/5.0, 3.0/5.0);
const vec3 FourFifths = vec3(4.0/5.0, 4.0/5.0, 4.0/5.0);


//////////////////////////////////////////////////////////////////////
// C O L O R S ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


const vec3 Black = vec3(0.0, 0.0, 0.0);
const vec3 White = vec3(1.0, 1.0, 1.0);
const vec3 Red = vec3(1.0, 0.0, 0.0);
const vec3 Orange = vec3(1.0, 0.5, 0.0);
const vec3 Yellow = vec3(1.0, 1.0, 0.0);
const vec3 YellowGreen = vec3(0.5, 1.0, 0.0);
const vec3 Green = vec3(0.0, 1.0, 0.0);
const vec3 GreenBlue = vec3(0.0, 1.0, 0.5);
const vec3 Cyan = vec3(0.0, 1.0, 1.0);
const vec3 BlueGreen = vec3(0.0, 0.5, 1.0);
const vec3 Blue = vec3(0.0, 0.0, 1.0);
const vec3 Purple = vec3(0.5, 0.0, 1.0);
const vec3 Magenta = vec3(1.0, 0.0, 1.0);
const vec3 Rose = vec3(1.0, 0.0, 0.5);
const vec3 ArneBlack = vec3(0.0, 0.0, 0.0);
const vec3 ArneGray = vec3(0.616, 0.616, 0.616);
const vec3 ArneWhite = vec3(1.0, 1.0, 1.0);
const vec3 ArneRed = vec3(0.745, 0.149, 0.2);
const vec3 ArneMeat = vec3(0.878, 0.435, 0.545);
const vec3 ArneDarkBrown = vec3(0.286, 0.235, 0.169);
const vec3 ArneBrown = vec3(0.643, 0.392, 0.133);
const vec3 ArneOrange = vec3(0.922, 0.537, 0.192);
const vec3 ArneYellow = vec3(0.969, 0.886, 0.42);
const vec3 ArneDarkGreen = vec3(0.184, 0.282, 0.306);
const vec3 ArneGreen = vec3(0.267, 0.537, 0.102);
const vec3 ArneSlimeGreen = vec3(0.639, 0.808, 0.153);
const vec3 ArneNightBlue = vec3(0.106, 0.149, 0.196);
const vec3 ArneSeaBlue = vec3(0, 0.341, 0.518);
const vec3 ArneSkyBlue = vec3(0.192, 0.635, 0.949);
const vec3 ArneCloudBlue = vec3(0.698, 0.863, 0.937);
const vec3 ArneDarkBlue = vec3(0.204, 0.165, 0.592);
const vec3 ArneDarkGray = vec3(0.396, 0.427, 0.443);
const vec3 ArneLightGray = vec3(0.8, 0.8, 0.8);
const vec3 ArneDarkRed = vec3(0.451, 0.161, 0.188);
const vec3 ArneRose = vec3(0.796, 0.263, 0.655);
const vec3 ArneTaupe = vec3(0.322, 0.31, 0.251);
const vec3 ArneGold = vec3(0.678, 0.616, 0.2);
const vec3 ArneTangerine = vec3(0.925, 0.278, 0);
const vec3 ArneHoney = vec3(0.98, 0.706, 0.043);
const vec3 ArneMossyGreen = vec3(0.067, 0.369, 0.2);
const vec3 ArneDarkCyan = vec3(0.078, 0.502, 0.494);
const vec3 ArneCyan = vec3(0.082, 0.761, 0.647);
const vec3 ArneBlue = vec3(0.133, 0.353, 0.965);
const vec3 ArneIndigo = vec3(0.6, 0.392, 0.976);
const vec3 ArnePink = vec3(0.969, 0.557, 0.839);
const vec3 ArneSkin = vec3(0.957, 0.725, 0.565);


//////////////////////////////////////////////////////////////////////
// BEGIN SUNFISH GLSL RAY TRACER /////////////////////////////////////
//////////////////////////////////////////////////////////////////////


struct Material {
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    float indexOfRefraction;
    float roughness;
    int type;
};


struct Hitable {
    int type;      // see HITABLE_ constants at top of file
    vec3 position; // for spheres, cylinders, and cones
    float radius;  // for spheres, cylinders, and cones
    float width;   // for rects
    float height;  // for rects, cylinders and cones
    float a;       // for torus
    float b;       // for torus
    float n;       // for superquadric ellipsoid/toroid
    float e;       // for superquadric ellipsoid/toroid
    vec3 box0;     // 
    vec3 box1;
    vec3 normal;   // for planes
    Material material;
};
    

struct Light {
    int type;
    vec3 position;
    vec3 direction;
    vec3 color;
};

    
struct Ray {
    vec3 origin;
    vec3 direction;
};

    
struct HitRecord {
    int i;    // which Hitable object
    float t;  // Ray = x0 + t*x1
    vec3 P;   // Point where ray intersects object
    vec3 N;   // Geometric normal
    vec3 UVW; // Texture Coordinates
    vec3 Kd;  // Object color
    int isEmissive;
};
    
    
    
//////////////////////////////////////////////////////////////////////
// S C E N E G R A P H ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


Hitable Hitables[MAX_HITABLES];
Light Lights[MAX_LIGHTS];
int HitableCount = 0;    
int LightCount = 0;


void sfAddHitable(Hitable hitable)
{
    if (HitableCount >= MAX_HITABLES)
        return;
    for (int i = 0; i < MAX_HITABLES; i++)
    {
        if (i == HitableCount)
        {
            Hitables[i] = hitable;
            HitableCount++;
            break;
        }
    }
}


void sfAddLight(Light light)
{
    if (LightCount >= MAX_LIGHTS)
        return;
    for (int i = 0; i < MAX_LIGHTS; i++)
    {
        if (i == LightCount)
        {
            Lights[i] = light;
            LightCount++;
            break;
        }
    }
}


//////////////////////////////////////////////////////////////////////
// M A T H E M A T I C S /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


float seed = 0.0;

void srand(vec2 fragCoord) {
    seed = dot(fragCoord, vec2(12.9898,78.233));
}

float rand() {
    seed = fract(sin(seed) * 43758.5453);
    return seed;
}

    
vec3 sfRayOffset(Ray r, float t)
{
    return r.origin + t * r.direction;
}


vec3 sfRandomDirection(vec3 N)
{
    return normalize(vec3(rand(), rand(), rand()) * N);
}


bool sfSolveQuadratic(in float a, in float b, in float c, out float t1, out float t2)
{
    float discriminant = b * b - 4.0 * a * c;
    float denom = 2.0 * a;
    if (denom != 0.0 && discriminant > 0.0) {
        float s = sqrt(discriminant);
        float r1 = (-b - s) / denom;
        float r2 = (-b + s) / denom;
        if (r1 < r2) {
            t1 = r1;
            t2 = r2;
        }
        else
        {
            t1 = r2;
            t2 = r1;
        }
        return true;
    }
    return false;
}


bool sfNewtonQuadratic(vec3 abc, out float x)
{
    float xo = 0.0;
    float a = abc.x;
    float b = abc.y;
    float c = abc.z;

    for (int i = 0; i < SOLVER_MAX_ITERATIONS; i++) {
        float xo2 = xo*xo;
        float f = a*xo2 + b*xo + c;
        if (f < SOLVER_MAX_ERROR) {
            x = xo;
            return true;
        }
        float g = 2.0*a*xo + 2.0*b;
        xo = xo - f/g;
    }
    return false;
}


bool sfNewtonCubic(vec4 abcd, out float x)
{
    float xo = 0.0;
    float a = abcd.x;
    float b = abcd.y;
    float c = abcd.z;
    float d = abcd.w;

    for (int i = 0; i < SOLVER_MAX_ITERATIONS; i++) {
        float xo2 = xo*xo;
        float xo3 = xo2*xo;
        float f = a*xo3 + b*xo2 + c*xo + d;
        if (f < SOLVER_MAX_ERROR) {
            x = xo;
            return true;
        }
        float g = 3.0*a*xo2 + 2.0*b*xo + c;
        xo = xo - f/g;
    }
    return false;
}


bool sfNewtonQuartic(in vec4 bcde, out float x)
{
    float xo = 0.0;
    float b = bcde.x;
    float c = bcde.y;
    float d = bcde.z;
    float e = bcde.w;

    for (int i = 0; i < SOLVER_MAX_ITERATIONS; i++) {
        float xo2 = xo*xo;
        float xo3 = xo2*xo;
        float xo4 = xo2*xo2;
        float f = xo4 + b*xo3 + c*xo2 + d*xo + e;
        if (f < SOLVER_MAX_ERROR) {
            x = xo;
            return true;
        }
        float g = 4.0*xo3 + 3.0*b*xo2 + 2.0*c*xo + d;
        xo = xo - f/g;
    }
    return false;
}


// solve depressed cubic
bool sfSolveDCubic(vec2 pq, out vec2 roots)
{
    float p = pq.x;
    float q = pq.y;
    float discriminant = q*q - 4.0/27.0 * p*p*p;
    if (discriminant < 0.0) return false;
    if (discriminant < EPSILON) {
        roots.x = -0.5 * q;
        roots.y = 1e6;
        return true;
    }
    
    vec2 w = pow(vec2(-0.5 * (q - sqrt(discriminant)),
                      -0.5 * (q + sqrt(discriminant))),
                 vec2(1.0/3.0));
    vec2 x = w - p / (3.0 * w);
    
    if (x.x < x.y) {
        roots = x;
    } else {
        roots = x.yx;
    }
    return true;
}


/*
bool sfSolveQuartic(vec4 bcde, out vec4 roots)
{
    float b = bcde.x;
    float c = bcde.y;
    float d = bcde.z;
    float e = bcde.w;
    const float inv27 = 1.0/27.0;
    float b2 = b*b;
    float c2 = c*c;
    float delta0 = c*c - 3.0*b*d + 12.0*e;
    float delta1 = 2.0*c*c*c - 9.0*b*c*d + 27.0*b*b*e + 27.0*d*d - 72.0*c*e;
    float discriminant = inv27 * (4.0*pow(delta0, 3.0) - delta1*delta1);
    if (discriminant > 0.0) {
        // check if all roots are complex
        float P = 8.0*c - 3.0*b2;
        float D = 64.0*e + 16.0*(b2*c - c2 - b*d) - 3.0*b2*b2;
        if (P < 0.0 && D < 0.0) return false;
    }
    // Convert to depressed quartic
    // t^4 + pt^2 + qt + r = 0
    float p = c - (3.0/8.0)*b2;
    float q = 0.125 * (b2*b - 4.0*b*c + 8.0*d);
    float r = (-3.0*b2*b2 + 256.0*e - 64.0*b*d + 16.0*b2*c) / (256.0);
    float p2 = p*p;
    float q2 = q*q;
    
    // solve for m, 8m^3 + 8pm^2 + (2p^2-ur)m - q^2 = 0
    float cubic_pq = vec2(
        -0.25 * p2 - 3.0*r,
        -0.009259259*p2*p + 0.33333*p*r + 0.125*q2
    );
    vec2 cubic_roots;
    if (!sfSolveDCubic(cubic_pq, cubic_roots))
        return false;
    
    
    
    float phi = acos(delta1 / (2.0 * pow(delta1, 3.0/2.0)));
    float cos_phi = cos(phi/3.0);
    float S = 0.5 * sqrt(-0.66666 * (p + sqrt(delta0)*cos_phi));
    discriminant = 
    roots.x = -0.25*b - S + 0.5*sqrt(-4.0*S*S - 2.0*p + q/S);
    roots.y = -0.25*b - S - 0.5*sqrt(-4.0*S*S - 2.0*p + q/S);
    roots.z = -0.25*b + S + 0.5*sqrt(-4.0*S*S - 2.0*p + q/S);
    roots.w = -0.25*b + S - 0.5*sqrt(-4.0*S*S - 2.0*p + q/S);    
    return true;
}
*/

    
//////////////////////////////////////////////////////////////////////
// F A C T O R Y /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
    

Material sfCreateMaterial(vec3 Kd, vec3 Ks, float roughness)
{
    Material m;
    m.Kd = Kd;
    m.Ks = Ks;
    m.Ke = Black;
    m.roughness = roughness;
    m.indexOfRefraction = 1.333;
    m.type = MATERIAL_DIFFUSE;
    return m;
}


Material sfCreateDiffuseMaterial(vec3 Kd, float roughness)
{
    Material m;
    m.Kd = Kd;
    m.Ks = White;
    m.Ke = Black;
    m.roughness = roughness;
    m.indexOfRefraction = 1.0;
    m.type = MATERIAL_DIFFUSE;
    return m;
}


Material sfCreateSpecularMaterial(vec3 Ks, float roughness)
{
    Material m;
    m.Kd = Black;
    m.Ks = Ks;
    m.Ke = Black;
    m.roughness = roughness;
    m.indexOfRefraction = 1.0;
    m.type = MATERIAL_SPECULAR;
    return m;
}


Material sfCreateDielectricMaterial(vec3 Kd, float indexOfRefraction)
{
    Material m;
    m.Kd = Kd;
    m.Ks = White;
    m.Ke = Black;
    m.roughness = 0.0;
    m.indexOfRefraction = indexOfRefraction;
    m.type = MATERIAL_DIELECTRIC;
    return m;
}


Material sfCreateEmissionMaterial(vec3 Ke)
{
    Material m;
    m.Kd = ArneBlack;
    m.Ks = ArneBlack;
    m.Ke = Ke;
    m.roughness = 0.0;
    m.indexOfRefraction = 1.0;
    m.type = MATERIAL_EMISSION;
    return m;
}


Ray sfCreateRay(vec3 origin, vec3 dir)
{
    Ray r;
    r.origin = origin;
    r.direction = normalize(dir);
    return r;
}


Light sfCreateLight(int type, vec3 position, vec3 direction, vec3 color)
{
    Light l;
    l.type = type;
    l.position = position;
    l.direction = direction;
    l.color = color;
    return l;
}


HitRecord sfCreateHitRecord(float t, vec3 P, vec3 N)
{
    HitRecord h;
    h.t = t;
    h.P = P;
    h.N = N;
    return h;
}


Hitable sfCreateSphere(vec3 position, float radius, Material material)
{
    Hitable h;
    h.type = HITABLE_SPHERE;
    h.position = position;
    h.radius = radius;
    h.material = material;
    return h;
}


Hitable sfCreatePlane(vec3 position, vec3 normal, Material material)
{
    Hitable h;
    h.type = HITABLE_PLANE;
    h.position = position;
    h.normal = normalize(normal);
    h.material = material;
    return h;
}


Hitable sfCreateDisk(vec3 position, vec3 normal, float radius, Material material)
{
    Hitable h;
    h.type = HITABLE_DISK;
    h.position = position;
    h.normal = normalize(normal);
    h.radius = radius;
    h.material = material;
    return h;
}


Hitable sfCreateCylinder(vec3 position, float radius, float height, Material material)
{
    Hitable h;
    h.type = HITABLE_CYLINDER;
    h.position = position;
    h.radius = radius;
    h.height = height;
    h.material = material;
    return h;
}


Hitable sfCreateCone(vec3 position, float radius, float height, Material material)
{
    Hitable h;
    h.type = HITABLE_CONE;
    h.position = position;
    h.radius = radius;
    h.height = height;
    h.material = material;
    return h;
}


Hitable sfCreateTorus(vec3 position, float radiusA, float radiusB, Material material)
{
    Hitable h;
    h.type = HITABLE_TORUS;
    h.position = position;
    h.a = radiusA; // theoretically, we could support ellipses
    h.b = radiusB; // and a is x-axis, b is y-axis
    h.material = material;
    return h;
}


Hitable sfCreateSuperquadric(vec3 position, float radius, float n, float e, Material material)
{
    Hitable h;
    h.type = HITABLE_SQUADRIC;
    h.position = position;
    h.radius = radius;
    h.n = 2.0/n;
    h.e = 2.0/e;
    h.material = material;
    return h;
}


Hitable sfCreateBox(vec3 position, vec3 p1, vec3 p2, Material material)
{
    Hitable h;
    h.type = HITABLE_BOX;
    h.position = position;
    h.box0 = min(p1, p2);
    h.box1 = max(p1, p2);
    h.material = material;
    return h;
}


Hitable sfCreateRect(int type, vec3 position, vec3 p1, vec3 p2, Material material)
{
    Hitable h;
    h.type = type;
    h.position = position;
    h.box0 = min(p1, p2);
    h.box1 = max(p1, p2);
    h.material = material;
    return h;
}


Hitable sfCreateMesh(vec3 position, Material material)
{
    Hitable h;
    h.type = HITABLE_MESH;
    h.position = position;
    h.material = material;
    return h;
}


//////////////////////////////////////////////////////////////////////
// I N T E R S E C T I O N S /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


bool sfAnyHitSphere(Ray r, vec3 position, float radius, float thickness)
{
    vec3 O = r.origin - position;
    vec3 D = r.direction;
    float a = dot(D,D);
    float b = dot(O,D);
    float c = dot(O,O) - radius*radius;
    float t1, t2;
    if (!sfSolveQuadratic(a, b, c, t1, t2)) return false;
    // DEBUG: ignore the slabs for now...
    return true;
    if (t1 > 0.0) {
        vec3 p = sfRayOffset(r, t1);
        if (p.y > position.y + thickness) return false;
        if (p.y < position.y - thickness) return false;
    }
    if (t2 > 0.0) {
        vec3 p = sfRayOffset(r, t2);
        if (p.y > position.y + thickness) return false;
        if (p.y < position.y - thickness) return false;
    }
    return true;
}


bool sfRayIntersectSphere(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    vec3 originToCenter = r.origin - s.position;
    // solve quadratic equation
    float a = dot (r.direction, r.direction);
    float b = dot(originToCenter, r.direction);
    float c = dot(originToCenter, originToCenter) - s.radius*s.radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0.0) {
        float t = (-b - sqrt(discriminant)) / a;
        if (t < tMax && t > tMin) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = (h.P - s.position) / s.radius;
            return true;
        }
        t = (-b - sqrt(discriminant)) / a;
        if (t < tMax && t > tMin) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = (h.P - s.position) / s.radius;
            return true;
        }
    }
    return false;
}


bool sfRayIntersectBox(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{    
    // s.box0 and s.box1 are the minimum and maximum coordinates
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    vec3 diff = 0.5 * (s.box0 + s.box1);
    vec3 planesN[6];
    vec3 planesP[6];
    planesN[0] = Left;
    planesN[1] = Right;
    planesN[2] = Up;
    planesN[3] = Down;
    planesN[4] = Backward;
    planesN[5] = Forward;
    planesP[0] = vec3(s.box0.x, diff.y, diff.z);
    planesP[1] = vec3(s.box1.x, diff.y, diff.z);
    planesP[2] = vec3(diff.x, s.box0.y, diff.z);
    planesP[3] = vec3(diff.x, s.box1.y, diff.z);
    planesP[4] = vec3(diff.x, diff.y, s.box0.z);
    planesP[5] = vec3(diff.x, diff.y, s.box1.z);
    vec3 Nmin;
    vec3 Nmax;
    float t0 = 1e6;
    float t1 = 0.0;
    float t = -1.0;
    for (int i = 0; i < 6; i++) {
        float cos_theta = dot(D, -planesN[i]);
        if (cos_theta >= EPSILON) {
            vec3 diff = planesP[i] - O;
            float t = dot(diff, -planesN[i]) / cos_theta;
            if (t < t0) {
                Nmin = planesN[i];
                t0 = t;
            }
            if (t > t1) {
                Nmax = planesN[i];
                t1 = t;
            }
        } else if (-cos_theta >= EPSILON) {
            vec3 diff = planesP[i] - O;
            float t = dot(diff, planesN[i]) / -cos_theta;
            if (t < t0) {
                Nmin = -planesN[i];
                t0 = t;
            }
            if (t > t1) {
                Nmax = -planesN[i];
                t1 = t;
            }
        }
    }
    if (t0 > t1) return false;
    if (t0 > tMin && t0 < tMax) {
        h.t = t0;
        h.P = sfRayOffset(r, t0);
        h.N = Nmin;
        return true;
    }
    if (t1 > tMin && t1 < tMax) {
        h.t = t0;
        h.P = sfRayOffset(r, t0);
        h.N = Nmax;
        return true;
    }
    return false;
}


bool sfRayIntersectPlane(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    float cos_theta = dot(r.direction, -s.normal);
    if (abs(cos_theta) >= EPSILON) {
        vec3 diff = s.position - r.origin;
        float t = dot(diff, -s.normal) / cos_theta;
        if (t > tMin && t < tMax) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = cos_theta > 0.0 ? s.normal : -s.normal;
            return true;
        }        
    }
    return false;
}


bool sfRayIntersectDisk(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    float cos_theta = dot(r.direction, -s.normal);
    if (cos_theta >= EPSILON) {
        vec3 diff = s.position - r.origin;
        float t = dot(diff, -s.normal) / cos_theta;
        vec3 p = sfRayOffset(r, t);
        bool inside = sqrt(dot(p, p)) <= s.radius;
        if (t > tMin && t < tMax && inside) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = cos_theta > 0.0 ? -s.normal : s.normal;
            return true;
        }        
    }
    return false;
}


bool sfRayIntersectCone(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    float root1 = -1.0;
    float root2 = -1.0;
    vec3 D = r.direction;
    vec3 E = r.origin - s.position;
    E.y -= s.height;
    float a = D.x*D.x/s.radius + D.z*D.z/s.radius - D.y*D.y;
    float b = 2.0 * (D.x*E.x/s.radius + D.z*E.z/s.radius - D.y*E.y);
    float c = E.x*E.x/s.radius + E.z*E.z/s.radius - E.y*E.y;
    if (!sfSolveQuadratic(a, b, c, root1, root2)) {
        return false;
    }
    float t;
    if (root1 > 0.0)
        t = root1;
    else
        t = root2;
    if (t < EPSILON)
        return false;
    if (t > tMin && t < tMax) {
        vec3 P = sfRayOffset(r, t);
        if (P.y < s.position.y) return false;
        if (P.y > s.position.y + s.height) return false;
        vec3 N = vec3(P.x - s.position.x, 0.0, P.z - s.position.z);
        h.t = t;
        h.P = P;
        h.N = N;
        return true;
    }
    
    return false;
}


bool sfRayIntersectCylinder(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    float root1 = -1.0;
    float root2 = -1.0;
    vec2 rdir = r.direction.xz;
    vec2 rpos = r.origin.xz - s.position.xz;
    float a = dot(rdir, rdir);
    float b = 2.0 * dot(rpos, rdir);
    float c = dot(rpos, rpos) - s.radius*s.radius;
    if (!sfSolveQuadratic(a, b, c, root1, root2)) {
        return false;
    }
    float t;
    if (root1 > 0.0)
        t = root1;
    else
        t = root2;
    if (t < EPSILON)
        return false;
    if (t > tMin && t < tMax) {
        vec3 P = sfRayOffset(r, t);
        if (P.y < s.position.y) return false;
        if (P.y > s.position.y + s.height) return false;
        vec3 N = vec3(P.x - s.position.x, 0.0, P.z - s.position.z);
        h.t = t;
        h.P = P;
        h.N = N;
        return true;
    }
    
    return false;
}


bool sfRayIntersectXYRect(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    float t = -O.z / D.z;
    if (t < tMin || t > tMax)
        return false;
    
    vec3 P = O + t * D;
    if (P.x < s.box0.x || P.x > s.box1.x ||
        P.y < s.box0.y || P.y > s.box1.y)
        return false;
    
    h.t = t;
    h.P = sfRayOffset(r, t);
    h.N = D.z > 0.0 ? Forward : Backward;
    h.UVW = vec3((P.xy - s.box0.xy) / (s.box1.xy - s.box0.xy), 0.0);
    return true;
}


bool sfRayIntersectXZRect(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    float t = -O.y / D.y;
    if (t < tMin || t > tMax)
        return false;
    
    vec3 P = O + t * D;
    if (P.x < s.box0.x || P.x > s.box1.x ||
        P.z < s.box0.z || P.z > s.box1.z)
        return false;
    
    h.t = t;
    h.P = sfRayOffset(r, t);
    h.N = D.y > 0.0 ? Up : Down;
    h.UVW = vec3((P.xz - s.box0.xz) / (s.box1.xz - s.box0.xz), 0.0);
    return true;
}


bool sfRayIntersectYZRect(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    float t = -O.x / D.x;
    if (t < tMin || t > tMax)
        return false;
    
    vec3 P = O + t * D;
    if (P.y < s.box0.y || P.y > s.box1.y ||
        P.z < s.box0.z || P.z > s.box1.z)
        return false;
    
    h.t = t;
    h.P = sfRayOffset(r, t);
    h.N = D.x > 0.0 ? Left : Right;
    h.UVW = vec3((P.yz - s.box0.yz) / (s.box1.yz - s.box0.yz), 0.0);
    return true;
}


float sdfSphere(vec3 p, float s) {
    return length(p) - s;
}


vec3 sdfSphereNormal(vec3 p, float s) {
    return normalize(vec3(
        sdfSphere(vec3(p.x + EPSILON, p.y, p.z), s) - sdfSphere(vec3(p.x - EPSILON, p.y, p.z), s),
        sdfSphere(vec3(p.x, p.y + EPSILON, p.z), s) - sdfSphere(vec3(p.x, p.y - EPSILON, p.z), s),
        sdfSphere(vec3(p.x, p.y, p.z + EPSILON), s) - sdfSphere(vec3(p.x, p.y, p.z - EPSILON), s)));
}

float sphSDF(vec3 p, Hitable s) {
    float r = length(p);
    float phi = acos(p.y / r);
    float theta = atan(p.z, p.x);
    vec3 d = vec3(pow(abs(cos(phi)), s.n) * pow(abs(cos(theta)), s.e),
	              pow(abs(sin(phi)), s.n),
    	          pow(abs(cos(phi)), s.n) * pow(abs(sin(theta)), s.e));
    return r - length(d);
    //float m = s.n;//max(s.e, s.n);
    //float a = pow(abs(p.x), m);
    //float b = pow(abs(p.y), m);
    //float c = pow(abs(p.z), m);
    //return (pow(a + b + c, 1.0/m) - 1.0);    
}


vec3 sphNormalSDF(vec3 p, Hitable s) {
    float base = sphSDF(p, s);
    return normalize(vec3(
        sphSDF(p + XEPSILON, s) - base,
        sphSDF(p + YEPSILON, s) - base,
        sphSDF(p + ZEPSILON, s) - base));
}


bool sfRayIntersectSuperquadric(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    float t = tMin;
    const int MAX_RAY_STEPS = 64;
    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        vec3 P = O + t * D;
        //float d = sphereSDF(P / s.radius) * s.radius;
        float d = sphSDF(P/s.radius, s)*s.radius;
        if (d < EPSILON) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = -sphNormalSDF(P, s);
            return true;
        }
        t += d;
        if (t > tMax) {
            return false;
        }
    }
    return false;
}


float sdfTorus(in vec3 p, in Hitable s)
{
    vec2 q = vec2(length(p.xy) - s.a, p.z);
    return length(q) - s.b;
}


vec3 sdfTorusNormal(vec3 p, Hitable s) {
    float base = sdfTorus(p, s);
    return normalize(vec3(
        sdfTorus(p + XEPSILON, s) - base,
        sdfTorus(p + YEPSILON, s) - base,
        sdfTorus(p + ZEPSILON, s) - base));
}


bool sfRayIntersectTorus(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{    
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    
    float t = tMin;
    const int MAX_RAY_STEPS = 64;
    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        vec3 P = O + t * D;
        float d = sdfTorus(P, s);
        if (d < EPSILON) {
            h.t = t;
            h.P = sfRayOffset(r, t);
            h.N = -sdfTorusNormal(P, s);
            return true;
        }
        t += d;
        if (t > tMax) {
            return false;
        }
    }
    return false;
}


// from http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float dot2(vec3 v) { return dot(v,v); }

float sdfTriangle(in vec3 p,
                  in vec3 a,
                  in vec3 b,
                  in vec3 c,
                  out vec3 N)
{
    vec3 ba = b - a; vec3 pa = p - a;
    vec3 cb = c - b; vec3 pb = p - b;
    vec3 ac = a - c; vec3 pc = p - c;
    N = cross( ba, ac );

    return sqrt(
    (sign(dot(cross(ba,N),pa)) +
     sign(dot(cross(cb,N),pb)) +
     sign(dot(cross(ac,N),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(N,pa)*dot(N,pa)/dot2(N) );
}


bool sfRayIntersectMesh(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    const int NUM_VERTICES = 11;
    const int NUM_TRIANGLES = 10;
    const int MAX_ITERATIONS = 64;
    vec3 vertices[NUM_VERTICES];
    vertices[0] = 0.25 * vec3( 0.0,  4.0, 0.0);
    vertices[1] = 0.25 * vec3(-2.0, -1.0, 0.0);
    vertices[2] = 0.25 * vec3( 0.0, -2.0, 0.0);
    vertices[3] = 0.25 * vec3( 2.0, -1.0, 0.0);
    vertices[4] = 0.25 * vec3( 0.0,  0.0, 1.0);
    vertices[5] = 0.25 * vec3( 3.0,  0.0, 0.0);
    vertices[6] = 0.25 * vec3( 4.0, -1.0, 0.0);
    vertices[7] = 0.25 * vec3( 3.0, -4.0, 0.0);
    vertices[8] = 0.25 * vec3(-3.0,  0.0, 0.0);
    vertices[9] = 0.25 * vec3(-4.0, -1.0, 0.0);
    vertices[10]= 0.25 * vec3(-3.0, -4.0, 0.0);
    int indices[NUM_TRIANGLES * 3];
    indices[0] = 0; // 0
    indices[1] = 1;
    indices[2] = 2;
    indices[3] = 2; // 1
    indices[4] = 3;
    indices[5] = 0;
    indices[6] = 0; // 2
    indices[7] = 1;
    indices[8] = 4;
    indices[9] = 4; // 3
    indices[10] = 3;
    indices[11] = 0;
    indices[12] = 2; // 4
    indices[13] = 4;
    indices[14] = 1;
    indices[15] = 2; // 5
    indices[16] = 3;
    indices[17] = 4;
    indices[18] = 6; // 6
    indices[19] = 5;
    indices[20] = 3;
    indices[21] = 7; // 7
    indices[22] = 6;
    indices[23] = 3;
    indices[24] = 1;
    indices[25] = 8;
    indices[26] = 9;
    indices[27] = 1;
    indices[28] = 9;
    indices[29] = 10;
    float bestT = tMax;
    vec3 bestN;
    vec3 O = r.origin - s.position;
    vec3 D = r.direction;
    mat3 R = mat3(cos(iTime), 0.0, -sin(iTime),
                  0.0, 1.0, 0.0,
                  sin(iTime), 0.0, cos(iTime)
                  );
    O = R * O;
    D = R * D;
    int idx = 0;
    for (int i = 0; i < NUM_TRIANGLES; i++, idx += 3)
    {
        float t = tMin;
        float lastD = 1e6;
        for (int j = 0; j < MAX_ITERATIONS; j++)
        {
            vec3 testN;
            float d = sdfTriangle(O + t * D,
                                  vertices[indices[idx+0]],
                                  vertices[indices[idx+1]],
                                  vertices[indices[idx+2]],
                                  testN);
            if (d < EPSILON) {
                if (bestT > t) {
                    bestT = t;
                    bestN = testN;
                    break;
                }
            }
            if (lastD > d) { lastD = d; }
            else if (j != 0) {
                break;
            }
            t += d;
            if (t > bestT) break;
        }
    }
    if (bestT > tMin && bestT < tMax) {
        h.t = bestT;
        h.P = sfRayOffset(r, bestT);
        h.N = -normalize(bestN);
        return true;
    }
    return false;
}


bool sfRayIntersect(Hitable s, Ray r, float tMin, float tMax, out HitRecord h)
{
    if (s.type == HITABLE_MESH) {
        return sfRayIntersectMesh(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_SPHERE) {
        return sfRayIntersectSphere(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_BOX) {
        return sfRayIntersectBox(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_PLANE) {
        return sfRayIntersectPlane(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_SQUADRIC) {
        return sfRayIntersectSuperquadric(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_DISK) {
        return sfRayIntersectDisk(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_CONE) {
        return sfRayIntersectCone(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_CYLINDER) {
        return sfRayIntersectCylinder(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_TORUS) {
        return sfRayIntersectTorus(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_XYRECT) {
        return sfRayIntersectXYRect(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_XZRECT) {
        return sfRayIntersectXZRect(s, r, tMin, tMax, h);
    }
    if (s.type == HITABLE_YZRECT) {
        return sfRayIntersectYZRect(s, r, tMin, tMax, h);
    }
    return false;
}


//////////////////////////////////////////////////////////////////////
// S U N F I S H   R A Y   T R A C E R ///////////////////////////////
//////////////////////////////////////////////////////////////////////
// Copyright (C) 2018 Jonathan Metzgar                              //
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// S H A D E R S /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


vec3 sfShadeSkyCubeMap(Ray r)
{
    return texture(iChannel0, r.direction).rgb;
}


vec3 sfShadeSkyCubeMapBlur(Ray r)
{
    return texture(iChannel1, r.direction).rgb;
}


vec3 sfShadeSkyShirley(Ray r)
{
	float t = 0.5 * (r.direction.y + 1.0);
	return (1.0 - t) * ArneWhite + t * ArneSkyBlue;
}


vec3 sfShadeSkyDawn(Ray r)
{
	float t = 0.5 * (r.direction.y + 1.0);
	return (1.0 - t) * Orange + t * ArneSkyBlue;
}


vec3 sfShadeSky(Ray r)
{
    if (iSkyMode == SKY_SHIRLEY) return sfShadeSkyShirley(r);
    if (iSkyMode == SKY_DAWN)    return sfShadeSkyDawn(r);
    if (iSkyMode == SKY_CUBEMAP) return sfShadeSkyCubeMap(r);
    if (iSkyMode == SKY_CUBEMAPBLUR) return sfShadeSkyCubeMapBlur(r);
    return Black;
}



void test_ray()
{
	Ray r = sfCreateRay(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0));
}

// Shaders


bool sfClosestHit(in Ray r, out HitRecord h)
{		
    int hit = -1;
    float t_min = 0.0;
    float t_max = 1e6;
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (i >= HitableCount) break;
		if (sfRayIntersect(Hitables[i], r, t_min, t_max, h))
		{
            hit = i;
			t_max = h.t;
            if (Hitables[i].material.type == MATERIAL_EMISSION)
            {
                h.Kd = Hitables[i].material.Ke;
                h.isEmissive = 1;
            }
            else
            {
                h.Kd = Hitables[i].material.Kd;
                h.isEmissive = 0;
            }
		}
	}
    h.i = hit;
    if (hit < 0) return false;
    return true;
}


vec3 sfBounce1(Ray r)
{
	HitRecord h = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
	int hit = -1;
	float t_min = 0.0;
	float t_max = 1e6;
	vec3 Kd;
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (i >= HitableCount) break;
		if (sfRayIntersect(Hitables[i], r, t_min, t_max, h))
		{
			t_max = h.t;
			hit = i;
		}
	}
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (hit == i)
		{
			return Hitables[i].material.Kd;
		}
	}
	return sfShadeSky(r);
}

vec3 sfBounce2(Ray r)
{
	HitRecord h = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
	int hit = -1;
	float t_min = 0.0;
	float t_max = 1e6;
	vec3 Kd;
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (i >= HitableCount) break;
		if (sfRayIntersect(Hitables[i], r, t_min, t_max, h))
		{
			t_max = h.t;
			hit = i;
		}
	}
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (hit == i)
		{
            h.N = normalize(h.N);
			Ray reflectedRay = sfCreateRay(
				h.P,
				reflect(r.direction, h.N)
				);
			return Hitables[i].material.Kd * sfBounce1(reflectedRay);
		}
	}
	return sfShadeSky(r);
}

vec3 sfBounce3(Ray r)
{
	HitRecord h = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
	int hit = -1;
	float t_min = 0.0;
	float t_max = 1e6;
	vec3 Kd;
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (i >= HitableCount) break;
		if (sfRayIntersect(Hitables[i], r, t_min, t_max, h))
		{
			t_max = h.t;
			hit = i;
		}
	}
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (hit == i)
		{
            if (Hitables[i].material.type == MATERIAL_EMISSION) {
                return Hitables[i].material.Ke;
            }
            h.N = normalize(h.N);

			Ray reflectedRay = sfCreateRay(
				h.P,
				reflect(r.direction, h.N)
				);
			return Hitables[i].material.Kd * sfBounce2(reflectedRay);
		}
	}
	return sfShadeSky(r);
}


vec3 sfRayTrace(Ray r)
{
	HitRecord h = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
	int hit = -1;
	float t_min = 0.0;
	float t_max = 1e6;
	vec3 Kd;
    if (!sfClosestHit(r, h))
        return sfShadeSky(r);    
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (h.i == i)
		{
            if (Hitables[i].material.type == MATERIAL_EMISSION) {
                return Hitables[i].material.Ke;
            }
            //if (Hitables[i].type == HITABLE_TORUS) {
            //    return Magenta;
            //}
            
            h.N = normalize(h.N);
            
			Ray reflectedRay = sfCreateRay(
				h.P,
				reflect(r.direction, h.N)
				);
            float NdotL = dot(reflectedRay.direction, h.N);
			return Hitables[i].material.Kd * NdotL * sfBounce3(reflectedRay);
		}
	}
	return sfShadeSky(r);
}


vec3 sfPathTrace(Ray r)
{
    vec3 accum = Black;
    const int MaxRaysPerPixel = 25;
    const int MaxPaths = 5;    
    int path;
    float anim = 0.0*(sin(iTime));
    for (int i = 0; i < MaxRaysPerPixel; i++)
    {
    	HitRecord h[MaxPaths];
        float costs[MaxPaths];
        vec3  color[MaxPaths];
        Ray ray = r;
        ray.direction += sfRandomDirection(r.direction) * 0.001;
        ray.direction = normalize(r.direction);
        ray.origin.x += anim * rand();
        for (path = 0; path < MaxPaths; path++)
        {
            //ray.origin.x += anim * rand();
            h[path] = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
            if (!sfClosestHit(ray, h[path])) {
                costs[path] = 1.0;
                h[path].Kd = sfShadeSky(ray);
                break;
            }
            
            if (h[path].isEmissive != 0) {
                costs[path] = 1.0;
                break;
            }
            for (int k = 0; k < MAX_HITABLES; k++) {
                if (k == h[path].i && Hitables[k].type == HITABLE_PLANE) {                    
                    float sines = sin(10. * h[path].P.x) * 
                        sin(10. * h[path].P.y) * 
                        sin(10. * h[path].P.z);
                    if (sines < 0.0) {
                        h[path].Kd = White;
                    }
                    break;
                }
            }
                                float sines = sin(10. * h[path].P.x) * 
                        sin(10. * h[path].P.y) * 
                        sin(10. * h[path].P.z);
                    if (sines < 0.0) {
                        h[path].Kd = White;
                    }
            vec3 N = normalize(h[path].N);
            float willReflect = rand();
            const float F0 = 0.33 / 2.33;
            float cos_d = dot(N, ray.direction);
            float fresnel = F0 + (1.0 - F0) * (pow(cos_d, 5.0));
            if (willReflect > fresnel){
                ray = sfCreateRay(h[path].P,
                                  reflect(ray.direction, N));                
            } else {
            	ray = sfCreateRay(h[path].P,
                                  sfRandomDirection(h[path].N));
            }
            costs[path] = max(0.0, dot(N, ray.direction));
        }
        // special case, we only hit the sky
        if (path == 0) accum += h[path].Kd;
        for (int j = 1; j < MaxPaths; j++) {
            if (j > path) break;
            
            accum += costs[j-1] * h[j].Kd;
        }
    }
    accum /= float(MaxRaysPerPixel);
    return accum;
}


vec3 sfRayCast(Ray r)
{
	HitRecord h = sfCreateHitRecord(1e6, vec3(0.0), vec3(0.0));
	int hit = -1;
	float t_min = 0.0;
	float t_max = 1e6;
	vec3 Kd;
    if (!sfClosestHit(r, h))
        return sfShadeSky(r);    
	for (int i = 0; i < MAX_HITABLES; i++)
	{
		if (h.i == i)
		{
            if (Hitables[i].material.type == MATERIAL_EMISSION) {
                return Hitables[i].material.Ke;
            }
            
            //if (Hitables[i].type == HITABLE_TORUS) {
            //    return Magenta;
            //}
            
            vec3 N = normalize(h.N);
            /*
            vec3 N = normalize(h.N);            
            vec3 R = reflect(r.direction, h.N);
			Ray reflectedRay = sfCreateRay(h.P, R);
            float NdotL = dot(R, h.N);
            // shadow ray
            if (sfClosestHit(reflectedRay, h)) {
                return Hitables[i].material.Kd * 0.5;
            }
            vec3 color = sfShadeSky(reflectedRay);
            */
            return Hitables[i].material.Kd * 0.5 + (0.25 * N + 0.25);// + NdotL * color;// * NdotL * color;
		}
	}
	return sfShadeSky(r);
}


Ray sfCreateCameraRay(vec2 uv) {
    vec3 eye = vec3(0.0, 0.0, 5.0);
    vec3 center = vec3(0.0, 0.0, 0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    float aspectRatio = iResolution.x / iResolution.y;
    float fovy = 45.0;
    
    float theta = fovy * FX_DEGREES_TO_RADIANS;
    float halfHeight = tan(theta / 2.0);
    float halfWidth = aspectRatio * halfHeight;
    float distanceToFocus = length(eye - center);
    vec3 w = normalize(eye - center);
    vec3 u = cross(up, w);
    vec3 v = cross(w, u);
    vec3 horizontal = 2.0 * distanceToFocus * halfWidth * u;
    vec3 vertical = 2.0 * distanceToFocus * halfHeight * v;
    vec3 lowerLeftCorner = eye
        - (distanceToFocus*halfWidth) * u
        - (distanceToFocus*halfHeight) * v
        - distanceToFocus * w;
    vec3 window = uv.s * horizontal + uv.t * vertical;
    return sfCreateRay(eye, lowerLeftCorner + window - eye);
}


vec3 Sunfish(in Ray r)
{
    if (RenderMode == RAY_CAST) return sfRayCast(r);
    if (RenderMode == RAY_TRACE) return sfRayTrace(r);
    if (RenderMode == PATH_TRACE) return sfPathTrace(r);
    return sfShadeSky(r);
}


// END SUNFISH GLSL RAY TRACER ///////////////////////////////////////
float osc(float size, float phase) {
    return size * sin(iTime + phase);
}

void CreateScene()
{
    float maxX = 8.0;
    float maxX2 = 4.0;
    vec3 offset = maxX * vec3(iMouse.xy / iResolution.xy, 0.0).xzy - maxX2;
    offset.y += 4.0 + 0.025*sin(iTime);
    //offset = vec3(0.0);
	//Hitables[0] = sfCreateSphere(offset + vec3(2.0, 0.0, 0.5), 0.5,
	//	sfCreateMaterial(ArneWhite, ArneBlack, 0.0));
    //Hitables[1] = sfCreateSphere(offset + vec3(-2.0, 0.0, 0.5), 0.5,
    //    sfCreateMaterial(ArneRed, ArneRed, 0.0));
	//HitableCount = 2;
	//sfAddHitable(sfCreateSphere(offset + vec3(0.0, -1000.5, -0.5), 1000.0, sfCreateMaterial(ArneBrown, ArneBrown, 0.0)));
    
    //return;
    /*
    sfAddHitable(sfCreatePlane(vec3(0.0, -5.0, 0.0),
                               Up,
                               sfCreateDiffuseMaterial(0.5*White, 0.0)));
    sfAddHitable(sfCreateSphere(offset + vec3(2.0, 0.0, 0.5), 0.5,
                                sfCreateMaterial(Blue, White, 0.0)));
    sfAddHitable(sfCreateSphere(offset + vec3(-2.0, 0.0, 0.5), 0.5,
                                sfCreateMaterial(Red, White, 0.0)));

    float x =  0.0;
    float y = -0.5;
    float z =  0.0;
    float size = 1.0;
    float lightSize = 0.2;
    sfAddHitable(sfCreateRect(HITABLE_XZRECT,
                              offset+vec3(x,y+0.99*size,z),
                              vec3(-lightSize, 0.0, -lightSize),
                              vec3( lightSize, 0.0,  lightSize),
                              sfCreateEmissionMaterial(White)));
    sfAddHitable(sfCreateRect(HITABLE_XZRECT,
                              offset+vec3(x,y+size,z),
                              0.5*vec3(-size, 0.0, -size),
                              0.5*vec3( size, 0.0,  size),
                              sfCreateDiffuseMaterial(White, 1.0)));
    sfAddHitable(sfCreateRect(HITABLE_XZRECT,
                              offset+vec3(x,y,z),
                              0.5*vec3(-size, 0.0, -size),
                              0.5*vec3( size, 0.0,  size),
                              sfCreateDiffuseMaterial(White, 1.0)));
    sfAddHitable(sfCreateRect(HITABLE_YZRECT,
                              offset+vec3(x+size*0.5,y+size*0.5,z),
                              0.5*vec3(0.5, -size, -size),
                              0.5*vec3(0.5,  size,  size),
                              sfCreateDiffuseMaterial(Green, 1.0)));
    sfAddHitable(sfCreateRect(HITABLE_YZRECT,
                              offset+vec3(x-size*0.5,y+size*0.5,z),
                              0.5*vec3(0.5, -size, -size),
                              0.5*vec3(0.5,  size,  size),
                              sfCreateDiffuseMaterial(Red, 1.0)));
    sfAddHitable(sfCreateRect(HITABLE_XYRECT,
                              offset+vec3(x,y+size*0.5,z-size*0.5),
                              0.5*vec3(-size, -size, 0.0),
                              0.5*vec3( size,  size, 0.0),
                              sfCreateDiffuseMaterial(White, 1.0)));
    */
    
    //sfAddHitable(sfCreateRect(HITABLE_XYRECT, offset+vec3(0.0,0.0,0.0), vec3(-0.5, -0.5,  0.0), vec3(0.5, 0.5, 0.0), sfCreateMaterial(Red, White, 1.0)));//sfCreateEmissionMaterial(White)));
    //sfAddHitable(sfCreateRect(HITABLE_XZRECT, offset+vec3(0.0,0.5,0.5), vec3(-0.5,  0.0, -0.5), vec3(0.5, 0.0, 0.5), sfCreateMaterial(Green, White, 1.0)));
    //sfAddHitable(sfCreateRect(HITABLE_YZRECT, offset+vec3(0.5,0.0,0.0), vec3( 0.0, -0.5, -0.5), vec3(0.0, 0.5, 0.5), sfCreateMaterial(Blue, White, 1.0)));
    
    //sfAddHitable(sfCreateBox(offset, -OneHalf, OneHalf, sfCreateMaterial(White, White, 1.0)));
    //sfAddHitable(sfCreateSphere(offset + vec3(0.0, 4.0, 0.5), 0.25, sfCreateEmissionMaterial(White)));
    //sfAddHitable(sfCreateCylinder(offset + vec3(1.0, osc(0.5, -0.5), 0.0), 0.5, 1.0, sfCreateMaterial(ArneBlue, White, 0.0)));
    //sfAddHitable(sfCreateCone(offset + vec3(-1.0, -0.5, 0.0), 0.25, 1.0, sfCreateMaterial(ArneBlue, White, 0.0)));
    //sfAddHitable(sfCreateDisk(offset + Zero, vec3(0.0, 0.5, 1.0), 0.25, sfCreateMaterial(Purple, White, 0.0)));
    //sfAddLight(sfCreateLight(LIGHT_DIRECTION, Zero, vec3(1.0, 1.0, 1.0), 8.0*White));

    sfAddHitable(sfCreateTorus(offset - vec3(2.0,0.0,0.0), 0.5, 0.25, sfCreateDiffuseMaterial(Green, 1.0)));
	sfAddHitable(sfCreateSuperquadric(offset - vec3(0.5,0.0,0.0), 0.5, 2.0*sin(iTime) + 2.5, cos(iTime/3.14) + 1.5, sfCreateDiffuseMaterial(Blue, 1.0)));

    sfAddHitable(sfCreateMesh(offset + Right, sfCreateDiffuseMaterial(White * 0.25, 1.0)));
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    srand(fragCoord.xy / iResolution.xy);
    seed += iTime;
    CreateScene();
    Ray cameraRay = sfCreateCameraRay(0.5 * fragCoord.xy / iResolution.xy + 0.25);
   	fragColor = vec4(Sunfish(cameraRay), 1.0);
	// vec2 uv = fragCoord.xy / iResolution.xy;
	// fragColor = vec4(uv,0.5+0.5*sin(iTime),1.0);
}