
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <glad/glad.h>
#include <glfw3.h>
#define canvasWidthPx 1000 
#define canvasHeightPx 1000
#define xMax (100.0F)
#define xMin (-100.0F)
#define yMax (100.0F)
#define yMin (-100.0F)
#define xDiff xMax-xMin
#define yDiff yMax-yMin
#define numThreads 256
#define numBlocks 200

#define false 0
#define true 1
#define positive 1

#define realtime

typedef struct {
    void* data;
    void* output_stream;
}task;

typedef struct {
    unsigned char8744 r;
    unsigned char g;
    unsigned char b;
}color;

typedef struct {
    color c;
    unsigned char roughness;
    unsigned char brightness;
    unsigned char transparency;
    unsigned char refrac_coef;
}material;

typedef struct {
    float x;
    float y;
    float z;
    uint8_t null;
}vec3;

typedef struct {
    vec3 origin;
    vec3 vector;
    material material_touched;
    unsigned char num_bounces;
    unsigned char in_medium;
    unsigned char prev_refrac_coef;
}ray;

typedef struct {
    vec3 vertice_a;
    vec3 vertice_b;
    vec3 vertice_c;
    material mat;
    unsigned char planar;
}triangle;

typedef struct {
    vec3 center;
    float radius;
    material mat;
}sphere;

#define num_triangles 1
__device__ triangle triangles[num_triangles];
int num_t;
__constant__ sphere* spheres;
__constant__ int num_spheres;
int num_s;
__device__ vec3 normal_vectors[num_triangles];
__device__ int f;
#ifndef realtime
__device__ color cur_color[numThreads * numBlocks];
#endif
#ifndef quality
__device__ color cur_color[canvasWidthPx * canvasHeightPx];
#endif

void add_triangles(triangle ts[num_triangles]) {
    num_t = num_triangles;
    triangle t = ts[0];
    t.vertice_a.x = 0.0;
    cudaError_t e = cudaMemcpyToSymbol(triangles, ts, sizeof(triangle) * num_triangles);
    cudaMemcpyFromSymbol(&t, triangles, sizeof(triangle));
}

void add_spheres(sphere* s, int ns) {
    num_s = ns;
    cudaMalloc(&spheres, sizeof(sphere) * ns);
    cudaMemcpyToSymbol(spheres, s, num_s * sizeof(sphere), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(&num_spheres, &ns, sizeof(int), cudaMemcpyHostToDevice);
}

ray get_ray_CPU(int x_ind, int y_ind) {
    ray ret;
    ret.origin.x = (float)x_ind;
    ret.origin.y = (float)y_ind;
    ret.origin.z = 0.0F;
    ret.vector.x = x_ind * 0.1F;
    ret.vector.y = y_ind * 0.1F;
    ret.vector.z = 1.0F;
    return ret;
}

__device__ float dist_between_vec(vec3 v1, vec3 v2) {
    return sqrt(pow(v2.z - v1.z, 2) + pow(v2.y - v1.y, 2) + pow(v2.x - v1.x, 2));
}

__device__ float fast_invsqrt(float f) {
    float th = 1.5F;
    float h = f * 0.5F;
    long temp = *((long*)&f);
    temp = 0x5f3759df - (temp >> 1);
    float temp2 = *((float*)&temp);
    temp2 = temp2 * (th - (h * temp2 * temp2));
    return temp2;
}

float fast_invsqrt_CPU(float f) {
    float th = 1.5F;
    float h = f * 0.5F;
    long temp = *((long*)&f);
    temp = 0x5f3759df - (temp >> 1);
    float temp2 = *((float*)&temp);
    temp2 = temp2 * (th - (h * temp2 * temp2));
    return temp2;
}

__device__ int return_sign(float d) {
    if (d == 0.0) {
        return 0;
    }
    return (d > 0.0F) ? 1 : -1;
}

__device__  float rad_to_deg(float r) {
    return r * 180.0 * 7.0 / 22.0;
}

__device__ vec3 flip_vec(vec3 v) {
    v.x = -1.0 * v.x;
    v.y = -1.0 * v.y;
    v.z = -1.0 * v.z;
    return v;
}

__device__ vec3 subtract_3Dvectors_result(vec3 v1, vec3 v2) {
    vec3 ret;
    ret.x = v1.x - v2.x;
    ret.y = v1.y - v2.y;
    ret.z = v1.z - v2.z;
    return ret;
}

vec3 subtract_3Dvectors_result_CPU(vec3 v1, vec3 v2) {
    vec3 ret;
    ret.x = v1.x - v2.x;
    ret.y = v1.y - v2.y;
    ret.z = v1.z - v2.z;
    return ret;
}

__device__ vec3 add_vectors_ret(vec3 v1, vec3 v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}

__device__ vec3 multiply_3Dvector_by_num_ret(vec3 v, float m) {
    v.x *= m;
    v.y *= m;
    v.z *= m;
    return v;
}

__device__ vec3 multiply_3D_vector_by_vec_ret(vec3 v, vec3 v2) {
    v.x *= v2.x;
    v.y *= v2.y;
    v.z *= v2.z;
    return v;
}

__device__ float matrix2D(float d1, float d2, float d21, float d22) {
    return d1 * d22 - d2 * d21;
}

float matrix2D_CPU(float d1, float d2, float d21, float d22) {
    return d1 * d22 - d2 * d21;
}

__device__ vec3 cross(vec3 a, vec3 b) {
    vec3 ret;
    ret.x = matrix2D(a.y, a.z, b.y, b.z);
    ret.y = matrix2D(a.x, a.z, b.x, b.z);
    ret.z = matrix2D(a.x, a.y, b.x, b.y);
    ret.null = false;
    return ret;
}

vec3 cross_CPU(vec3 a, vec3 b) {
    vec3 ret;
    ret.x = matrix2D_CPU(a.y, a.z, b.y, b.z);
    ret.y = matrix2D_CPU(a.x, a.z, b.x, b.z);
    ret.z = matrix2D_CPU(a.x, a.y, b.x, b.y);
    ret.null = false;
    return ret;
}

__device__ float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float calc_magnitude(vec3 v) {
    return fast_invsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float calc_magnitude_CPU(vec3 v) {
    return fast_invsqrt_CPU(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ vec3 normalize_vector(vec3 v) {
    vec3 ret;
    float d = calc_magnitude(v);
    ret.x = v.x * d;
    ret.y = v.y * d;
    ret.z = v.z * d;
    ret.null = false;
    return ret;
}

vec3 normalize_vector_CPU(vec3 v) {
    vec3 ret;
    float d = calc_magnitude_CPU(v);
    ret.x = v.x * d;
    ret.y = v.y * d;
    ret.z = v.z * d;
    ret.null = false;
    return ret;
}

__device__ float angle_between_vectors(vec3 v1, vec3 v2) {
    float t = dot(v1, v2) * (calc_magnitude(v2) * calc_magnitude(v1));
    return rad_to_deg(acos(t));
}

__device__ material init_material(int r, int g, int b, int br, int t, int rh, unsigned char refrac) {
    material ret;
    ret.c.r = r;
    ret.c.g = g;
    ret.c.b = b;
    ret.brightness = br;
    ret.transparency = t;
    ret.roughness = rh;
    ret.refrac_coef = refrac;
    return ret;
}

__device__ vec3 init_vec3(float x, float y, float z) {
    vec3 ret;
    ret.x = x;
    ret.y = y;
    ret.z = z;
    return ret;
}

vec3 init_vec3_CPU(float x, float y, float z) {
    vec3 ret;
    ret.x = x;
    ret.y = y;
    ret.z = z;
    return ret;
}


__device__ ray init_ray(float ox, float oy, float oz, float vx, float vy, float vz) {
    ray ret;
    ret.origin.x = ox;
    ret.origin.y = oy;
    ret.origin.z = oz;
    ret.vector.x = vx;
    ret.vector.y = vy;
    ret.vector.z = vz;
    return ret;
}

__device__ void set_vec(vec3* v, float x, float y, float z) {
    v->x = x;
    v->y = y;
    v->z = z;
}

__device__ int point_in_triangle(vec3 p, triangle t) {
    t.vertice_a = subtract_3Dvectors_result(t.vertice_a, p);
    t.vertice_b = subtract_3Dvectors_result(t.vertice_b, p);
    t.vertice_c = subtract_3Dvectors_result(t.vertice_c, p);
    vec3 c1 = cross(t.vertice_b, t.vertice_a);
    vec3 c2 = cross(t.vertice_a, t.vertice_c);
    vec3 c3 = cross(t.vertice_c, t.vertice_b);
    if (dot(c1, c2) <= 0.0F) {
        return false;
    }
    if (dot(c1, c3) <= 0.0F) {
        return false;
    }
    return true;
}

__device__ uint8_t same_dir(vec3 v1, vec3 v2) {
    return return_sign(v1.x) == return_sign(v2.x) && return_sign(v1.y) == return_sign(v2.y) && return_sign(v1.z) == return_sign(v2.z);
}

__device__ uint8_t point_on_ray(ray r, vec3 p) {
    vec3 vec_rayp = subtract_3Dvectors_result(p, r.origin);
    if ((fabs(subtract_3Dvectors_result(r.origin, p).x) + fabs(subtract_3Dvectors_result(r.origin, p).y) + fabs(subtract_3Dvectors_result(r.origin, p).z)) > 0.1F) {
        return same_dir(r.vector, vec_rayp);
    }
    return false;
}

__device__ vec3 get_norm(triangle t) {
    vec3 ab = subtract_3Dvectors_result(t.vertice_a, t.vertice_b);
    vec3 bc = subtract_3Dvectors_result(t.vertice_b, t.vertice_c);
    vec3 normalized_vector = cross(ab, bc);
    return normalize_vector(normalized_vector);
}

vec3 get_norm_CPU(triangle t) {
    vec3 ab = subtract_3Dvectors_result_CPU(t.vertice_a, t.vertice_b);
    vec3 bc = subtract_3Dvectors_result_CPU(t.vertice_b, t.vertice_c);
    vec3 normalized_vector = cross_CPU(ab, bc);
    return normalize_vector_CPU(normalized_vector);
}

void get_norms() {
    vec3 nor[num_triangles];
    triangle temp_t[num_triangles];
    cudaMemcpyFromSymbol(temp_t, triangles, sizeof(triangle) * num_triangles);
    for (int n = 0; n < num_triangles; n++) {
        nor[n] = get_norm_CPU(temp_t[n]);
    }
    cudaMemcpyToSymbol(normal_vectors, nor, sizeof(vec3) * num_triangles);
}

__device__ vec3 triangle_ray_intersection(ray r, triangle t, vec3 norm_v) {
    vec3 ret;
    ret.null = false;

    // prvo parametrizujemo vektor

    /* parametrizovan vektor:
     x = ox + t * xdist
     y = oy + t * ydist
     z = oz + t * zdist
    */

    //normalizovan vektor ravne:
    // A-B x C-B
    //print_3Dvector(bc);

    /*tu se izracuna cross
      i = (1, 0, 0)
      j = (0, 1, 0)
      k = (0, 0, 1)

      matrica(racunamo discriminant):
      i    j    k
      abx  aby  abz
      bcx  bcy  bcz

      uprostena matrica:

      i * ((aby*bcz) - (bcy * abz)) - j * ((abx*bcz) - (bcx * abz)) + k * ((abx*bcy) - (bcx * aby))
    */
    vec3 normalized_vector = norm_v;
    /*normalized_vector = nv
    a je tacka na ravni
    jednacina ravne: nvx * (x-ax) + nvy* (y-ay) + nvz*(z-az) = 0
    */
    /* zameni vektor (x, y, z) sa (ox+t*xdist, oy+t*ydist, oz+t*zdist)
    resi za t
    nvx*(ox+t*xdist-ax)+nvy*(oy+t*ydist-ay)+nvz*(oz+t*zdist-az) = 0
    nvx*ox-nvx*ax+nvy*oy-nvy*ay+nvz*oz-nvz*az = 0
    nvx*t*xdist+nvy*t*ydist+nvz*t*zdist=nvx*ox-nvx*ax+nvy*oy-nvy*ay+nvz*oz-nvz*az
    t*(nvx*xdist+nvy*ydist+nvz*zdist) = nvx*ox-nvx*ax+nvy*oy-nvy*ay+nvz*oz-nvz*az
    t = (nvx*ox-nvx*ax+nvy*oy-nvy*ay+nvz*oz-nvz*az)/(nvx*xdist+nvy*ydist+nvz*zdist)
    */
    normalized_vector = normalize_vector(normalized_vector);
    float tmp_f = dot(r.vector, normalized_vector);
    if (tmp_f == 0.0F) {
        ret.null = true;
        return ret;
    }
    float p_t = dot(subtract_3Dvectors_result(t.vertice_a, r.origin), normalized_vector) / tmp_f;
    set_vec(&ret, r.origin.x + (r.vector.x * p_t), r.origin.y + (r.vector.y * p_t), r.origin.z + (r.vector.z * p_t));
    if (t.planar) { goto skptri; }
    if (!point_in_triangle(ret, t)) {
        ret.null = true;
        return ret;
    }
skptri:
    ret.null = false;
    return ret;
}

__device__ ray bounce_ray(ray r, triangle t, vec3 nv, curandState s) {
    ray ret;
    vec3 intersect = triangle_ray_intersection(r, t, nv);
    if (intersect.null) {
        ret.origin.null = true;
        ret.vector.null = true;
        return ret;
    }
    if (!point_on_ray(r, intersect)) {
        ret.origin.null = true;
        ret.vector.null = true;
        return ret;
    }
    vec3 normalized_line_of_reflection = normalize_vector(nv);
    float temp = dot(normalized_line_of_reflection, r.vector) * (calc_magnitude(r.vector) * calc_magnitude(normalized_line_of_reflection));
    float angle_between_ray_and_nlr = rad_to_deg(acos(temp));
    //printf("%f\n", calc_magnitude(normal_vec));
    if (angle_between_ray_and_nlr > 90.0) {
        normalized_line_of_reflection = flip_vec(normalized_line_of_reflection);
    }
    vec3 rand_vec;
    rand_vec.x = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
    rand_vec.y = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
    rand_vec.z = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
    float a1 = angle_between_vectors(rand_vec, normalized_line_of_reflection);
    float a2 = angle_between_vectors(normalized_line_of_reflection, rand_vec);
    float sa = (a1 < a2) ? a1 : a2;


    while (true) {
        rand_vec.x = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
        rand_vec.y = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
        rand_vec.z = (curand_uniform(&s) * 100 / 99.0 - 0.5) * 2.0;
        a1 = angle_between_vectors(rand_vec, normalized_line_of_reflection);
        a2 = angle_between_vectors(normalized_line_of_reflection, rand_vec);
        sa = (a1 < a2) ? a1 : a2;
        if (sa < 90.0) {
            break;
        }
    }
    //w = v - 2 * (v ∙ n) * n
    float dt = dot(normalized_line_of_reflection, normalized_line_of_reflection);
    vec3 reflected_vec = subtract_3Dvectors_result(r.vector, multiply_3Dvector_by_num_ret(normalized_line_of_reflection, dt * 2));
    reflected_vec = add_vectors_ret(multiply_3Dvector_by_num_ret(reflected_vec, (1.0F - t.mat.roughness / 255.0F)), multiply_3Dvector_by_num_ret(rand_vec, (t.mat.roughness / 255.0F)));
    if (intersect.null == false) {
        ret.vector = reflected_vec;
        ret.origin = intersect;
        ret.vector.null = false;
        //ret.origin.null = false;
        ret.material_touched = t.mat;
        return ret;
    }
    ret.origin.null = true;
    ret.vector.null = true;
    return ret;
}


__device__ ray bounce_ray_on_t(ray r, triangle t, vec3 nv, curandState s) {
    ray ret;
    ret.material_touched = init_material(0, 0, 0, 0, 0, 0, 0);
    nv = normalize_vector(nv);
    r.vector = normalize_vector(r.vector);
    vec3 intersect = triangle_ray_intersection(r, t, nv);
    if (intersect.null || !point_on_ray(r, intersect)) {
        ret.origin.null = 1;
        ret.vector.null = 1;
        return ret;
    }
    vec3 normalized_line_of_reflection = normalize_vector(nv);
    float angle_btw = angle_between_vectors(normalized_line_of_reflection, r.vector);
    if (angle_btw < 90.0F) {
        normalized_line_of_reflection = flip_vec(normalized_line_of_reflection);
    }
    float projected_t = dot(normalized_line_of_reflection, r.vector);
    vec3 reflected_vec = subtract_3Dvectors_result(multiply_3Dvector_by_num_ret(nv, projected_t * 2.0F), r.vector);
    ret.origin = intersect;
    ret.vector = reflected_vec;
    ret.material_touched = t.mat;
    return ret;
}






/*
__device__ vec3 sphere_intersect(ray ra, sphere cs) {
    vec3 ret;
    ret.null = false;
    *jednacina zraka/vektora: f(t) = origin + t*vec
     * deo x: fx(t)= originx + vecx*t
     * deo y: fy(t)= originy + vecy*t
     * deo z: fz(t)= originz + vecz*t
     * dist formula kruga: sqrt((pointx-centerx)^2 + (pointy-centery)^2 + (pointz-centerz)^2) = d
     * zameni point sa (fx, fy, fz)
     * d = r^2
     * sqrt((originx+vecx*t-centerx)^2 + (originy+vecy*t-centery)^2 + (originz+vecz*t-centerz)^2) = r^2
     * sqrt(originx^2 + originx*vecx*t - originx*centerx + vecx*t*originx + (vecx*t)^2 - vecx*t*centerx - centerx*originx - centerx*vecx*t + centerx^2 + originy^2 + originy*vecy*t - originy*centery + vecy*t*originy + (vecy*t)^2 - vecy*t*centery - centery*originy - centery*vecy*t + centery^2 + originz^2 + originz*vecz*t - originz*centerz + vecz*t*originz + (vecz*t)^2 - vecz*t*centerz - centerz*originz - centerz*vecz*t + centerz^2) = r^2
     * t terms: originx^2-originx*centerx-centerx*originx+centerx^2+originy^2-originy*centery-centery*originy+centery^2+originz^2-originz*centerz-centerz*originz+centerz^2
     * originx*vecx*t + vecx*t*origin + vecx^2*t^2 - vecx*t*centerx - centerx*vecx*t +originy*vecy*t + vecy*t*originy + vecy^2*t^2 - vecy*t*centery - cecentery*vecy*t + originz*vecz*t+vecz*t*originz + vecz^2*t^2 - vecz*t*centerz - centerz*vecz*t
     *
     * -1*(originx^2-originx*centerx-centerx*originx+centerx^2+originy^2-originy*centery-centery*originy+centery^2+originz^2-originz*centerz-centerz*originz+centerz^2) + r^4 = originx*vecx*t + vecx*t*origin + vecx^2*t^2 - vecx*t*centerx - centerx*vecx*t +originy*vecy*t + vecy*t*originy + vecy^2*t^2 - vecy*t*centery - cecentery*vecy*t + originz*vecz*t+vecz*t*originz + vecz^2*t^2 - vecz*t*centerz - centerz*vecz*t
     * (-1*(originx^2-originx*centerx-centerx*originx+centerx^2+originy^2-originy*centery-centery*originy+centery^2+originz^2-originz*centerz-centerz*originz+centerz^2) + r^4)/t =
     * a = vecx^2 + vecy^2 + vecz^2
     * b = originx*vecx+originy*vecy + originz*vecz - 2*vecx*centerx - 2*vecy*centery - 2*vecz*centerz
     * c = originx*originx - 2*originx*centerx+centerx*centerx+originy*originy-2*originy*centery+centery*centery+originz*originz-2*originz*centerz+centerz*centerz+r*r*r*r
     * t = -1*b+-(sqrt(b*b-4*a*c)/2*a)
    *
    float cx = cs.center.x;
    float cy = cs.center.y;
    float cz = cs.center.z;

    float px = ra.origin.x;
    float py = ra.origin.y;
    float pz = ra.origin.z;

    float vx = ra.vector.x;
    float vy = ra.vector.y;
    float vz = ra.vector.z;

    float a = vx * vx + vy * vy + vz * vz;
    float b = 2.0 * (px * vx + py * vy + pz * vz - vx * cx - vy * cy - vz * cz);
    float c = px * px - 2 * px * cx + cx * cx + py * py - 2 * py * cy + cy * cy +
        pz * pz - 2 * pz * cz + cz * cz - cs.radius * cs.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) {
        ret.null = true;
        return ret;
    }
    float sq = sqrtf(disc);
    float t1 = (-1 * b + sq) / (2 * a);
    if (disc == 0) {
        ret.x = ra.origin.x + ra.vector.x * t1;
        ret.y = ra.origin.y + ra.vector.y * t1;
        ret.z = ra.origin.z + ra.vector.z * t1;
    }
    float t2 = (-1 * b - sq) / (2 * a);
    ret.x = ra.origin.x + ra.vector.x * ((t1 < t2) ? t1 : t2);
    ret.y = ra.origin.y + ra.vector.y * ((t1 < t2) ? t1 : t2);
    ret.z = ra.origin.z + ra.vector.z * ((t1 < t2) ? t1 : t2);
    return ret;
}

__device__ ray bounce_ray_sphere(ray r, sphere s, vec3* nv) {
    ray ret;
    vec3 intersect = sphere_intersect(r, s);
    if (intersect.null || !point_on_ray(r, intersect)) {
        ret.origin.null = true;
        ret.vector.null = true;
        return ret;
    }
    vec3 normal_vec = subtract_3Dvectors_result(intersect, s.center);
    normal_vec.null = false;
    vec3 normalized_line_of_reflection = flip_vec(normalize_vector(normal_vec));
    *nv = normalized_line_of_reflection;
    vec3 rand_vec;
    float a1;
    float a2;
    float sa;
    while (true) {
        rand_vec.x = ((rand() % 100) / 99.0 - 0.5) * 2.0;
        rand_vec.y = ((rand() % 100) / 99.0 - 0.5) * 2.0;
        rand_vec.z = ((rand() % 100) / 99.0 - 0.5) * 2.0;
        a1 = angle_between_vectors(rand_vec, normalized_line_of_reflection);
        a2 = angle_between_vectors(normalized_line_of_reflection, rand_vec);
        sa = (a1 < a2) ? a1 : a2;
        if (sa < 90.0) {
            break;
        }
    }
    float dt = dot(normalized_line_of_reflection, normalized_line_of_reflection);
    vec3 reflected_vec = subtract_3Dvectors_result(r.vector, multiply_3Dvector_by_num_ret(normalized_line_of_reflection, dt * 2));
    reflected_vec = add_vectors_ret(multiply_3Dvector_by_num_ret(reflected_vec, (1.0F - s.mat.roughness / 255.0F)), multiply_3Dvector_by_num_ret(rand_vec, (s.mat.roughness / 255.0F)));
    if (intersect.null == false) {
        ret.vector = reflected_vec;
        ret.origin = intersect;
        ret.vector.null = false;
        ret.material_touched = s.mat;
        return ret;
    }
    ret.origin.null = true;
    ret.vector.null = true;
    return ret;
}
*/

__device__ ray bounce_first_triangle(ray r, curandState s) {
    r.origin.null = 0;
    r.vector.null = 0;
    float best_dist = -1.0F;
    ray best_ray;
    for (int t_ind = 0; t_ind < num_triangles; t_ind++) {
        ray temp_bounce = bounce_ray_on_t(r, triangles[t_ind], normal_vectors[t_ind], s);
        if (temp_bounce.origin.null) { continue; }
        float new_dist = dist_between_vec(r.origin, temp_bounce.origin);
        if (best_dist == -1.0F || best_dist > new_dist) {
            best_dist = new_dist;
            best_ray = temp_bounce;
            r.origin.null = 0;
            r.vector.null = 0;
        }
    }
    if (best_dist != -1.0F) {
        return best_ray;
    }
    r.origin.null = 1;
    r.vector.null = 1;
    r.material_touched = init_material(0, 0, 0, 0, 0, 0, 0);
    return r;
}

__device__ int get_x_component(int ind) {
    return ind % canvasWidthPx - 500;
}

__device__ int get_y_component(int ind) {
    return ind / canvasWidthPx - 500;
}

__device__ color bounce_ray_repeatedly(ray r, curandState s, int num_bounces) {
    ray cur_bounce = bounce_first_triangle(r, s);
    color ret;
    ret.r = 0;
    ret.g = 0;
    ret.b = 0;
    if (cur_bounce.origin.null) {
        return ret;
    }
    ret.r += cur_bounce.material_touched.c.r;
    ret.g += cur_bounce.material_touched.c.g;
    ret.b += cur_bounce.material_touched.c.b;
    int max_brightness = cur_bounce.material_touched.brightness;
    int b = 1;
    for (; b < num_bounces; b++) {
        cur_bounce = bounce_first_triangle(cur_bounce, s);
        if (cur_bounce.origin.null) {
            break;
        }
        ret.r += cur_bounce.material_touched.c.r;
        ret.g += cur_bounce.material_touched.c.g;
        ret.b += cur_bounce.material_touched.c.b;
        max_brightness = cur_bounce.material_touched.brightness;
    }
    ret.r /= b;
    ret.g /= b;
    ret.b /= b;
    ret.r *= max_brightness / 255.0F;
    ret.g *= max_brightness / 255.0F;
    ret.b *= max_brightness / 255.0F;
    return ret;
}

__global__ void rtPrecisionKernel(ray r, int num_bounces)
{
    curandState state;
    int t_ind = threadIdx.x + blockIdx.x * numThreads;
    curand_init(1234, t_ind, 0, &state);
    cur_color[t_ind] = bounce_ray_repeatedly(r, state, num_bounces);

}

__global__ void rtRealtimeKernel(float fov, int num_bounces, int threads_in_repetition, int num_repetitions) {
    curandState state;
    unsigned int t_ind = threadIdx.x + blockIdx.x * numThreads + threads_in_repetition * num_repetitions;
    int px, py;
    px = t_ind % canvasWidthPx;
    py = t_ind / canvasWidthPx;
    curand_init(1234, t_ind, 0, &state);
    float x = xMin + xDiff * px / ((float)canvasWidthPx);
    float y = yMin + yDiff * py / ((float)canvasHeightPx);
    float vx = x * fov;
    float vy = y * fov;
    ray cur_r;
    cur_r.origin = init_vec3(x, y, 0.0F);
    cur_r.vector = init_vec3(vx, vy, 1.0F);
    cur_color[px + py * canvasWidthPx] = bounce_ray_repeatedly(cur_r, state, num_bounces);
}

void single_frameBuffer(int num_bounces, color* c, int maxThreads, int maxBlocks, float fov) {
    int b = canvasHeightPx * canvasWidthPx / maxThreads;
    int repetitions = 1;
    if ((b > maxBlocks)) {
        repetitions = b / maxBlocks;
        repetitions += (b / maxBlocks * maxBlocks) != b ? 1 : 0;
    }
    b /= repetitions;
    printf("%d\n", repetitions);
    for (int r = 0; r < repetitions; r++) {
        rtRealtimeKernel<<<maxThreads, b>>>(fov, num_bounces, b * maxThreads, r);
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        }
    }
    cudaMemcpyFromSymbol(c, cur_color, sizeof(color) * canvasHeightPx * canvasWidthPx);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
unsigned int SCR_WIDTH = canvasWidthPx;
unsigned int SCR_HEIGHT = canvasHeightPx;


char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aColor;\n"
"layout(location = 2) in vec2 aTexCoord;\n"
"out vec3 ourColor;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   ourColor = aColor;\n"
"   TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
"}\0";

char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D texture1;\n"
"void main()\n"
"{\n"
"   FragColor = texture(texture1, TexCoord);\n"
"}\n\0";

int main()
{
    triangle triangles_CPU[1];
    triangles_CPU[0].vertice_a = init_vec3_CPU(-200.0, -150.0, 1.0);
    triangles_CPU[0].vertice_b = init_vec3_CPU(-100.0, -150.0, 1.0);
    triangles_CPU[0].vertice_c = init_vec3_CPU(-150.0, -100.0, 1.0);
    triangles_CPU[0].mat.brightness = 255;
    triangles_CPU[0].mat.c.r = 9;
    triangles_CPU[0].mat.c.g = 255;
    triangles_CPU[0].mat.c.b = 255;
    triangles_CPU[0].mat.transparency = 0;
    triangles_CPU[0].mat.roughness = 0;
    triangles_CPU[0].mat.refrac_coef = 255;
    triangles_CPU[0].planar = 0;
    add_triangles(triangles_CPU);
    get_norms();
    cudaDeviceSynchronize();
    color* c = (color*)malloc(sizeof(color) * canvasHeightPx * canvasWidthPx);
    printf("%s\n", "start");
    single_frameBuffer(1, c, numThreads, numBlocks, 0.0F);
    printf("%s\n", "end");
    for (int i = 900000; i < 900100; i++) {
        printf("%u\n", c[i].r);
    }
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        return -1;
    }
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::FRAGMENT::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------

    // glad: load all OpenGL function pointers
    // ---------------------------------------


    // build and compile our shader zprogram
    // ------------------------------------

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // load and create a texture 
    // -------------------------
    unsigned int texture1;
    uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * canvasHeightPx * canvasWidthPx * 3);
    for (int p = 0; p < canvasHeightPx * canvasWidthPx * 3; p += 3) {
        color tempc = c[p / 3];
        pixels[p] = tempc.r;
        pixels[p + 1] = tempc.g;
        pixels[p + 2] = tempc.b;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //printf("%s\n", "done");
    // texture 1
    // ---------
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    glUseProgram(shaderProgram); // don't forget to activate/use the shader before setting uniforms!
    // either set it manually like so:
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);

        // render container
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    
    /*for (int r = 0; r < canvasHeightPx; r++) {
        for (int c = 0; c < canvasWidthPx; c++) {
            float x = xMin + xDiff * r / ((float)canvasWidthPx);
            float y = yMin + yDiff * c / ((float)canvasHeightPx);
            float vx = x * fov;
            float vy = y * fov;
            ray cur_r;
            cur_r.origin = init_vec3_CPU(x, y, 0.0F);
            cur_r.vector = init_vec3_CPU(vx, vy, 1.0F);
            rtPrecisionKernel << <numBlocks, numThreads >> > (cur_r, 5);
            color c2[numThreads * numBlocks];
            cudaError_t e = cudaMemcpyFromSymbol(c2, cur_color, sizeof(color) * numThreads * numBlocks);
            long int avgr, avgg, avgb;
            avgr = 0;
            avgg = 0;
            avgb = 0;
            for (int ci = 0; ci < numThreads * numBlocks; ci++) {
                avgr += c2[ci].r;
                avgg += c2[ci].g;
                avgb += c2[ci].b;
            }
            color avg_color;
            avg_color.r = avgr / (numThreads * numBlocks);
            avg_color.g = avgg / (numThreads * numBlocks);
            avg_color.b = avgb / (numThreads * numBlocks);
        }
        printf("%d\n", r);  
    }*/




    return 0;
}
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}