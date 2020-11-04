// Copyright (c) 2015-2020 Jonathan Metzgar
//
// Sunfish is a Monte Carlo path tracer for physically based rendering in as
// simple a way possible. To that end the following features are planned:
//
// * Ray Generation, Ray Intersection, and custom shaders
//   * Intersection
//   * Closest hit
//   * Any hit
//   * Miss
// * Ray-Sphere, Ray-Box, and Ray-Triangle intersections
// * Signed distance functions
// * Custom number of samples per pixel
// * Custom camera to support DOF
//
// The design of this software is to have an OpenGL based via GLFW viewer showing
// the current progress of the rendering.
////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <iostream>
#include <fluxions_stdcxx.hpp>
#include <fluxions_gte.hpp>
#include <fluxions_gte_image.hpp>
#include <hatchetfish.hpp>
#include <fluxions_ssg_environment.hpp>
#include <fluxions_ssg.hpp>
//#include <viperfish_utilities.hpp>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "damselfish.lib")		// used by fluxions for the rendering system
#pragma comment(lib, "hatchetfish.lib")		// used by all for debug messages
#pragma comment(lib, "starfish.lib")		// used for astronomy calculations
#pragma comment(lib, "fluxions-gte.lib")	// used by all for math calculations
#pragma comment(lib, "fluxions-base.lib")	// used for image loading and opengl drawing
#pragma comment(lib, "fluxions.lib")		// used for simplescenegraph and rendering

using namespace std;
using namespace Fluxions;


//////////////////////////////////////////////////////////////////////
// RandomLUT /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


class RandomLUT {
public:
	RandomLUT();
	RandomLUT(int size);

	void init(int size);
	void seed(int seed);

	float frand();
	double drand();
	int irand();

	int size;
	int curIndex;
	vector<float> frandom;
	vector<double> drandom;
	vector<int> irandom;
private:
	float _frand(float min0, float max1);
	double _drand(double min0, double max1);
	int _irand(int min0, int max1);

	mt19937 _frand_s;
	mt19937 _drand_s;
	mt19937 _irand_s;
};


RandomLUT::RandomLUT() {
	init(32768);
}


RandomLUT::RandomLUT(int size) {
	init(size);
}


void RandomLUT::init(int size) {
	this->size = size;
	frandom.resize(size);
	drandom.resize(size);
	irandom.resize(size);

	for (int i = 0; i < size; i++) {
		frandom[i] = _frand(0.0f, 1.0f);
		drandom[i] = _drand(0.0f, 1.0f);
		irandom[i] = _irand(0, RAND_MAX);
	}

	curIndex = 0;
}


float RandomLUT::frand() {
	curIndex = (curIndex + 1);
	if (curIndex >= size) curIndex = 0;
	return frandom[curIndex];
}


double RandomLUT::drand() {
	curIndex = (curIndex + 1);
	if (curIndex >= size) curIndex = 0;

	return drandom[curIndex];
}


int RandomLUT::irand() {
	curIndex = (curIndex + 1);
	if (curIndex >= size) curIndex = 0;

	return irandom[curIndex];
}


float RandomLUT::_frand(float t0, float t1) {
	uniform_real_distribution<float> urd(t0, t1);
	return urd(_frand_s);
}


double RandomLUT::_drand(double t0, double t1) {
	uniform_real_distribution<double> urd(t0, t1);
	return urd(_drand_s);
}


int RandomLUT::_irand(int t0, int t1) {
	uniform_int_distribution<int> uid(t0, t1);
	return uid(_irand_s);
}


void RandomLUT::seed(int seed) {
	curIndex = seed % frandom.size();
}


//////////////////////////////////////////////////////////////////////
// Random Vectors ////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


RandomLUT RTrandom;


Vector3f getRandomUnitSphereVector() {
	Vector3f p;
	do {
		p = 2.0f * Vector3f(RTrandom.frand(), RTrandom.frand(), RTrandom.frand()) - Vector3f(1.0f);
	} while (dot(p, p) >= 1.0f);
	return p;
}


Vector3f getRandomUnitDiscVector() {
	Vector3f p;
	do {
		p = 2.0f * Vector3f(RTrandom.frand(), RTrandom.frand(), 0.0f) - Vector3f(1.0f, 1.0f, 0.0f);
	} while (dot(p, p) >= 1.0f);
	return p;
}


//////////////////////////////////////////////////////////////////////
// Utility Functions /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


Vector3f reflect(const Vector3f& v, const Vector3f& n) {
	return v - 2.0f * dot(v, n) * n;
}


bool refract(const Vector3f& v, const Vector3f& n, float ni_over_nt, Vector3f& refracted) {
	Vector3f uv = v.unit();
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
	if (discriminant > 0.0f) {
		refracted = ni_over_nt * (v - n * dt) - n * sqrt(discriminant);
		return true;
	}
	return false;
}


float schlick(float cosine, float F_0) {
	float r0 = (1.0f - F_0) / (1.0f + F_0);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}


//////////////////////////////////////////////////////////////////////
// Camera ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// A Camera is a type of ray generation because we generate rays
// from the source of the camera into the scene.
//
// TODO:
// [ ] Reframe as a subclass of Ray Generation class
//////////////////////////////////////////////////////////////////////


class Camera {
public:
	Camera();
	Camera(Vector3f eye, Vector3f center, Vector3f up, float yfovInDegrees, float aspectRatio, float znear, float zfar, float aperture = 2.0f, float distance_to_focus = 0.0f);

	void init(Vector3f eye, Vector3f center, Vector3f up, float yfovInDegrees, float aspectRatio, float znear, float zfar, float aperture = 2.0f, float distance_to_focus = 0.0f);
	void setLense(float aperture, float distance_to_focus);
	void setProjection(float fovyInDegrees, float aspectRatio, float znear, float zfar);
	void setLookAt(Vector3f eye, Vector3f center, Vector3f up);

	Rayf getRay(float u, float v);
	Rayf getRayDOF(float u, float v);

	// Projection Matrix
	float fovy;
	float aspectRatio;
	float znear;
	float zfar;
	Matrix4f ProjectionMatrix;
	Matrix4f InverseProjectionMatrix;

	// Camera Matrix
	Vector3f eye;
	Vector3f center;
	Vector3f up;
	Matrix4f ViewMatrix;
	Matrix4f InverseViewMatrix;

	Matrix4f ProjectionViewMatrix;
	Matrix4f InverseProjectionViewMatrix;

	// Ray casting parameters
	float aperture;
	float distance_to_focus;
	float lensRadius;

	Vector3f lowerLeftCorner;
	Vector3f horizontal;
	Vector3f vertical;
	Vector3f origin;
	Vector3f u;
	Vector3f v;
	Vector3f w;
private:
	void computeParameters_();
};


Camera::Camera() {
	init(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, -1.0f), Vector3f(0.0f, 1.0f, 0.0f), 90.0f, 2.0f, 0.001f, 100.0f);
	lensRadius = 1.0f;
	//lowerLeftCorner=Vector3f(-2.0, -1.0, -1.0);
	//horizontal=Vector3f(4.0, 0.0, 0.0);
	//vertical=Vector3f(0.0, 2.0, 0.0);
	//origin=Vector3f(0.0, 0.0, 0.0);
}


Camera::Camera(Vector3f eye, Vector3f center, Vector3f up, float yfovInDegrees, float aspectRatio, float znear, float zfar, float aperture, float distance_to_focus) {
	init(origin, center, up, yfovInDegrees, aspectRatio, znear, zfar);
	setLense(aperture, (center - eye).length());
}


void Camera::init(Vector3f eye, Vector3f center, Vector3f up, float yfovInDegrees, float aspectRatio, float znear, float zfar, float aperture, float distance_to_focus) {
	setProjection(yfovInDegrees, aspectRatio, znear, zfar);
	setLookAt(eye, center, up);
	setLense(aperture, distance_to_focus);
}


void Camera::setLense(float aperture, float distance_to_focus) {
	this->aperture = aperture;
	if (distance_to_focus <= 0.0f)
		this->distance_to_focus = (eye - center).length();
	else
		this->distance_to_focus = distance_to_focus;
}


void Camera::setProjection(float fovyInDegrees, float aspectRatio, float znear, float zfar) {
	this->fovy = fovyInDegrees;
	this->aspectRatio = aspectRatio;
	this->znear = znear;
	this->zfar = zfar;

	ProjectionMatrix.LoadIdentity();
	ProjectionMatrix.PerspectiveY(fovyInDegrees, aspectRatio, znear, zfar);

	InverseProjectionMatrix = ProjectionMatrix.AsInverse();

	ProjectionViewMatrix = ProjectionMatrix * ViewMatrix;
	InverseProjectionViewMatrix = ProjectionViewMatrix.AsInverse();

	computeParameters_();
}


void Camera::setLookAt(Vector3f eye, Vector3f center, Vector3f up) {
	this->eye = eye;
	this->center = center;
	this->up = up;
	this->distance_to_focus = (center - eye).length();

	ViewMatrix.LoadIdentity();
	ViewMatrix.LookAt(eye, center, up);
	InverseViewMatrix = ViewMatrix.AsInverse();
	ProjectionViewMatrix = ProjectionMatrix * ViewMatrix;
	InverseProjectionViewMatrix = ProjectionViewMatrix.AsInverse();

	computeParameters_();
}


Rayf Camera::getRay(float s, float t) {
	return Rayf(origin, lowerLeftCorner + s * horizontal + t * vertical - origin);
}


Rayf Camera::getRayDOF(float s, float t) {
	Vector3f rd = lensRadius * getRandomUnitDiscVector();
	Vector3f offset = u * rd.x + v * rd.y;
	return Rayf(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset);
}


void Camera::computeParameters_() {
	float theta = float(fovy * FX_DEGREES_TO_RADIANS);
	float halfHeight = tan(theta / 2.0f);
	float halfWidth = aspectRatio * halfHeight;
	origin = eye;
	w = (eye - center).unit();
	u = cross(up, w).unit();
	v = cross(w, u);
	horizontal = Vector3f(2.0f * distance_to_focus * halfWidth) * u;
	vertical = Vector3f(2.0f * distance_to_focus * halfHeight) * v;
	lowerLeftCorner = origin - (distance_to_focus * halfWidth) * u - (distance_to_focus * halfHeight) * v - distance_to_focus * w;
}


//////////////////////////////////////////////////////////////////////
// HitRecord /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// A HitRecord is used to fill in details about a ray-object
// intersection.
//////////////////////////////////////////////////////////////////////


class Material;

struct HitRecord {
	float t{ 0.0f };
	Vector3f p;
	Vector3f normal;

	Material* pmaterial{ nullptr };
};


//////////////////////////////////////////////////////////////////////
// RayTraceObject ////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// A RayTraceObject is used as a base class for an object that can
// be intersected by a ray. It supports the virtual methods:
// - closestHit
// - anyHit
//////////////////////////////////////////////////////////////////////


class RayTraceObject {
public:
	RayTraceObject() {}
	RayTraceObject(const string& name) : name(name) {}

	virtual bool closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
		rec.pmaterial = material;
		return false;
	}

	virtual bool anyHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
		return false;
	}

	string name;
	Material* material = nullptr;
};


//////////////////////////////////////////////////////////////////////
// InstancedRTO //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// An InstancedRTO implements the closestHit and anyHit methods from
// a RayTraceObject. It is initialized with a pointer to a different
// object.
//
// TODO:
// [ ] Allow separate material
//////////////////////////////////////////////////////////////////////


class InstancedRTO : public RayTraceObject {
public:
	InstancedRTO() : pRTO(nullptr) {}
	InstancedRTO(const string& name, RayTraceObject* rto)
		: RayTraceObject(name), pRTO(rto) {}

	virtual bool closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
		if (pRTO) pRTO->closestHit(r, tMin, tMax, rec);
		return false;
	}

	virtual bool anyHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
		if (pRTO) pRTO->anyHit(r, tMin, tMax, rec);
		return false;
	}

	RayTraceObject* pRTO;
};


//////////////////////////////////////////////////////////////////////
// RtoList ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// An RtoList is an object with several objects which is useful for
// representing a group or scene with a common transformation.
//////////////////////////////////////////////////////////////////////


class RtoList : RayTraceObject {
public:
	RtoList() {}
	RtoList(size_t numElements) { RTOs.resize(numElements); }
	~RtoList() {
		for (auto rto = RTOs.begin(); rto != RTOs.end(); rto++) {
			delete (*rto);
		}
		RTOs.clear();
	}

	virtual bool closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const;
	virtual bool anyHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const;

	vector<RayTraceObject*> RTOs;
};


bool RtoList::closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
	HitRecord tempRec;

	bool hitAnything = false;
	float closestHitT = tMax;

	for (auto rto = RTOs.begin(); rto != RTOs.end(); rto++) {
		if ((*rto)->closestHit(r, tMin, closestHitT, tempRec)) {
			hitAnything = true;
			closestHitT = tempRec.t;
			rec = tempRec;
		}
	}

	return hitAnything;
}


bool RtoList::anyHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
	HitRecord tempRec;

	for (auto rto = RTOs.begin(); rto != RTOs.end(); rto++) {
		if ((*rto)->anyHit(r, tMin, tMax, tempRec)) {
			return true;
		}
	}

	return false;
}


//////////////////////////////////////////////////////////////////////
// RtoSphere /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// An RtoSphere is a ray trace object representing a sphere.
//
// TODO:
// [ ] implement anyHit
//////////////////////////////////////////////////////////////////////


class RtoSphere : public RayTraceObject {
public:
	RtoSphere() {}
	RtoSphere(Vector3f _center, float _radius, Material* pmat)
		: center(_center), radius(_radius) {
		material = pmat;
	}

	virtual bool closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const;
	//virtual bool anyHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const;

	Vector3f center;
	float radius{ 1.0f };
};


bool RtoSphere::closestHit(const Rayf& r, float tMin, float tMax, HitRecord& rec) const {
	Vector3f oc = r.origin - center;
	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	float temp;
	if (discriminant > 0) {
		temp = (-b - sqrt(discriminant)) / a;
		if (temp < tMax && temp > tMin) {
			rec.t = temp;
			rec.p = r.getPointAtParameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.pmaterial = material;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < tMax && temp > tMin) {
			rec.t = temp;
			rec.p = r.getPointAtParameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.pmaterial = material;
			return true;
		}
	}
	return false;
}


//float hit_sphere(const Vector3f &center, float radius, const Rayf &r)
//{
//	Vector3f oc = r.origin - center;
//	float a = dot(r.direction, r.direction);
//	float b = 2.0f*dot(oc, r.direction);
//	float c = dot(oc, oc) - radius*radius;
//	float discriminant = b*b - 4 * a*c;
//	if (discriminant < 0)
//	{
//		return -1.0f;
//	}
//	else
//	{
//		return (-b - sqrt(discriminant)) / (2.0f*a);
//	}
//}


//////////////////////////////////////////////////////////////////////
// Material //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// A Material represents the custom appearance of an object. Child
// classes would override the following virtual methods:
// - shadeClosestHit
// - shadeAnyHit
// - shadeMissedHit
// - scatter
//
// The scatter function is used to spawn a ray in a random direction
// according to the material properties. For example, a smooth metal
// object might only spawn a ray in the reflection direction. A rough
// object might spawn a ray in a random direction in the hemisphere.
//
// TODO:
// [ ] Change virtual methods to pointers to objects or functions
//////////////////////////////////////////////////////////////////////


class Material {
public:
	Material() {}

	// default shader returns GREEN
	virtual Vector3f shadeClosestHit(const Rayf& r, const HitRecord& rec);
	// default shader returns RED
	virtual Vector3f shadeAnyHit(const Rayf& r, const HitRecord& rec);
	// default shader returns BLUE gradient
	virtual Vector3f shadeMissedHit(const Rayf& r, const SimpleEnvironment& environment);

	virtual bool scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const;

	Vector3f shadeShirleySky(const Rayf& r, const SimpleEnvironment& environment);
	Vector3f shadeHosekWilkieSky(const Rayf& r, const SimpleEnvironment& environment);
};


Vector3f Material::shadeClosestHit(const Rayf& r, const HitRecord& rec) {
	return Vector3f(0.0f, 1.0f, 0.0f);
}


Vector3f Material::shadeAnyHit(const Rayf& r, const HitRecord& rec) {
	return Vector3f(1.0f, 0.0f, 0.0f);
}


Vector3f Material::shadeMissedHit(const Rayf& r, const SimpleEnvironment& environment) {
	return shadeHosekWilkieSky(r, environment);
}


bool Material::scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const {
	return true;
}


Vector3f Material::shadeShirleySky(const Rayf& r, const SimpleEnvironment& environment) {
	// no hits, so return background color
	Vector3f unit_direction = r.direction.unit();
	float t = 0.5f * unit_direction.y + 1.0f;
	return (1.0f - t) * Vector3f(1.0f, 1.0f, 1.0f) + t * Vector3f(0.5f, 0.7f, 1.0f);
}


Vector3f Material::shadeHosekWilkieSky(const Rayf& r, const SimpleEnvironment& environment) {
	Color4f color = environment.getPixelCubeMap({ r.direction.x, max(0.0f, r.direction.y), r.direction.z });
	return color.ToVector3();
	//// no hits, so return background color
	//Vector3f unit_direction = r.direction.norm();
	//float t = 0.5f * unit_direction.y + 1.0f;
	//return (1.0f - t)*Vector3f(1.0f, 1.0f, 1.0f) + t*Vector3f(0.5f, 0.7f, 1.0f);
}


//////////////////////////////////////////////////////////////////////
// LambertianMaterial ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// LambertianMaterial is a child class of Material which 
//////////////////////////////////////////////////////////////////////


class LambertianMaterial : public Material {
public:
	LambertianMaterial(Vector3f color) : albedo(color) {}
	virtual bool scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const;

	Vector3f albedo;
};

bool LambertianMaterial::scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const {
	Vector3f scatterDir = rec.p + rec.normal + getRandomUnitSphereVector();
	scatteredRay = Rayf(rec.p, scatterDir - rec.p);
	attenuation = albedo;
	return true;
}


//////////////////////////////////////////////////////////////////////
// MetalMaterial /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// MetalMaterial is a child of Material and simulates a metal surface
// which may have a little roughness associated with it.
//////////////////////////////////////////////////////////////////////


class MetalMaterial : public Material {
public:
	MetalMaterial(Vector3f color, float f) : albedo(color) { fuzz = min(1.0f, f); }
	virtual bool scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const;

	Vector3f albedo;
	float fuzz;
};


bool MetalMaterial::scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const {
	Vector3f reflected = reflect(rayIn.direction.unit(), rec.normal);
	scatteredRay = Rayf(rec.p, reflected + fuzz * getRandomUnitSphereVector());
	attenuation = albedo;
	return (dot(scatteredRay.direction, rec.normal) > 0);
}


//////////////////////////////////////////////////////////////////////
// DielectricMaterial ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// DielectricMaterial is a child of Material which simulates a 
// dielectric translucent object. It uses Russian Roulette to refract
// or reflect a ray according to the Fresnel reflection.
//////////////////////////////////////////////////////////////////////


class DielectricMaterial : public Material {
public:
	DielectricMaterial() : F_0(1.5f) {}
	DielectricMaterial(float f) : F_0(f) {}
	virtual bool scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const;
	float F_0;
};


bool DielectricMaterial::scatter(const Rayf& rayIn, const HitRecord& rec, Vector3f& attenuation, Rayf& scatteredRay) const {
	Vector3f outwardNormal;
	Vector3f reflected = reflect(rayIn.direction, rec.normal);
	float ni_over_nt;
	attenuation = Vector3f(1.0f, 1.0f, 1.0f);
	Vector3f refracted;
	float reflectProb;
	float cosine;

	// inside or outside?
	if (dot(rayIn.direction, rec.normal) > 0) {
		outwardNormal = Vector3f(-rec.normal.x, -rec.normal.y, -rec.normal.z);
		ni_over_nt = F_0;
		cosine = F_0 * dot(rayIn.direction, rec.normal) / rayIn.direction.length();
	}
	else {
		outwardNormal = rec.normal;
		ni_over_nt = 1.0f / F_0;
		cosine = -dot(rayIn.direction, rec.normal) / rayIn.direction.length();
	}

	if (refract(rayIn.direction, outwardNormal, ni_over_nt, refracted)) {
		reflectProb = schlick(cosine, F_0);
	}
	else {
		scatteredRay = Rayf(rec.p, reflected);
		reflectProb = 1.0f;
	}

	// Russian roulette
	if (RTrandom.frand() < reflectProb) {
		scatteredRay = Rayf(rec.p, reflected);
	}
	else {
		scatteredRay = Rayf(rec.p, refracted);
	}
	return true;
}


//////////////////////////////////////////////////////////////////////
// NormalShadeMaterial ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// NormalShadeMaterial is a child of Material which implements a debug
// material to return the normal of the surface.
//////////////////////////////////////////////////////////////////////


class NormalShadeMaterial : public Material {
public:
	NormalShadeMaterial() {}

	virtual Vector3f shadeClosestHit(const Rayf& r, const HitRecord& rec) {
		return 0.5f * (1.0f + rec.normal);
	}
};


//Vector3f Trace(const Rayf &r, RtoList &world, int depth, const SimpleEnvironment &environment)
//{
//	static NormalShadeMaterial defaultMaterial;
//	HitRecord rec;
//
//	if (world.closestHit(r, 0.001f, FLT_MAX, rec))
//	{
//		Rayf scatteredRay;
//		Vector3f attenuation;
//		if (depth < 50 && rec.pmaterial->scatter(r, rec, attenuation, scatteredRay))
//		{
//			return attenuation.multiply(Trace(scatteredRay, world, depth + 1, environment));
//		}
//		else
//		{
//			return Vector3f(0.0f, 0.0f, 0.0f);
//		}
//	}
//	else
//	{
//		//if (rec.pmaterial) return rec.pmaterial->shadeMissedHit(r);
//
//		return defaultMaterial.shadeMissedHit(r, environment);
//	}
//}
//
//Vector3f color(const Rayf &r)
//{
//	float t = hit_sphere(Vector3f(0.0f, 0.0f, -1.0f), 0.5, r);
//	if (t > 0.0f)
//	{
//		Vector3f N = (r.getPointAtParameter(t) - Vector3f(0.0f, 0.0f, -1.0f));
//		return 0.5f*(1 + N);
//	}
//		
//	// no hits, so return background color
//	Vector3f unit_direction = r.direction.norm();
//	t = 0.5f * unit_direction.y + 1.0f;
//	return (1.0f - t)*Vector3f(1.0f, 1.0f, 1.0f) + t*Vector3f(0.5f, 0.7f, 1.0f);
//}


//////////////////////////////////////////////////////////////////////
// Scene /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Scene is a class to put all the objects, cameras, and lights in a
// collection.
// 
// TODO: Perhaps this would be better as a simple database
//////////////////////////////////////////////////////////////////////


class Scene {
public:
	Scene();
	~Scene();

	void addRTO(const string& name, RayTraceObject* rto);
	void addMaterial(const string& name, Material* material);
	void addInstance(const string& instanceName, const string& geometryName);
	Vector3f trace(const Rayf& r, int depth);
	void render();

	SimpleSceneGraph ssg;
	//SimpleEnvironment environment;
	Camera camera;
	RtoList world;
	map<string, RayTraceObject*> geometry;
	map<string, Material*> materials;
private:
	Material* pCurMtl{ nullptr };
};


Scene::Scene() {}


Scene::~Scene() {}


void Scene::addRTO(const string& name, RayTraceObject* rto) {
	if (geometry[name] != nullptr) {
		delete geometry[name];
		geometry[name] = nullptr;
	}
	rto->name = name;
	geometry[name] = rto;
}


void Scene::addMaterial(const string& name, Material* material) {
	materials[name] = material;
}


void Scene::addInstance(const string& instanceName, const string& geometryName) {
	world.RTOs.push_back(new InstancedRTO(instanceName, geometry[geometryName]));
}


Vector3f Scene::trace(const Rayf& r, int depth) {
	static NormalShadeMaterial defaultMaterial;
	HitRecord rec;

	if (world.closestHit(r, 0.001f, FLT_MAX, rec)) {
		Rayf scatteredRay;
		Vector3f attenuation;
		if (depth < 50 && rec.pmaterial->scatter(r, rec, attenuation, scatteredRay)) {
			return attenuation * trace(scatteredRay, depth + 1);
		}
		else {
			return Vector3f(0.0f, 0.0f, 0.0f);
		}
	}
	else {
		return defaultMaterial.shadeMissedHit(r, ssg.environment);
	}
}


void Scene::render() {

}


//////////////////////////////////////////////////////////////////////
// SceneConfiguration ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// SceneConfiguration stores the configuration of the path tracer.
// Perhaps this would be better if we renamed it SunfishConfig instead.
//////////////////////////////////////////////////////////////////////

class SceneConfiguration {
public:
	SceneConfiguration(int argc, const char** argv);
	~SceneConfiguration();

	void printHelp();

	unsigned imageWidth;
	unsigned imageHeight;
	float imageAspect{ 1.0f };
	bool isServer;
	bool isWorker;
	int numWorkers;
	int serverPort;
	string serverAddr;
	string inputFilename;
	string outputFilename;

	int workgroupSizeX;
	int workgroupSizeY;
	int workgroupSizeZ;
	int raysPerPixel;
	int samplesPerPixel;
	float jitterRadius;
	int rayDepth;

	float sun_turbidity;
	float sun_albedo[3];

private:
	bool getParameterf(int argc, const char** argv, int* i, const string& parameter, float* value);
	int getParameteri(int argc, const char** argv, int* i, const string& parameter);
	string getParameters(int argc, const char** argv, int* i, const string& parameter);
};


bool SceneConfiguration::getParameterf(int argc, const char** argv, int* i, const string& parameter, float* value) {
	if (*i < 0 || *i >= argc) return false;

	if (parameter == argv[*i]) {
		if (*i + 1 <= argc) {
			// peek next
			*value = (float)atof(argv[*i + 1]);
			(*i)++;
			return true;
		}
	}
	return false;
}


int SceneConfiguration::getParameteri(int argc, const char** argv, int* i, const string& parameter) {
	if (*i < 0 || *i >= argc) return 0;

	int value{ 0 };
	if (parameter == argv[*i]) {
		if (*i + 1 <= argc) {
			// peek next
			value = (int)atoi(argv[*i + 1]);
			// increment next parameter
			(*i)++;
		}
	}
	return value;
}


string SceneConfiguration::getParameters(int argc, const char** argv, int* i, const string& parameter) {
	if (*i < 0 || *i >= argc) return string();

	string value{ "" };
	if (parameter == argv[*i]) {
		if (*i + 1 <= argc) {
			// peek next
			value = argv[*i + 1];
			(*i)++;
		}
	}
	return value;
}


SceneConfiguration::SceneConfiguration(int argc, const char** argv) {
	imageWidth = 1280;
	imageHeight = 720;
	isServer = true;
	isWorker = true;
	numWorkers = 16;
	serverPort = 43316;
	serverAddr = "127.0.0.1";
	inputFilename = "input.scn";
	outputFilename = "output.ppm";

	workgroupSizeX = 128;
	workgroupSizeY = 128;
	workgroupSizeZ = 128;
	raysPerPixel = 1;
	samplesPerPixel = 1;
	jitterRadius = 1.0f;
	rayDepth = 16;

	sun_turbidity = 1.0f;

	// start from index 1 (ignoring the path of the executable)
	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], "-server", strlen(argv[i])) == 0) {
			isServer = true;
			isWorker = false;
		}

		if (strncmp(argv[i], "-worker", strlen(argv[i])) == 0) {
			isServer = false;
			isWorker = true;
		}

		if (strncmp(argv[i], "-workers", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				numWorkers = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-width", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				imageWidth = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-height", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				imageHeight = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-port", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				serverPort = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-addr", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				serverAddr = argv[i + 1];
				i++;
			}
		}

		if (strncmp(argv[i], "-i", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				inputFilename = argv[i + 1];
				i++;
			}
		}

		if (strncmp(argv[i], "-o", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				outputFilename = argv[i + 1];
				i++;
			}
		}

		if (strncmp(argv[i], "-h", strlen(argv[i])) == 0) {
			printHelp();
		}

		if (strncmp(argv[i], "-wgX", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				workgroupSizeX = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-wgY", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				workgroupSizeY = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-wgZ", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				workgroupSizeZ = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-rays", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				raysPerPixel = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-samples", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				samplesPerPixel = atoi(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-jitter", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				jitterRadius = (float)atof(argv[i + 1]);
				i++;
			}
		}

		if (strncmp(argv[i], "-depth", strlen(argv[i])) == 0) {
			if (i + 1 <= argc) {
				// peek next
				rayDepth = atoi(argv[i + 1]);
				i++;
			}
		}

		if (getParameterf(argc, argv, &i, "-turbidity", &sun_turbidity)) {
		}
	}

	if (argc == 1) {
		printHelp();
	}
}


SceneConfiguration::~SceneConfiguration() {

}


void SceneConfiguration::printHelp() {
	cerr << "Sunfish" << endl;
	cerr << "Physically Based Monte Carlo Path Tracer" << endl;
	cerr << "========================================" << endl;
	cerr << "by Jonathan Metzgar" << endl;
	cerr << "Licensed via the MIT License" << endl;
	cerr << endl;
	cerr << "Help" << endl;
	cerr << "----" << endl;
	cerr << "Command line options:" << endl;
	cerr << "---------------------" << endl;
	cerr << "-h                     display help" << endl;
	cerr << "-i <input filename>    default: input.scn" << endl;
	cerr << "-o <output filename>   default: output.ppm" << endl;
	cerr << "-server                starts as server, waits for workers to connect" << endl;
	cerr << "-worker                starts as worker, connects to server" << endl;
	cerr << "-workers <count>       creates N worker threads, default: 16" << endl;
	cerr << "-addr <ip address>     default: 127.0.0.1" << endl;
	cerr << "-port <port number>    default: 43316" << endl;
	cerr << "-width <# pixels>      default: 1280" << endl;
	cerr << "-height <# pixels>     default: 720" << endl;
	cerr << endl;
	cerr << "Path Tracer options:" << endl;
	cerr << "--------------------" << endl;
	cerr << "-wgX <pixels>          default: 16, workgroup size X" << endl;
	cerr << "-wgY <pixels>          default: 16, workgroup size Y" << endl;
	cerr << "-wgZ <pixels>          default: 16, workgroup size Z" << endl;
	cerr << "-rays <# rays>         default: 1, number of camera rays per pixel" << endl;
	cerr << "-samples <# samples>   default: 1, number of samples per pixel" << endl;
	cerr << "-jitter <float>        default: 1.0, radius where pixel samples are chosen" << endl;
	cerr << "-depth <recursion>     default: 16, max number of recursions" << endl;
	cerr << "-turbidity <amount>    default: 1.0, ranges from 1.0 to 10.0" << endl;
}


//////////////////////////////////////////////////////////////////////
// WorkerContext /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// WorkerContext is the information needed for a worker to trace rays
// for a subrectangle on the screen.
//////////////////////////////////////////////////////////////////////


struct WorkerContext {
	int left{ 0 };
	int bottom{ 0 };
	int right{ 0 };
	int top{ 0 };
	Scene* scene{ nullptr };
	Image4f* framebuffer{ nullptr };
	SceneConfiguration* sceneConfig{ nullptr };
};


int pathTraceWorker(WorkerContext* wc);


mutex framebuffer_mutex;


int pathTraceWorker(WorkerContext* wc) {
	if (wc == nullptr || wc->scene == nullptr || wc->sceneConfig == nullptr || wc->framebuffer == nullptr) {
		cerr << "blah!";
		return -1;
	}

	Image4f tmpImage(wc->right - wc->left, wc->bottom - wc->top);

	for (int i = wc->left; i <= wc->right; i++) {
		for (int j = wc->top; j <= wc->bottom; j++) {
			Vector3f output(0.0f, 0.0f, 0.0f);
			for (int s = 0; s < wc->sceneConfig->samplesPerPixel; s++) {
				float u = float(i + RTrandom.frand() * wc->sceneConfig->jitterRadius) / (float)wc->sceneConfig->imageWidth;
				float v = float(j + RTrandom.frand() * wc->sceneConfig->jitterRadius) / (float)wc->sceneConfig->imageHeight;

				Rayf r = wc->scene->camera.getRayDOF(u, v);
				Vector3f p = r.getPointAtParameter(2.0f);
				output += wc->scene->trace(r, 0);
				//output += Trace(r, wc->scene->world, 0, wc->scene->environment);
			}

			// Post processing step...
			output /= wc->sceneConfig->samplesPerPixel;

			// TODO: change this to a programmable stage

			// gamma correct image

			Color4f gammaCorrectedColor{ sqrt(output.x), sqrt(output.y), sqrt(output.z), 1.0f };

			// Framebuffer is write only, so mutex not really needed
			// But if reading and writing was conditional, then mutex would be needed.
			// lock_guard<mutex> guard(framebuffer_mutex);
			wc->framebuffer->setPixel(i, j, gammaCorrectedColor);
		}
	}
	return 0;
}


//////////////////////////////////////////////////////////////////////
// Sunfish ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Renders a scene. If no scene is provided, then it creates a default
// scene. The order to call Sunfish is:
//
// Sunfish sunfish{argc, argv};
// sunfish.loadScene();
// sunfish.renderStart();
// for (int i = 0; i < 100; i++) {
//   sunfish.renderStep();
// }
// sunfish.renderStop();
// sunfish.saveImage();
//////////////////////////////////////////////////////////////////////


class Sunfish {
public:
	Sunfish(int argc, const char** argv);

	void loadScene();
	void renderStart();
	bool renderStep();
	void renderStop();
	void render(unsigned numIterations);
	void saveImage();
private:
	SceneConfiguration sceneConfig;
	RtoList world;
	NormalShadeMaterial normalShader;
	Scene pathTracerScene;
	Image4f framebuffer;

	void makeDefaultScene_();
};


Sunfish::Sunfish(int argc, const char** argv) :
	sceneConfig(argc, argv) {}


void Sunfish::loadScene() {
	if (sceneConfig.imageWidth == 0) {
		sceneConfig.imageWidth = 1280;
	}
	if (sceneConfig.imageHeight == 0) {
		sceneConfig.imageHeight = 720;
	}
	sceneConfig.imageAspect = float(sceneConfig.imageWidth) / float(sceneConfig.imageHeight);
}


void Sunfish::makeDefaultScene_() {
	world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, -0.5f, -0.5f), 0.25f, new MetalMaterial(Vector3f(0.8f, 0.1f, 0.1f), 0.05f)));
	world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, 0.0f, -1.0f), 0.5f, new LambertianMaterial(Vector3f(0.1f, 0.2f, 0.5f))));
	world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, -100.5f, -1.0f), 100.0f, new LambertianMaterial(Vector3f(0.8f, 0.8f, 0.0f))));
	world.RTOs.push_back(new RtoSphere(Vector3f(1.0f, 0.0f, -1.0f), 0.5f, new MetalMaterial(Vector3f(0.8f, 0.6f, 0.2f), 0.0f)));
	world.RTOs.push_back(new RtoSphere(Vector3f(-1.0f, 0.0f, -1.0f), 0.5f, new DielectricMaterial(1.5f)));

	pathTracerScene.camera.lensRadius = 0.0f;// 0.1f;
	pathTracerScene.camera.setProjection(45.0f, sceneConfig.imageAspect, 0.001f, 100.0f);
	pathTracerScene.camera.setLookAt(Vector3f(0.0f, 0.0f, 1.0f), Vector3f(0.0f, 0.0f, -1.0f), Vector3f(0.0f, 1.0f, 0.0f));

	auto start = std::chrono::steady_clock::now();

	Hf::StopWatch stopwatch;
	pathTracerScene.ssg.environment.setGroundAlbedo({ 0.1f, 0.1f, 0.1f });
	Sf::PA::CivilDateTime dtg = {
		1, 5, 2016, true, -4,
		10,0,0,0.0f,
		27.907360f, -82.324440f };
	pathTracerScene.ssg.environment.setCivilDateTime(dtg);
	pathTracerScene.ssg.environment.setTurbidity(sceneConfig.sun_turbidity);
	pathTracerScene.ssg.environment.setNumSamples(1);
	pathTracerScene.ssg.environment.computeAstroFromLocale();
	pathTracerScene.ssg.environment.computePBSky();
	//pathTracerScene.ssg.environment.ComputeCubeMap(256, false, 8.0f, true);
	//pathTracerScene.ssg.environment.ComputeCylinderMap(512, 128);
	stopwatch.Stop();
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	std::cerr << "Hosek Wilkie: " << stopwatch.GetMillisecondsElapsed() << " ms" << std::endl;

	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_0.ppm", 0);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_1.ppm", 1);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_2.ppm", 2);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_3.ppm", 3);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_4.ppm", 4);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCubeMap.savePPM("pbsky_cubemap_5.ppm", 5);
	//pathTracerScene.ssg.environment.pbsky.generatedSunCylMap.savePPM("pbsky_cylmap.ppm");

	pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, -0.5f, -0.5f), 0.25f, new MetalMaterial(Vector3f(0.8f, 0.1f, 0.1f), 0.05f)));
	pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, 0.0f, -1.0f), 0.5f, new LambertianMaterial(Vector3f(0.1f, 0.2f, 0.5f))));
	pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(0.0f, -10000.5f, -1.0f), 10000.0f, new LambertianMaterial(Vector3f(0.8f, 0.8f, 0.0f))));
	pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(1.0f, 0.0f, -1.0f), 0.5f, new MetalMaterial(Vector3f(0.8f, 0.6f, 0.2f), 0.0f)));
	pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(-1.0f, 0.0f, -1.0f), 0.5f, new DielectricMaterial(1.5f)));

	if (0) {
		for (int curObj = 0; curObj < 100; curObj++) {
			Vector3f disc = getRandomUnitDiscVector();
			Vector3f color = 0.5f * (1.0f + getRandomUnitSphereVector().unit());

			//pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(4.0f * disc.x, 0.025f, 4.0f * disc.y), 0.025f, new LambertianMaterial(color)));
			pathTracerScene.world.RTOs.push_back(new RtoSphere(Vector3f(4.0f * disc.x, rand() / (float)RAND_MAX, 4.0f * disc.y), 0.1f * rand() / (float)RAND_MAX, new MetalMaterial(color, rand() / (float)RAND_MAX)));
		}
	}
	//pathTracer.AddRTO("smallSphere", new RtoSphere(Vector3f(0.0f, 0.0f, -1.0f), 0.5f));
	//pathTracer.AddRTO("largeSphere", new RtoSphere(Vector3f(0.0f, -100.5f, -1.0f), 100.0f));
	//pathTracer.AddInstance("smallSphere01", "smallSphere");
	//pathTracer.AddInstance("largeSphere01", "largeSphere");
	//pathTracer.AddMaterial("normalShader", new NormalShadeMaterial);
}


void Sunfish::render(unsigned numIterations) {
	renderStart();
	for (auto i = 0U; i < numIterations; i++) {
		renderStep();
	}
	renderStop();
}


void Sunfish::renderStart() {
	time_t t0, t1, dt;

	t0 = time(NULL);
	framebuffer = Image4f{ (int)sceneConfig.imageWidth, (int)sceneConfig.imageHeight, 1 };

	makeDefaultScene_();

	auto start = std::chrono::steady_clock::now();

	if (0) {
		WorkerContext wc;

		wc.left = 0;
		wc.right = sceneConfig.imageWidth - 1;
		wc.top = 0;
		wc.bottom = sceneConfig.imageHeight - 1;
		wc.framebuffer = &framebuffer;
		wc.scene = &pathTracerScene;
		wc.sceneConfig = &sceneConfig;

		pathTraceWorker(&wc);
	}
	else {
		vector<WorkerContext> wcs;
		vector<future<int>> futures;

		for (int i = 0; i < sceneConfig.imageWidth; i += sceneConfig.workgroupSizeX) {
			for (int j = 0; j < sceneConfig.imageHeight; j += sceneConfig.workgroupSizeY) {
				WorkerContext wc;

				wc.left = i;
				wc.right = std::min<int>(i + sceneConfig.workgroupSizeX - 1, sceneConfig.imageWidth - 1);
				wc.top = j;
				wc.bottom = std::min<int>(j + sceneConfig.workgroupSizeY - 1, sceneConfig.imageHeight - 1);
				wc.framebuffer = &framebuffer;
				wc.scene = &pathTracerScene;
				wc.sceneConfig = &sceneConfig;

				wcs.push_back(wc);
			}
		}

		for (auto wc = wcs.begin(); wc != wcs.end(); wc++) {
			WorkerContext* pwc = &(*wc);
			futures.push_back(async(pathTraceWorker, pwc));
		}

		int i = 0;
		for (auto& f : futures) {
			int result = f.get();
			i++;
			cerr << i << " " << flush;
		}
		cerr << endl;
	}

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	std::cerr << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;


	//const int maxSamples = sceneConfig.samplesPerPixel;
	//const float invSampleScale = 1.0f / maxSamples;

	//for (int j = ny - 1; j >= 0; j--)
	//{
	//	for (int i = 0; i < nx; i++)
	//	{
	//		Vector3f finalColor(0.0f, 0.0f, 0.0f);
	//		for (int s = 0; s < maxSamples; s++)
	//		{
	//			float u = float(i + RTrandom.frand()*sceneConfig.jitterRadius) / float(nx);
	//			float v = float(j + RTrandom.frand()*sceneConfig.jitterRadius) / float(ny);

	//			Rayf r = pathTracerScene.camera.getRayDOF(u, v);
	//			Vector3f p = r.getPointAtParameter(2.0f);
	//			finalColor += Trace(r, world, 0);
	//		}

	//		// Post processing step...
	//		finalColor *= invSampleScale;

	//		// gamma correct image
	//		//finalColor = Vector3f(sqrt(finalColor.r), sqrt(finalColor.g), sqrt(finalColor.b));
	//		
	//		Color4f gammaCorrectedColor = Color4f(sqrt(finalColor.r), sqrt(finalColor.g), sqrt(finalColor.b), 1.0f);
	//		framebuffer.setPixel(i, j, gammaCorrectedColor);

	//		//int ir = int(255.99f*finalColor.r);
	//		//int ig = int(255.99f*finalColor.g);
	//		//int ib = int(255.99f*finalColor.b);

	//		//cout << ir << " " << ig << " " << ib << "\n";
	//	}
	//	//cout << flush;
	//}

	t1 = time(NULL);
	dt = t1 - t0;
	cerr << "Total time: " << dt << endl;
}


bool Sunfish::renderStep() {
	return true;
}


void Sunfish::renderStop() {
	lock_guard<mutex> guard(framebuffer_mutex);

	int maxColor = 0;
	long long int total = 0;

	for (int j = sceneConfig.imageHeight - 1; j >= 0; j--) {
		for (int i = 0; i < sceneConfig.imageWidth; i++) {
			Color4f finalColor = framebuffer.getPixel(i, j);
			int ir = int(255.99f * finalColor.r);
			int ig = int(255.99f * finalColor.g);
			int ib = int(255.99f * finalColor.b);
			if (ir > maxColor) maxColor = ir;
			if (ig > maxColor) maxColor = ig;
			if (ib > maxColor) maxColor = ib;

			total += (long long)ir + ig + ib;
		}
	}

	total /= (long long)(3LL * sceneConfig.imageWidth * sceneConfig.imageHeight);
	cerr << "avg: " << total << endl;
	cerr << "max: " << maxColor << endl;
}


void Sunfish::saveImage() {
	//cout << "P3\n" << sceneConfig.imageWidth << " " << sceneConfig.imageHeight << "\n" << maxColor << "\n";

	//for (int j = sceneConfig.imageHeight - 1; j >= 0; j--) {
	//	for (int i = 0; i < sceneConfig.imageWidth; i++) {
	//		Color4f finalColor = framebuffer.getPixel(i, j);
	//		int ir = int(255.99f * finalColor.r);
	//		int ig = int(255.99f * finalColor.g);
	//		int ib = int(255.99f * finalColor.b);

	//		cout << ir << " " << ig << " " << ib << "\n";
	//	}
	//	cout << flush;
	//}

	framebuffer.flipY();
	framebuffer.saveEXR("output.exr");
}


//////////////////////////////////////////////////////////////////////
// main() ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// entry point for the program
//////////////////////////////////////////////////////////////////////


int main(int argc, const char** argv) {
	RTrandom.init(32768);

	Sunfish sunfish{ argc, argv };
	sunfish.loadScene();
	sunfish.render(100);
	sunfish.saveImage();

	return 0;
}
