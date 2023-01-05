import numpy as np
import matplotlib.pyplot as plt

#Find intersection with sphere 
def sphereIntersect(center, radius, rayOrigin, rayDirection):
    b = 2 * np.dot(rayDirection, rayOrigin - center)
    c = (np.linalg.norm(rayOrigin - center) ** 2) - (radius ** 2)
    discriminant = (b ** 2) - (4*c)
    if discriminant > 0:
        t0 = (-b - (np.sqrt(discriminant))) / 2
        t1 = (-b + (np.sqrt(discriminant))) / 2
        if t0 > 0 and t1 > 0:
            return min(t0, t1)

    return None

#Find intersection with closest object
def closestObjectIntersection(objects, rayOrigin, rayDirection):
    distances = []
    for object in objects:
        tempDistance = sphereIntersect(object['center'], object['radius'], rayOrigin, rayDirection)
        distances.append(tempDistance)
    
    closestObject = None
    minDistance = np.inf

    for i, distance in enumerate(distances):
        if distance and distance < minDistance:
            minDistance = distance
            closestObject = objects[i]

    return minDistance, closestObject

def normalizeVector(vector):
    return vector / np.linalg.norm(vector)

def findSurfaceNormal(pointOnSurface, center):
    return pointOnSurface - center

aspectRatio = 16 / 9
width = 800
height = width / aspectRatio
viewSpace = (-1, 1 / aspectRatio, 1, -1 / aspectRatio) #left, top, right, bottom
camera = np.array([0, 0, 1])

objects = [
    {'center' : np.array([0, 0, -1]), 'radius' : 0.5, 'ambient' : np.array([0.01, 0, 0]), 'diffuse' : np.array([0.84, 0.03, 0.023]), 'specular' : np.array([0.72, 0.013, 0.03]), 'shiny' : 32},
    {'center' : np.array([0, 0.5, -0.5]), 'radius' : 0.1, 'ambient' : np.array([0, 0.01, 0]), 'diffuse' : np.array([0.031, 0.712, 0.0533]), 'specular' : np.array([0.060, 0.81, 0.083]), 'shiny' : 64}
    ]
lights = [
    {'position' : np.array([0,1,0]), 'diffuse' : np.array([0.99, 0.98, 0.82]), 'specular' : np.array([0.71, 0.76, 0.73])}
    ]

ambient = np.array([1, 1, 1])

image = np.zeros((int(height), int(width), 3))

for i, y in enumerate(np.linspace(viewSpace[1], viewSpace[3], int(height))):
    for j, x in enumerate(np.linspace(viewSpace[0], viewSpace[2], width)):
        color = np.zeros((3))

        #Find closest object intersection 
        origin = camera
        direction = normalizeVector(np.array([x, y, 0]) - origin)
        minDistance, closestObj = closestObjectIntersection(objects, origin, direction)

        #If no intersection then don't draw anything
        if closestObj == None:
            continue

        #Calculate intersection point
        intersectPoint = origin + (minDistance * direction)

        #Calculate surface normal
        surfaceNormal = normalizeVector(findSurfaceNormal(intersectPoint, closestObj['center']))

        movedPoint = intersectPoint + (1e-5 * surfaceNormal)
        #check if point is shadowed 
        isShadowed = False
        for light in lights:
            shadowRay = normalizeVector(light['position'] - movedPoint)
            minDistance, closestObjShadow = closestObjectIntersection(objects, movedPoint, shadowRay)
            toLightDistance = np.linalg.norm(light['position'] - intersectPoint)

            if minDistance < toLightDistance:
                isShadowed = True
                break
        #Don't draw anything if shadowed
        if isShadowed:
            continue

        #Calculate ambient lighting
        color += closestObj['ambient'] * ambient

        for light in lights:
            lightRay = normalizeVector(light['position'] - intersectPoint)
            #Calculate diffuse lighting 
            color += closestObj['diffuse'] * light['diffuse'] * max(np.dot(lightRay, surfaceNormal), 0)
            #Calculate specular lighting
            viewerRay = normalizeVector(camera - intersectPoint)
            reflectedRay = (2*np.dot(lightRay, surfaceNormal)*surfaceNormal) - lightRay
            color += closestObj['specular'] * light['specular'] * max((np.dot(viewerRay, reflectedRay) ** closestObj['shiny']), 0)

        image[i, j] = np.clip(color, 0, 1)

plt.imsave("rayTracedImage.png", image)