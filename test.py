import math
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import time

def get_forces(x, y, obstacles_params):
    force = jnp.zeros(2)
    for obs in obstacles_params:
        ox, oy, _ = obs
        dx = x - ox
        dy = y - oy
        dist = jnp.sqrt(dx**2 + dy**2)
        rep = jnp.where(dist == 0, jnp.zeros(2), jnp.exp(-0.5*(dist)**2) * jnp.array([dx, dy]) / dist)
        force = force + rep

    obj_x = obstacles_params[-1, 0] + jnp.cos(obstacles_params[-1,2])
    obj_y = obstacles_params[-1, 1] + jnp.sin(obstacles_params[-1,2])
    dx = obj_x - x
    dy = obj_y - y
    dist = jnp.sqrt(dx**2 + dy**2)
    att = jnp.where(dist == 0, jnp.zeros(2), jnp.array([dx, dy]) / (dist * dist))
    force = force + att
    norm = jnp.sqrt(force[0]**2 + force[1]**2)
    force = jnp.where(norm == 0, force, force / norm)
    return force

def loss(obstacles_params):
    x = obstacles_params[0,0] + jnp.cos(obstacles_params[0,2])
    y = obstacles_params[0,1] + jnp.sin(obstacles_params[0,2])
    depart = jnp.array([x,y])
    arrive = jnp.array([obstacles_params[-1, 0] + jnp.cos(obstacles_params[-1,2]),
                          obstacles_params[-1, 1] + jnp.sin(obstacles_params[-1,2])])
    direction = (arrive - depart) / jnp.linalg.norm(arrive - depart)
    loss = 0.0
    speed = 0.5
    for i in range(50):
        fx, fy = get_forces(x, y, obstacles_params)
        x = x + fx * speed
        y = y + fy * speed
        
        proj = depart + jnp.dot(jnp.array([x - depart[0], y - depart[1]]), direction) * direction
        dist = jnp.sqrt((x - proj[0])**2 + (y - proj[1])**2)
        loss = loss + dist**2

        distance_to_goal = jnp.sqrt((x - arrive[0])**2 + (y - arrive[1])**2)
        if distance_to_goal < 0.5:
            break

    for i in range(len(obstacles_params)):
        for j in range(i + 1, len(obstacles_params)):
            ox1, oy1, _ = obstacles_params[i]
            ox2, oy2, _ = obstacles_params[j]
            dist_between_obs = jnp.sqrt((ox1 - ox2)**2 + (oy1 - oy2)**2)
            loss = loss + jnp.where(dist_between_obs < 2.0, (2.0 - dist_between_obs)**2 * 50.0, 0.0)

    return loss

def affichage(obstacles_params):

    def calculer_trajectoire(obstacles_params):
        x = obstacles_params[0,0] + jnp.cos(obstacles_params[0,2])
        y = obstacles_params[0,1] + jnp.sin(obstacles_params[0,2])
        xs = [x]
        ys = [y]

        arrive = jnp.array([obstacles_params[-1, 0] + jnp.cos(obstacles_params[-1,2]),
                          obstacles_params[-1, 1] + jnp.sin(obstacles_params[-1,2])])

        speed = 0.5
        for i in range(50):
            fx, fy = get_forces(x, y, obstacles_params)
            x = x + fx * speed
            y = y + fy * speed
            xs.append(x)
            ys.append(y)

            distance_to_goal = jnp.sqrt((x - arrive[0])**2 + (y - arrive[1])**2)
            if distance_to_goal < 0.5:
                break

        return xs, ys
    
    def calculer_champ_de_forces(obstacles_params):
    
        X = np.arange(-10, 10, 1)
        Y = np.arange(-10, 10, 1)
        U, V = np.empty((20, 20)), np.empty((20, 20))
        for y in Y:
            for x in X:
                force_x, force_y = get_forces(x, y, obstacles_params)
                U[y + 10, x + 10] = force_x
                V[y + 10, x + 10] = force_y

        return X, Y, U, V

    X, Y, U, V = calculer_champ_de_forces(obstacles_params)
    xs, ys = calculer_trajectoire(obstacles_params)

    # Création de la figure et des axes
    fig, ax = plt.subplots()

    # Tracé des points
    ax.plot(xs, ys, marker='o', color='red', label='Trajectoire')

    # Ajout du champ de vecteurs
    q = ax.quiver(X, Y, U, V, color='blue')

    # Légende pour le quiver
    ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

    for obs in obstacles_params:
        circle = plt.Circle((obs[0], obs[1]), 1, color='green', fill=False, label='Obstacle')
        ax.add_artist(circle)

    # Légende pour le tracé
    ax.legend()

    ax.set_aspect('equal')

    # Affichage
    plt.show()

obstacles_params = np.array([[-8, -8, np.pi],
                             [-1, -1, 0],
                             [0,0,0],
                             [-1,1,0],
                             [1,1,0],
                             [4, 4, np.pi/4]])

loss_grad_fn = jax.grad(loss, argnums=0)

for _ in range(10):
    affichage(obstacles_params)

    t = time.time()
    grads = loss_grad_fn(obstacles_params)
    print(time.time() - t)

    for i, grad in enumerate(grads):
        print(f"Gradient obstacle {i} (x, y, rotation): {grad}")
    
    obstacles_params = obstacles_params - 0.02 * grads