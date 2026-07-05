
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

CSV_PATH=r"C:\Users\bhanu\Downloads\flamapp\xy_data.csv"

df=pd.read_csv(CSV_PATH)
points=df[['x','y']].to_numpy()

def curve(theta,M,X,samples=3000):
    t=np.linspace(6,60,samples)
    e=np.exp(M*np.abs(t))
    s=np.sin(0.3*t)
    x=t*np.cos(theta)-e*s*np.sin(theta)+X
    y=42+t*np.sin(theta)+e*s*np.cos(theta)
    return np.column_stack((x,y))

def objective(params):
    pred=curve(*params)
    tree=cKDTree(pred)
    d,_=tree.query(points)
    return np.sum(d)

bounds=[(0,np.deg2rad(50)),(-0.05,0.05),(0,100)]

de=differential_evolution(objective,bounds,popsize=20,maxiter=100,seed=42,polish=False)
opt=minimize(objective,de.x,method="L-BFGS-B",bounds=bounds)
theta,M,X=opt.x

predicted=curve(theta,M,X)


# Dense predicted curve
predicted_dense = curve(theta, M, X, samples=5000)

# KDTree on recovered curve
tree = cKDTree(predicted_dense)

# Nearest point on recovered curve
distances, indices = tree.query(points)

# Sort observed points according to recovered parameter order
order = np.argsort(indices)

ordered_points = points[order]

def uniform_arc_samples(curve_points, n=1500):

    dx = np.diff(curve_points[:,0])
    dy = np.diff(curve_points[:,1])

    seg = np.sqrt(dx**2 + dy**2)

    arc = np.concatenate(([0], np.cumsum(seg)))

    arc /= arc[-1]

    fx = interp1d(
        arc,
        curve_points[:,0],
        kind="cubic"
    )

    fy = interp1d(
        arc,
        curve_points[:,1],
        kind="cubic"
    )

    u = np.linspace(0,1,n)

    return np.column_stack((fx(u), fy(u)))

# Uniformly sampled observed curve
expected = uniform_arc_samples(ordered_points)

# Uniformly sampled recovered curve
predicted_uniform = uniform_arc_samples(predicted_dense)

# Distance from each observed point to the recovered curve
tree = cKDTree(predicted_dense)

kdtree_distances, _ = tree.query(points)

KD_L1 = np.sum(kdtree_distances)
KD_Mean = np.mean(kdtree_distances)
KD_Max = np.max(kdtree_distances)
KD_RMSE = np.sqrt(np.mean(kdtree_distances**2))

arc_distances = np.sqrt(
    (expected[:,0] - predicted_uniform[:,0])**2 +
    (expected[:,1] - predicted_uniform[:,1])**2
)

ARC_L1 = np.sum(arc_distances)
ARC_Mean = np.mean(arc_distances)
ARC_Max = np.max(arc_distances)
ARC_RMSE = np.sqrt(np.mean(arc_distances**2))


print("\n======================================================")
print("Recovered Parameters")
print("======================================================")

print(f"Theta (rad) : {theta:.10f}")
print(f"Theta (deg) : {np.degrees(theta):.10f}")
print(f"M           : {M:.10f}")
print(f"X           : {X:.10f}")

print("\n======================================================")
print("PRIMARY METRIC (Used During Optimization)")
print("KDTree Nearest-Neighbor Evaluation")
print("======================================================")

print(f"L1 Distance      : {KD_L1:.8f}")
print(f"Mean L1 Error    : {KD_Mean:.8f}")
print(f"Maximum Error    : {KD_Max:.8f}")
print(f"RMSE             : {KD_RMSE:.8f}")

print("\n======================================================")
print("ADDITIONAL VALIDATION")
print("Uniform Arc-Length Evaluation")
print("======================================================")

print(f"L1 Distance      : {ARC_L1:.8f}")
print(f"Mean L1 Error    : {ARC_Mean:.8f}")
print(f"Maximum Error    : {ARC_Max:.8f}")
print(f"RMSE             : {ARC_RMSE:.8f}")


plt.figure(figsize=(8,8))

plt.scatter(
    points[:,0],
    points[:,1],
    s=8,
    color="tab:blue",
    label="Observed Data"
)

plt.plot(
    predicted_dense[:,0],
    predicted_dense[:,1],
    color="red",
    linewidth=2,
    label="Recovered Curve"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Observed Data vs Recovered Curve")

plt.show()


plt.figure(figsize=(8,8))

plt.plot(
    expected[:,0],
    expected[:,1],
    color="tab:blue",
    linewidth=2,
    label="Expected Curve"
)

plt.plot(
    predicted_uniform[:,0],
    predicted_uniform[:,1],
    "--",
    color="tab:orange",
    linewidth=2,
    label="Recovered Curve"
)

plt.scatter(
    expected[::50,0],
    expected[::50,1],
    s=18,
    color="tab:blue"
)

plt.scatter(
    predicted_uniform[::50,0],
    predicted_uniform[::50,1],
    s=18,
    color="tab:orange"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Uniform Arc-Length Comparison")

plt.show()