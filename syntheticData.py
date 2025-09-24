import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KernelDensity, radius_neighbors_graph
from sklearn.metrics import roc_auc_score, pairwise_distances
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal, spearmanr
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
#from scipy.stats import entropy 


########################################
# Synthetic data generation
########################################
def create_data(mu0,mu1,sig0,sig1,N0,N1,test_split=0.3):
    np.random.seed(0)
    torch.manual_seed(0)

    X0 = np.random.multivariate_normal(mu0, sig0, N0)
    X1 = np.random.multivariate_normal(mu1, sig1, N1)
    X = np.vstack([X0, X1])
    y = np.array([0]*N0 + [1]*N1)
    return train_test_split(X, y, test_size=test_split, stratify=y)

################################################
# Analytical Bayes uncertainty
##################################################
def bayes_posterior(x, mu0, mu1, sig0, sig1, pi0=0.5, pi1=0.5):
    p0 = pi0 * multivariate_normal.pdf(x, mean=mu0, cov=sig0)
    p1 = pi1 * multivariate_normal.pdf(x, mean=mu1, cov=sig1)
    denom = p0 + p1
    return np.array([p0/denom, p1/denom])  # [P(y=0|x), P(y=1|x)]

def bayes_entropy(X, mu0, mu1, sig0, sig1, N0, N1):
    ent = []
    for x in X:
        probs = bayes_posterior(x, mu0, mu1, sig0, sig1, N0/(N0+N1), N1/(N0+N1))
        ent.append(-probs[0]*np.log(probs[0] + 1e-10)-probs[1]*np.log(probs[1] + 1e-10)) #entropy(probs, base=np.e))
    return np.array(ent)



###################################################
# NN classifier, do dropout for MC later
###################################################

class SimpleNN(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


################################################
# get MC-dropout epistemic proxy uncertainty
#################################################
def mc_dropout_preds(model, X, T=30):
    model.train()  # keep dropout on
    X = torch.tensor(X, dtype=torch.float32)
    preds = []
    for _ in range(T):
        with torch.no_grad():
            p = torch.softmax(model(X), dim=1).numpy()
        preds.append(p)
    preds = np.stack(preds)  # (T, N, 2)
    mean_p = preds.mean(axis=0)
    var_p = preds.var(axis=0).mean(axis=1)  # mean variance per sample
    return mean_p, var_p



def mixed_persistence_score(X, labels, radii):
    """
    Compute per-point mixed-component persistence scores.
    Args:
        X: ndarray (N, D) point cloud (embeddings or original 2D points)
        labels: ndarray (N,) integer labels (0/1)
        radii: iterable of radii to sweep (increasing)
    Returns:
        scores: ndarray (N,) values in [0,1], fraction of radii where point sits in mixed component
    """
    N = X.shape[0]
    T = len(radii)
    mixed_indicator = np.zeros((T, N), dtype=np.bool_)

    for ti, r in enumerate(tqdm(radii, desc="TDA radii sweep")):
        # build symmetric radius graph (sparse)
        # mode='connectivity' returns adjacency 1/0 for neighbors within r
        A = radius_neighbors_graph(X, radius=r, mode='connectivity', include_self=True, n_jobs=-1)
        # ensure symmetry (radius_neighbors_graph sometimes returns asymmetric)
        A = (A + A.T) > 0

        # connected components on adjacency
        n_components, labels_cc = connected_components(csgraph=A, directed=False, return_labels=True)

        # for each component, check composition
        # vectorized approach: for each component id, find if it contains both labels
        comp_has_label0 = np.zeros(n_components, dtype=bool)
        comp_has_label1 = np.zeros(n_components, dtype=bool)
        # Count composition per component
        for k in range(n_components):
            comp_idx = np.flatnonzero(labels_cc == k)
            # cheap check
            if np.any(labels[comp_idx] == 0):
                comp_has_label0[k] = True
            if np.any(labels[comp_idx] == 1):
                comp_has_label1[k] = True

        comp_mixed = comp_has_label0 & comp_has_label1  # boolean per component
        # for each point, mark True if its component is mixed
        mixed_indicator[ti, :] = comp_mixed[labels_cc]

    # per-point score: fraction of radii where it was in mixed component
    scores = mixed_indicator.mean(axis=0)  # in [0,1]
    return scores

# ---------------------------
# How to call it in Stage0
# ---------------------------
# Example: use raw X_test (2D points) or model embeddings.
# Choose radii grid relative to scale of X (e.g., quantiles of pairwise distances)



##########################################################
#  Training wrapper
##########################################################3
def train(X_train, y_train, nepochs, dropout=0.4, verbose=True):
    model = SimpleNN(p=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(nepochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 40 == 0):
            print(f"Epoch {epoch}, loss {loss.item():.4f}")
    
    return model

###########################################3
# Mahalanobis distance
############################################
def mahalanobis(x, mean, inv_cov):
    d = x-mean
    return np.sqrt(d @ inv_cov @ d.T)



def run_experiment(mu0, mu1, Sigma0, Sigma1, N0, N1, nepochs, dropout, exp_name):
    date="2025-09-19"
    print(f"mu0={mu0}, mu1={mu1}, Sigma0={Sigma0}, Sigma1={Sigma1}, N0={N0}, N1={N1}, nepochs={nepochs}, dropout={dropout}")
    X_train, X_test, y_train, y_test = create_data(mu0, mu1, Sigma0, Sigma1, N0, N1)

    model = train(X_train, y_train, nepochs, dropout=dropout, verbose=True)

    H_test = bayes_entropy(X_test, mu0, mu1, Sigma0, Sigma1, N0, N1)
    mean_p, var_p = mc_dropout_preds(model, X_test)

    # kNN distance proxy
    nbrs = NearestNeighbors(n_neighbors=6).fit(X_train)
    distances, _ = nbrs.kneighbors(X_test)
    knn_dist = distances[:,1:].mean(axis=1)

    # KDE log density
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_train)
    kde_score = kde.score_samples(X_test)

    # Mahalanobis distance
    mean0 = X_train[y_train==0].mean(axis=0)
    cov0 = np.cov(X_train[y_train==0].T)
    inv_cov0 = np.linalg.inv(cov0)
    mahal = np.array([mahalanobis(x, mean0, inv_cov0) for x in X_test])

    # TDA mixed-component persistence score
    # Use embeddings or original X_test points:
    X_for_tda = X_test.copy()  # or embeddings from a hidden layer
    labels_tda = y_test.copy()
    # Build a reasonable radii grid:
    dists = pairwise_distances(X_for_tda, metric='euclidean')
    # use upper triangle distances to get stats
    triu_idx = np.triu_indices(dists.shape[0], k=1)
    sampled = dists[triu_idx]
    # choose radii from small to moderate (percentiles)
    radii = np.percentile(sampled, np.linspace(1, 50, 25))  # 25 radii between 1st and 50th percentile
    # compute mixed persistence scores
    mixed_scores = mixed_persistence_score(X_for_tda, labels_tda, radii)

    

    # get stats
    print("MC-dropout vs Entropy:", spearmanr(var_p, H_test).correlation)
    print("kNN-dist vs Entropy:", spearmanr(knn_dist, H_test).correlation)
    print("KDE vs Entropy:", spearmanr(kde_score, H_test).correlation)
    print("Mahalanobis vs Entropy:", spearmanr(mahal, H_test).correlation)
    # Correlate with Bayes entropy H_test (must align indices)
    rho, pval = spearmanr(mixed_scores, H_test)
    print(f"Mmixed-persistence vs Entropy: rho={rho:.3f}, p={pval:.3g}")
    # You can also compute Spearman correlations of mixed_scores with MC-dropout variance etc.
    rho2, p2 = spearmanr(mixed_scores, var_p)  # var_p from earlier MC-dropout
    print(f"Mixed-persistence vs MC var: rho={rho2:.3f}, p={p2:.3g}")
    #with open("results.txt", "a") as f:
    #    f.write(f"{exp_name},{date},{nepochs},{dropout},{mu0},{mu1},{Sigma0},{Sigma1},{N0},{N1},")
    #    f.write(f"{spearmanr(var_p, H_test).correlation}, {spearmanr(knn_dist, H_test).correlation},{spearmanr(kde_score, H_test).correlation},{spearmanr(mahal, H_test).correlation}\n")

    # visualize
    plt.figure(figsize=(6,5))
    plt.scatter(X_test[:,0], X_test[:,1], c=H_test, cmap="viridis", s=10)
    plt.colorbar(label="Bayes entropy")
    plt.title("True Bayes aleatoric uncertainty")
    plt.savefig(f"uncertainty_{exp_name}.png")
    plt.close()

    plt.figure(figsize=(5,4))
    plt.scatter(mixed_scores, H_test, s=10, alpha=0.6)
    plt.xlabel("Mixed-component persistence score")
    plt.ylabel("Bayes entropy")
    plt.title("TDA mixed persistence vs Bayes entropy")
    plt.savefig(f"tda_{exp_name}.png")
    plt.close()
    return spearmanr(var_p, H_test).correlation, spearmanr(knn_dist, H_test).correlation, spearmanr(kde_score, H_test).correlation,spearmanr(mahal, H_test).correlation, rho, rho2





if __name__ == "__main__":
    #Q: what happens with uncertainty estimates and proxies if you train longer?
    Vseh = 24
    data = {'nepoch':[50,250,500,1000]+[1000]*(Vseh-4), 
            'dropout':[0.5,0.5,0.5,0.5, 0.1,0.3,0.5,0.7]+[0.5]*(Vseh-8), 
            'nsample_0':[2500]*8+[100,1000,2500,5000,500,1000,2000]+[2500]*(Vseh-15), 
            'nsample_1':[2500]*8+[100,1000,2500,5000,4500,4000,3000]+[2500]*(Vseh-15),
            'mu0': [0]*Vseh, 
            'mu1': [1]*15 + [0,0.5,1,2,4]+[1]*(Vseh-20), 
            'sigma0':[1]*Vseh,
            'sigma1': [1]*20+[0.5,1,1.5,2],
            'corrMC': [0]*Vseh, 'corrKNN': [0]*Vseh, 'corrKDE': [0]*Vseh, 'corrMahalanobis': [0]*Vseh}

    for idx in range(24):
        ep = data['nepoch'][idx]
        dr = data['dropout'][idx]
        n0 = data['nsample_0'][idx]
        n1 = data['nsample_1'][idx]
        mu0 = data['mu0'][idx]
        mu1 = data['mu1'][idx]
        sigma0 = data['sigma0'][idx]
        sigma1 = data['sigma1'][idx]
        sigma0 = np.eye(2)*sigma0
        sigma1 = np.eye(2)*sigma1
        mu0 = np.array([mu0, 0])
        mu1 = np.array([mu1, 0])
        mc, knn, kde, maha = run_experiment(mu0, mu1, sigma0, sigma1, n0, n1, nepochs=ep, dropout=dr, exp_name=f"{idx}")
        data['corrMC'][idx] = mc
        data['corrKNN'][idx] = knn
        data['corrKDE'][idx] = kde
        data['corrMahalanobis'][idx] = maha

    import pandas as pd
    pd.DataFrame(data).to_csv("experiment_summary.csv")
    #Q: what happens with epistemic estimates by MC dropout, if you vary dropout prob?
    #dropouts = [0.1, 0.3, 0.5, 0.7]
    #for dp in dropouts:
    #    run_experiment(np.array([0,0]), np.array([0,1]), np.eye(2), np.eye(2), 2500, 2500, nepochs=500, dropout=dp, exp_name=f"dp{dp}")
    #Q: what happens with entropy, if you vary nr of samples, mu separation, dimensionality, covariances?
    #nr_samples = [(100,100), (1000,1000), (2500,2500), (5000,5000), (500, 4500), (1000, 4000), (2000, 3000)]
    #for (n0, n1) in nr_samples:
    #    run_experiment(np.array([0,0]), np.array([0,1]), np.eye(2), np.eye(2), n0, n1, nepochs=500, dropout=0.5, exp_name=f"ns{n0}_{n1}")
    #mus1 = [[0, 0], [0.5, 0], [0, 2], [4, 0]]
    #for mu in mus1:
    #    run_experiment(np.array([0,0]), np.array(mu), np.eye(2), np.eye(2), 2500, 2500, nepochs=500, dropout=0.5, exp_name=f"mu{mu[0]}_{mu[1]}")
    #sigmas1 = [np.eye(2)*0.5, np.eye(2), np.eye(2)*2]
    #for sigma in sigmas1:
    #    run_experiment(np.array([0,0]), np.array([0,1]), np.eye(2), sigma, 2500, 2500, nepochs=500, dropout=0.5, exp_name=f"sig{sigma[0,0]}_{sigma[1,1]}")

    #mu0 = np.array([0, 0])
    #mu1 = np.array([2, 0])   # move closer/further for more/less overlap
    #Sigma0 = np.eye(2)

    #N0 = 2500
    #N1 = 2500

