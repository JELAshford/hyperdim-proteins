from hyperdim_proteins import embed_sequences_positional, py_generate_trimer_hdvs
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np


AMINO_ACIDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
N_DIM = 10_000
N_FAMILIES = 50
SEED = 1701
RANGE_STEP = 0.01
LEARNING_RATE = 1e-2


def range_hdvs(steps, seed=SEED):
    np.random.seed(seed)
    hdv = lambda size=N_DIM: np.random.choice([-1, 1], size=size)
    k = len(steps) - 1
    V = hdv((k + 1, N_DIM))
    for i in range(1, k + 1):
        for j in range(N_DIM):
            V[i, j] = -V[i - 1, j] if np.random.random() < 1 / k else V[i - 1, j]
    return V.astype(np.int8)


def bundle(hdv_list):
    out = np.sign(np.sum(np.stack(hdv_list), axis=0))
    out[out == 0] = 1
    return out


def cosine_sim_batch(v1, v2):
    v2_transpose = v2.T
    norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm_v2 = np.linalg.norm(v2_transpose, axis=0, keepdims=True)
    normed = np.dot(norm_v1, norm_v2)
    return (np.dot(v1, v2_transpose) / normed).astype(np.float16)


# Generate hypervector info
trimer_hdvs = py_generate_trimer_hdvs(AMINO_ACIDS, SEED)
position_hdvs = range_hdvs(list(np.arange(0, 1, RANGE_STEP)))
fig, axs = plt.subplots(1, 2)
axs[0].imshow(cosine_sim_batch(position_hdvs, position_hdvs))
axs[1].imshow(np.corrcoef(position_hdvs))
plt.savefig("results/position_positions.png")
plt.show()
embed = lambda seqs: np.stack(
    embed_sequences_positional(seqs, trimer_hdvs, position_hdvs)
)

# Load in the protein data
train_data = pd.read_feather("data/train.feather")
dev_data = pd.read_feather("data/dev.feather")
test_data = pd.read_feather("data/test.feather")

# Extract data from N most abundant fammilies
families = train_data.value_counts("family_accession")[:N_FAMILIES].index.values
train_family_data = train_data.query("family_accession.isin(@families)").reset_index()
dev_family_data = dev_data.query("family_accession.isin(@families)").reset_index()
test_family_data = test_data.query("family_accession.isin(@families)").reset_index()
# Convert sequences to embeddings
train_family_embeds = embed(train_family_data["sequence"])
dev_family_embeds = embed(dev_family_data["sequence"])
test_family_embeds = embed(test_family_data["sequence"])
# Extract accessions
train_family_accessions = train_family_data["family_accession"].values
dev_family_accessions = dev_family_data["family_accession"].values
test_family_accessions = test_family_data["family_accession"].values


# Train: Bundle training data into prototypes
all_prototypes = np.zeros(shape=(len(families), N_DIM))
for ind, family in enumerate(families):
    all_prototypes[ind, :] = bundle(
        train_family_embeds[train_family_accessions == family, :]
    )

# Predict: measure simularity between encoded sequencs and each prototype
similarity_array = cosine_sim_batch(all_prototypes, test_family_embeds)
pred_cat = families[np.argmax(similarity_array, axis=0)]
print(f"pred_acc = {balanced_accuracy_score(test_family_accessions, pred_cat):.2%}")
sns.heatmap(confusion_matrix(test_family_accessions, pred_cat, normalize="true"))
plt.savefig("results/position_confusion.png")
plt.show()


# Visualise embedding spaces
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

X = similarity_array.T
colors = LabelEncoder().fit_transform(test_family_accessions)
pca_embed = PCA()
X_new = pca_embed.fit_transform(X)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=colors, alpha=0.5)
plt.savefig("results/position_pca.png")
plt.show()


# Look at similarity of prototypes
proto_sim = cosine_sim_batch(all_prototypes, all_prototypes)

these_family_ids = (
    train_data.query("family_accession.isin(@families)")
    .filter(["family_id", "family_accession"], axis=1)
    .drop_duplicates()
    .set_index("family_accession")
    .reindex(families)["family_id"]
    .values
)
family_id = these_family_ids  # families
df = (
    pd.DataFrame(proto_sim, index=family_id)
    .assign(from_family=family_id)
    .melt(id_vars="from_family")
    .assign(to_family=lambda x: family_id[x["variable"].values.astype(int)])
    .drop(columns=["variable"])
)
g = sns.clustermap(
    df,
    pivot_kws=dict(index="from_family", columns="to_family", values="value"),
    cmap="Blues",
    robust=True,
    dendrogram_ratio=0.1,
    figsize=(30, 30),
)
plt.savefig("results/position_prototypes.png")
