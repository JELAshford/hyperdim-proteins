from hyperdim_proteins import embed_sequences, py_generate_trimer_hdvs
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np


AMINO_ACIDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
N_DIM = 10_000
N_FAMILIES = 50
SEED = 1701
LEARNING_RATE = 1e-2


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


def top_n_class_matches(class_embeds, sequence_embeds, n=1):
    """return the class indexes of top n matches between squence embeddings
    and class embeddings"""
    out = np.zeros(shape=(sequence_embeds.shape[0], n), dtype=np.int16)
    seq_norms = np.linalg.norm(sequence_embeds, axis=1)
    class_norms = np.linalg.norm(class_embeds, axis=1)
    for row, (seq, norm) in enumerate(zip(sequence_embeds, seq_norms)):
        class_cosine_sims = np.dot(seq, class_embeds.T) / (class_norms * norm)
        out[row, :] = np.argpartition(class_cosine_sims, -n)[-n:]
    return out


# Generate hypervector info
trimer_hdvs = py_generate_trimer_hdvs(AMINO_ACIDS, SEED)
embed = lambda seqs: np.stack(embed_sequences(seqs, trimer_hdvs))

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

# # Re-Train: update prototypes to discourage misclassification
# for _ in range(10):
#     # train_preds = cosine_sim_batch(all_prototypes, train_family_embeds)
#     # pred_train_families = families[np.argmax(train_preds, axis=0)]
#     pred_train_families = families[
#         top_n_class_matches(all_prototypes, train_family_embeds).flatten()
#     ]
#     print(balanced_accuracy_score(train_family_accessions, pred_train_families))
#     for index in np.where(train_family_accessions != pred_train_families)[0]:
#         embed = train_family_embeds[index]
#         true_family = train_family_accessions[index]
#         predicted_family = pred_train_families[index]
#         # Add this embed to it's correct family
#         all_prototypes[families == true_family, :] += LEARNING_RATE * embed
#         # Subtract this embed from it's mispredicted family
#         all_prototypes[families == predicted_family, :] -= LEARNING_RATE * embed
#     all_prototypes = np.sign(all_prototypes)
#     all_prototypes[all_prototypes == 0] = 1


# Predict: measure simularity between encoded sequencs and each prototype
# similarity_array = cosine_sim_batch(all_prototypes, test_family_embeds)
# pred_cat = families[np.argmax(similarity_array, axis=0)]
pred_cat = families[top_n_class_matches(all_prototypes, test_family_embeds).flatten()]
print(f"pred_acc = {balanced_accuracy_score(test_family_accessions, pred_cat):.2%}")
sns.heatmap(confusion_matrix(test_family_accessions, pred_cat, normalize="true"))
plt.show()


# # Visualise embedding spaces
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA

# X = similarity_array.T
# colors = LabelEncoder().fit_transform(test_family_accessions)
# pca_embed = PCA()
# X_new = pca_embed.fit_transform(X)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(projection="3d")
# scatter = ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=colors, alpha=0.5)
# plt.show()


# # Look at similarity of prototypes
# proto_sim = cosine_sim_batch(all_prototypes, all_prototypes)
# these_family_ids = (
#     train_data.query("family_accession.isin(@families)")
#     .filter(["family_id", "family_accession"], axis=1)
#     .drop_duplicates()
#     .set_index("family_accession")
#     .reindex(families)["family_id"]
#     .values
# )
# family_id = these_family_ids  # families
# df = (
#     pd.DataFrame(proto_sim, index=family_id)
#     .assign(from_family=family_id)
#     .melt(id_vars="from_family")
#     .assign(to_family=lambda x: family_id[x["variable"].values.astype(int)])
#     .drop(columns=["variable"])
# )
# g = sns.clustermap(
#     df,
#     pivot_kws=dict(index="from_family", columns="to_family", values="value"),
#     cmap="Blues",
#     robust=True,
#     dendrogram_ratio=0.1,
#     figsize=(30, 30),
# )
