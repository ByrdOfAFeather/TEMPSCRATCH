import pandas as pd
import matplotlib.pyplot as plt
import umap
import json

all_vectors = pd.read_csv("all.csv")
print(all_vectors.shape)
print("here")
mapper = umap.UMAP(random_state=225530)
umap_reduced = mapper.fit_transform(all_vectors.values[:, 1:])
fig, ax = plt.subplots()
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))


vectors=umap_reduced
only_use=""
index_mapping={}
with open("all_indicies.json", 'r') as f:
	bert_vectors = json.load(f)
colors = ['b', 'g', 'r', 'purple']
color_idx = 0
idx = 0
for key, value in bert_vectors.items():
	if only_use:
		if only_use not in key: continue
	if index_mapping:
		sc = ax.scatter(vectors[[index_mapping[i] for i in value], 0], vectors[[index_mapping[i] for i in value], 1], color=colors[color_idx], label=key, s=5)
	else:
		if idx == 0:
			sc = ax.scatter(vectors[value, 0], vectors[value, 1], color=colors[color_idx], label=key, s=5)
		else:
			ax.scatter(vectors[value, 0], vectors[value, 1], color=colors[color_idx], label=key, s=5)
	color_idx += 1
	idx += 1

def update_annot(ind):
	pos = sc.get_offsets()[ind["ind"][0]]
	annot.xy = pos
	text = "{}".format("".join(str([f"{n}" for n in ind["ind"]])))
	annot.set_text(text)
	annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
	vis = annot.get_visible()
	if event.inaxes == ax:
		cont, ind = sc.contains(event)
		if cont:
			update_annot(ind)
			annot.set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis:
				annot.set_visible(False)
				fig.canvas.draw_idle()
fig = plt.gcf()
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.legend()
plt.show()


plt.title("REDUCED BERT VECTORS")
plot_by_class()