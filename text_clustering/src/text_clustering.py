import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP
from edc.edc.utils.llm_utils import get_embedding_sts, openai_chat_completion_with_key

logging.basicConfig(level=logging.INFO)


DEFAULT_INSTRUCTION = (
    instruction
) = "Use three words total (comma separated)\
to describe general topics in above texts. Under no circumstances use enumeration. \
Example format: Tree, Cat, Fireman"

DEFAULT_TEMPLATE = "<s>[INST]{examples}\n\n{instruction}[/INST]"


class ClusterClassifier:
    def __init__(
        self,
        embed_model_name="all-MiniLM-L6-v2",
        embed_device="cpu",
        embed_batch_size=64,
        embed_max_seq_length=512,
        embed_agg_strategy=None,
        umap_components=2,
        umap_metric="cosine",
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        dbscan_eps=0.7,
        dbscan_min_samples=2,
        dbscan_n_jobs=16,
        summary_create=True,
        summary_model="gemini-1.5-flash",
        topic_mode="multiple_topics",
        summary_n_examples=10,
        summary_chunk_size=420,
        summary_model_token=True,
        summary_template=None,
        summary_instruction=None,
    ):
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy

        self.umap_components = umap_components
        self.umap_metric = umap_metric
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_n_jobs = dbscan_n_jobs

        self.summary_create = summary_create
        self.summary_model = summary_model
        self.topic_mode = topic_mode
        self.summary_n_examples = summary_n_examples
        self.summary_chunk_size = summary_chunk_size
        self.summary_model_token = summary_model_token

        if summary_template is None:
            self.summary_template = DEFAULT_TEMPLATE
        else:
            self.summary_template = summary_template

        if summary_instruction is None:
            self.summary_instruction = DEFAULT_INSTRUCTION
        else:
            self.summary_instruction = summary_instruction

        self.embeddings = None
        self.faiss_index = None
        self.cluster_labels = None
        self.texts = None
        self.projections = None
        self.umap_mapper = None
        self.id2label = None
        self.label2docs = None

    def fit(self, texts, embeddings=None):
        self.texts = texts

        if embeddings is None:
            logging.info("embedding texts...")
            self.embeddings = self.embed(texts)
        else:
            logging.info("using precomputed embeddings...")
            self.embeddings = embeddings

        logging.info("building faiss index...")
        self.faiss_index = self.build_faiss_index(self.embeddings)
        logging.info("projecting with umap...")
        self.projections, self.umap_mapper = self.project(self.embeddings)
        logging.info("dbscan clustering...")
        self.cluster_labels = self.cluster(self.projections)

        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

        if self.summary_create:
            logging.info("summarizing cluster centers...")
            self.cluster_summaries = self.summarize(self.texts, self.cluster_labels)
        else:
            self.cluster_summaries = None

        return self.embeddings, self.cluster_labels, self.cluster_summaries

    def infer(self, texts, top_k=1):
        embeddings = self.embed(texts)

        dist, neighbours = self.faiss_index.search(embeddings, top_k)
        inferred_labels = []
        for i in tqdm(range(embeddings.shape[0])):
            labels = [self.cluster_labels[doc] for doc in neighbours[i]]
            inferred_labels.append(Counter(labels).most_common(1)[0][0])

        return inferred_labels, embeddings

    def embed(self, texts):
        all_embeddings = []
        for text in tqdm(texts, desc="Computing embeddings"):
            embedding = get_embedding_sts(text, device=self.embed_device)
            all_embeddings.append(embedding)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        return embeddings

    def project(self, embeddings):
        mapper = UMAP(n_components=self.umap_components, metric=self.umap_metric, n_neighbors=self.umap_n_neighbors, min_dist=self.umap_min_dist).fit(
            embeddings
        )
        return mapper.embedding_, mapper

    def cluster(self, embeddings):
        print(
            f"Using DBSCAN (eps, nim_samples)=({self.dbscan_eps,}, {self.dbscan_min_samples})"
        )
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=self.dbscan_n_jobs,
        ).fit(embeddings)

        return clustering.labels_

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def summarize(self, texts, labels):
        unique_labels = len(set(labels)) - 1  # exclude the "-1" label
        cluster_summaries = {-1: "None"}

        for label in range(unique_labels):
            ids = np.random.choice(self.label2docs[label], self.summary_n_examples)
            examples = "\n\n".join(
                [
                    f"Example {i+1}:\n{texts[_id][:self.summary_chunk_size]}"
                    for i, _id in enumerate(ids)
                ]
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant that can summarize text clusters."},
                {"role": "user", "content": f"{examples}\n\n{self.summary_instruction}"}
            ]
            
            response = openai_chat_completion_with_key(
                model=self.summary_model,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )
            
            if label == 0:
                print(f"Request examples:\n{examples[:200]}...\nInstruction: {self.summary_instruction}")
                print(f"Response: {response}")
                
            cluster_summaries[label] = self._postprocess_response(response)
        print(f"Number of clusters is {len(cluster_summaries)}")
        return cluster_summaries

    def _postprocess_response(self, response):
        if self.topic_mode == "multiple_topics":
            summary = response.split("\n")[0].split(".")[0].split("(")[0]
            summary = ",".join(
                [txt for txt in summary.strip().split(",") if len(txt) > 0]
            )
            return summary
        elif self.topic_mode == "single_topic":
            first_line = response.split("\n")[0]
            topic, score = None, None
            try:
                topic = first_line.split("Topic:")[1].split("(")[0].split(",")[0].strip()
            except IndexError:
                print("No topic found")
            try:
                score = first_line.split("Educational value rating:")[1].strip().split(".")[0].strip()
            except IndexError:
                print("No educational score found")
            full_output = f"{topic}. Educational score: {score}"
            return full_output
        else:
            raise ValueError(
                f"Topic labeling mode {self.topic_mode} is not supported, use single_topic or multiple_topics instead."
            )

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        faiss.write_index(self.faiss_index, f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "wb") as f:
            np.save(f, self.projections)

        with open(f"{folder}/cluster_labels.npy", "wb") as f:
            np.save(f, self.cluster_labels)

        with open(f"{folder}/texts.json", "w") as f:
            json.dump(self.texts, f)

        with open(f"{folder}/mistral_prompt.txt", "w") as f:
            f.write(DEFAULT_INSTRUCTION)

        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)

    def load(self, folder):
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exsit.")

        with open(f"{folder}/embeddings.npy", "rb") as f:
            self.embeddings = np.load(f)

        self.faiss_index = faiss.read_index(f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "rb") as f:
            self.projections = np.load(f)

        with open(f"{folder}/cluster_labels.npy", "rb") as f:
            self.cluster_labels = np.load(f)

        with open(f"{folder}/texts.json", "r") as f:
            self.texts = json.load(f)

        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # those objects can be inferred and don't need to be saved/loaded
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def show(self, interactive=False, save_path=None):
        df = pd.DataFrame(
            data={
                "X": self.projections[:, 0],
                "Y": self.projections[:, 1],
                "labels": self.cluster_labels,
                "content_display": [
                    textwrap.fill(txt[:1024], 64) for txt in self.texts
                ],
            }
        )

        if interactive:
            self._show_plotly(df, save_path)
        else:
            self._show_mpl(df, save_path)

    def _show_mpl(self, df, save_path=None):
        # Set a more attractive style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

        # Create a custom colormap for better visual appeal
        from matplotlib.colors import ListedColormap
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        cluster_cmap = ListedColormap(palette)
        
        # Get unique labels (excluding -1) and their count
        unique_labels = sorted(list(set([l for l in df['labels'] if l != -1])))
        noise_mask = df['labels'] == -1
        
        # Check if there are any non-noise points
        has_clusters = sum(~noise_mask) > 0
        
        if has_clusters:
            # Plot the clusters with better aesthetics
            scatter = ax.scatter(
                df.loc[~noise_mask, 'X'], 
                df.loc[~noise_mask, 'Y'],
                c=df.loc[~noise_mask, 'labels'],
                cmap=cluster_cmap,
                s=150,  # Much larger point size
                alpha=0.9,
                edgecolor='black',
                linewidth=0.5,
            )
            
            # Add a legend for clusters
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Clusters")
            ax.add_artist(legend1)
        
        # Plot noise points with a different style
        if sum(noise_mask) > 0:
            noise_scatter = ax.scatter(
                df.loc[noise_mask, 'X'], 
                df.loc[noise_mask, 'Y'],
                c='gray',
                s=80,
                alpha=0.6,
                marker='x',
                linewidth=2,
                label='Noise'
            )
            
            # If there are only noise points, add a legend for just noise
            if not has_clusters:
                ax.legend(handles=[noise_scatter], loc="upper right")
        
        # Add point count labels near each point for extra visibility
        if len(df) < 50:  # Only for small datasets
            for i, row in df.iterrows():
                ax.annotate(
                    f"#{i}",
                    (row['X'], row['Y']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
            
        # Add more informative title and labels
        plt.title('Document Clusters Visualization', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        
        # Add cluster labels with better styling
        for label in self.cluster_summaries.keys():
            if label == -1:
                continue
            summary = self.cluster_summaries[label]
            position = self.cluster_centers[label]
            
            # Add a background bubble to highlight the cluster center
            ax.scatter(position[0], position[1], s=400, alpha=0.4, 
                      color=plt.cm.tab20(label % 20), edgecolor='white')
            
            # Add the label text with better styling
            t = ax.text(
                position[0],
                position[1],
                summary,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                fontweight='bold',
                color='black',
            )
            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1, 
                          edgecolor='black', boxstyle='round,pad=0.4'))
        
        # Add information about the number of clusters and documents
        info_text = f"Total Documents: {len(df)}\nClusters: {len(unique_labels)}\nNoise Points: {sum(noise_mask)}"
        ax.text(0.02, 0.98, 
               info_text,
               transform=ax.transAxes, fontsize=12,
               va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
        
        # Add grid for better visibility of point positions
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a small descriptor of what each point represents
        if len(df) < 20:  # Only for very small datasets
            ax.text(0.02, 0.02, 
                  "Each point represents a document\nLarger circles show cluster centers",
                  transform=ax.transAxes, fontsize=10,
                  va='bottom', ha='left',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Saved visualization to {save_path}")
        plt.show()

    def _show_plotly(self, df, save_path=None):
        # Create a more attractive and informative plotly visualization
        
        # Define a better color scale
        from plotly.colors import n_colors
        import plotly.graph_objects as go
        
        # Get unique labels and count them (excluding noise)
        unique_labels = sorted(list(set([l for l in df['labels'] if l != -1])))
        noise_mask = df['labels'] == -1
        
        # Create figure
        fig = go.Figure()
        
        # Custom colorscale
        colors = px.colors.qualitative.Plotly
        
        # Add traces for each cluster
        for i, label in enumerate(unique_labels):
            cluster_data = df[df['labels'] == label]
            color = colors[i % len(colors)]
            
            # Add scatter plot for this cluster
            fig.add_trace(go.Scatter(
                x=cluster_data['X'],
                y=cluster_data['Y'],
                mode='markers+text',  # Add text mode
                marker=dict(
                    size=20,  # Much larger size
                    color=color,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                customdata=cluster_data['content_display'],
                hovertemplate='<b>Cluster %s</b><br>%{customdata}<extra></extra>' % label,
                name=f'Cluster {label}: {self.cluster_summaries.get(label, "Unlabeled")}',
                text=cluster_data.index if len(df) < 50 else None,  # Add index as text for small datasets
                textposition="top center",
                textfont=dict(size=10, color='black')
            ))
            
            # Add an annotation for the cluster center
            if label in self.cluster_centers:
                position = self.cluster_centers[label]
                summary = self.cluster_summaries.get(label, f"Cluster {label}")
                
                # Add the summary annotation
                fig.add_annotation(
                    x=position[0],
                    y=position[1],
                    text=summary,
                    font=dict(size=14, color='black', family='Arial Black'),
                    bgcolor='white',
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=4,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                )
        
        # Add noise points if any exist
        if sum(noise_mask) > 0:
            noise_data = df[noise_mask]
            fig.add_trace(go.Scatter(
                x=noise_data['X'],
                y=noise_data['Y'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='grey',
                    opacity=0.7,
                    symbol='x',
                    line=dict(width=2, color='black')
                ),
                customdata=noise_data['content_display'],
                hovertemplate='<b>Noise</b><br>%{customdata}<extra></extra>',
                name='Noise',
                text=noise_data.index if len(noise_data) < 50 else None,
                textposition="top center",
                textfont=dict(size=10, color='gray')
            ))
            
        # Make sure axes ranges are appropriate and equal
        all_x = df['X'].values
        all_y = df['Y'].values
        x_range = [all_x.min() - 0.5, all_x.max() + 0.5]
        y_range = [all_y.min() - 0.5, all_y.max() + 0.5]
        
        # Ensure equal scaling to maintain proper visual distances
        range_max = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        x_center = sum(x_range) / 2
        y_center = sum(y_range) / 2
        x_range = [x_center - range_max/2, x_center + range_max/2]
        y_range = [y_center - range_max/2, y_center + range_max/2]
        
        # Update layout for a more attractive appearance
        fig.update_layout(
            title={
                'text': 'Interactive Document Clusters Visualization',
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            legend_title='Clusters',
            template='plotly_white',
            width=1600,
            height=900,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            # Add stats to the corner
            annotations=[
                dict(
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    text=f"Documents: {len(df)}<br>Clusters: {len(unique_labels)}<br>Noise: {sum(noise_mask)}",
                    showarrow=False,
                    font=dict(size=14),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    align="left"
                )
            ]
        )
        
        # Improve axis
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        if save_path:
            if save_path.endswith('.png'):
                img_path = save_path
            else:
                img_path = save_path.replace('.html', '.png') if save_path.endswith('.html') else f"{save_path}.png"
            fig.write_image(img_path, scale=2)  # Higher scale for better quality
            logging.info(f"Saved visualization to {img_path}")
            
            # Also save interactive HTML version
            html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else f"{save_path}.html"
            fig.write_html(html_path, include_plotlyjs='cdn')
            logging.info(f"Saved interactive visualization to {html_path}")
        
        fig.show()
        
    def plot_distributions(self, save_dir=None):
        """
        Create and save distribution plots related to clustering
        
        Args:
            save_dir: Directory to save the distribution plots
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Create distribution plots
        self._plot_cluster_sizes(save_dir)
        self._plot_embedding_distances(save_dir)
        self._plot_cluster_quality(save_dir)
        
    def _plot_cluster_sizes(self, save_dir=None):
        """Plot the distribution of cluster sizes with enhanced visualization"""
        # Skip the noise cluster (-1)
        cluster_sizes = {k: len(v) for k, v in self.label2docs.items() if k != -1}
        if not cluster_sizes:
            logging.warning("No clusters found to plot distribution")
            return
            
        # Set aesthetic style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort clusters by size for better visualization
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        cluster_ids = [str(k) for k, v in sorted_clusters]
        sizes = [v for k, v in sorted_clusters]
        
        # Create a visually appealing bar chart
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(cluster_ids)))
        bars = ax.bar(cluster_ids, sizes, color=colors, width=0.7, edgecolor='white', linewidth=1)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        # Add cluster summary labels if available
        if self.cluster_summaries:
            for i, cluster_id in enumerate(cluster_ids):
                if int(cluster_id) in self.cluster_summaries:
                    label_text = self.cluster_summaries[int(cluster_id)]
                    # Truncate long labels
                    if len(label_text) > 20:
                        label_text = label_text[:18] + '...'
                    
                    ax.text(
                        i,
                        -5,  # Place below the x-axis
                        label_text,
                        ha='center',
                        va='top',
                        fontsize=9,
                        rotation=45,
                        color='#555555'
                    )
        
        # Add useful annotations
        total_docs = sum(sizes)
        mean_size = total_docs / len(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        
        stats_text = (
            f"Total Documents: {total_docs}\n"
            f"Number of Clusters: {len(sizes)}\n"
            f"Average Cluster Size: {mean_size:.1f}\n"
            f"Largest Cluster: {max_size}\n"
            f"Smallest Cluster: {min_size}"
        )
        
        # Add a text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
        # Improve the appearance
        ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Documents Across Clusters', fontsize=16, fontweight='bold', pad=20)
        
        # Add grid only for y-axis
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Tighten layout and adjust to make room for rotated labels
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'cluster_sizes.png'), bbox_inches='tight', dpi=300)
            logging.info(f"Saved cluster size distribution to {os.path.join(save_dir, 'cluster_sizes.png')}")
        
        plt.show()
        
    def _plot_embedding_distances(self, save_dir=None):
        """Plot the distribution of embedding distances with enhanced visualization"""
        from scipy.spatial.distance import pdist
        
        # Set aesthetic style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Calculate pairwise distances
        distances = pdist(self.embeddings, metric=self.umap_metric)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a more attractive histogram with custom colors
        n, bins, patches = ax.hist(distances, bins=50, alpha=0.8, color='#61a4b3', edgecolor='white', linewidth=0.8)
        
        # Add a kernel density estimate
        from scipy.stats import gaussian_kde
        density = gaussian_kde(distances)
        x_vals = np.linspace(min(distances), max(distances), 1000)
        y_vals = density(x_vals) * len(distances) * (bins[1] - bins[0])  # Scale to match histogram
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Density Estimate')
        
        # Add statistics
        stats_text = (
            f"Mean Distance: {np.mean(distances):.4f}\n"
            f"Median Distance: {np.median(distances):.4f}\n"
            f"Std Dev: {np.std(distances):.4f}\n"
            f"Min: {np.min(distances):.4f}\n"
            f"Max: {np.max(distances):.4f}"
        )
        
        # Add a text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Mark mean and median
        ax.axvline(np.mean(distances), color='r', linestyle='dashed', alpha=0.7, label='Mean')
        ax.axvline(np.median(distances), color='g', linestyle='dashed', alpha=0.7, label='Median')
        
        # Add DBSCAN eps threshold if available
        if hasattr(self, 'dbscan_eps'):
            ax.axvline(self.dbscan_eps, color='orange', linestyle='solid', alpha=0.7, 
                      linewidth=2, label=f'DBSCAN eps={self.dbscan_eps}')
        
        # Improve the appearance
        ax.set_xlabel('Distance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of Pairwise Distances ({self.umap_metric} metric)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set grid
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'embedding_distances.png'), bbox_inches='tight', dpi=300)
            logging.info(f"Saved embedding distances distribution to {os.path.join(save_dir, 'embedding_distances.png')}")
        
        plt.show()
        
    def _plot_cluster_quality(self, save_dir=None):
        """Plot measures of cluster quality with enhanced visualization"""
        # Skip the noise cluster (-1)
        quality_metrics = {}
        
        for label, doc_indices in self.label2docs.items():
            if label == -1 or len(doc_indices) <= 1:
                continue
                
            # Get embeddings for this cluster
            cluster_embeddings = self.embeddings[doc_indices]
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distances to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            # Store metrics
            quality_metrics[label] = {
                'mean_distance': np.mean(distances),
                'max_distance': np.max(distances),
                'min_distance': np.min(distances),
                'std_distance': np.std(distances),
                'size': len(doc_indices)
            }
        
        if not quality_metrics:
            logging.warning("No clusters with more than one document found to assess quality")
            return
            
        # Set aesthetic style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get data for plotting
        labels = list(quality_metrics.keys())
        sizes = [quality_metrics[l]['size'] for l in labels]
        mean_distances = [quality_metrics[l]['mean_distance'] for l in labels]
        std_distances = [quality_metrics[l]['std_distance'] for l in labels]
        
        # Plot 1: Scatter plot of size vs. mean distance with size indicating std deviation
        scatter = ax1.scatter(
            sizes, 
            mean_distances, 
            c=labels, 
            cmap='viridis', 
            alpha=0.7, 
            s=[std * 500 for std in std_distances],  # Size based on std deviation
            edgecolor='white'
        )
        
        # Add labels for each point
        for i, label in enumerate(labels):
            if self.cluster_summaries and label in self.cluster_summaries:
                summary = self.cluster_summaries[label]
                # Truncate long summaries
                if len(summary) > 15:
                    summary = summary[:13] + "..."
                    
                # Create a more attractive label
                ax1.annotate(
                    summary, 
                    (sizes[i], mean_distances[i]),
                    xytext=(7, 0),
                    textcoords='offset points',
                    fontsize=9,
                    color='#333333',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgray')
                )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster ID', fontsize=10)
        
        # Improve appearance
        ax1.set_xlabel('Cluster Size (number of documents)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Distance to Centroid', fontsize=12, fontweight='bold')
        ax1.set_title('Cluster Quality: Mean Distance to Centroid vs Size', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add a legend for bubble size
        handles, labels_legend = [], []
        for std in [min(std_distances), np.median(std_distances), max(std_distances)]:
            handles.append(plt.scatter([], [], s=std*500, color='gray', alpha=0.5))
            labels_legend.append(f'Std Dev: {std:.3f}')
        
        ax1.legend(handles, labels_legend, loc='upper right', title='Bubble Size Represents')
        
        # Add trend line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, mean_distances)
        x_line = np.array([min(sizes), max(sizes)])
        y_line = slope * x_line + intercept
        
        # Add regression line and correlation information
        ax1.plot(x_line, y_line, color='red', linestyle='--', 
                alpha=0.7, label=f'r = {r_value:.2f}')
        ax1.legend(loc='best')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        
        # Plot 2: Quality metrics comparison across clusters
        # Sort clusters by size to see if there's a pattern
        sorted_indices = np.argsort(sizes)[::-1]  # Descending
        
        # Make sure sorted_indices is within range of labels
        sorted_indices = sorted_indices[sorted_indices < len(labels)]
        
        if len(sorted_indices) == 0:
            # No valid indices, skip second plot
            ax2.set_title('No data for comparison plot', fontsize=14)
            ax2.axis('off')
        else:
            sorted_labels = [labels[i] for i in sorted_indices]
            
            # Create a table-like visualization showing multiple metrics
            width = 0.2
            indices = np.arange(len(sorted_labels))
            
            # Normalize metrics for comparison
            norm_sizes = [s/max(sizes) for s in sizes]
            norm_means = [m/max(mean_distances) for m in mean_distances]
            norm_stds = [s/max(std_distances) for s in std_distances]
            
            sorted_norm_sizes = [norm_sizes[i] for i in sorted_indices]
            sorted_norm_means = [norm_means[i] for i in sorted_indices]
            sorted_norm_stds = [norm_stds[i] for i in sorted_indices]
            
            # Plot the metrics as grouped bars
            bars1 = ax2.bar(indices - width, sorted_norm_sizes, width, label='Relative Size', 
                           color='#5790fc', alpha=0.7)
            bars2 = ax2.bar(indices, sorted_norm_means, width, label='Relative Mean Distance', 
                           color='#d95f02', alpha=0.7)
            bars3 = ax2.bar(indices + width, sorted_norm_stds, width, label='Relative Std Dev', 
                           color='#7570b3', alpha=0.7)
            
            # Add labels and improve appearance
            ax2.set_ylabel('Normalized Value (0-1)', fontsize=12, fontweight='bold')
            ax2.set_title('Comparison of Cluster Metrics (Ordered by Size)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.set_xticks(indices)
            ax2.set_xticklabels([f'C{l}' for l in sorted_labels], rotation=45)
            ax2.legend()
            
            # Add grid
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax2.set_axisbelow(True)
            
            # Add a summary for the metrics
            metrics_summary = (
                f"Quality Overview:\n"
                f"- Best Coherence: Cluster {labels[np.argmin(mean_distances)]}\n"
                f"- Most Varied: Cluster {labels[np.argmax(std_distances)]}\n"
                f"- Largest: Cluster {labels[np.argmax(sizes)]}\n"
                f"- Correlation (Size vs Distance): {r_value:.2f}"
            )
            
            # Add the summary as a text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax2.text(0.02, 0.97, metrics_summary, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'cluster_quality.png'), bbox_inches='tight', dpi=300)
            logging.info(f"Saved cluster quality visualization to {os.path.join(save_dir, 'cluster_quality.png')}")
        
        plt.show()
