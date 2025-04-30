# rec_view.py  –  lightweight REPL + CLI for conversation‑graph lookup and recommendation
"""
Example usage
-------------
# start interactive shell
python rec_view.py  --nodes-dir embeddings_json  --csv nodes.csv  --repl

# one-shot recommendation
python rec_view.py  --nodes-dir embeddings_json  --csv nodes.csv  \
                    --recommend new_conv.json --topk 8 --alpha 1 --beta 0.2
"""
import argparse, cmd, json, pathlib, sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# DB‑like view
###############################################################################
class NodeView:
    def __init__(self, csv_path: str, nodes_dir: str):
        self.dir = pathlib.Path(nodes_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(self.dir)

        self.df = pd.read_csv(csv_path)
        # coerce betweenness to numeric, fill NaNs, then min‑max normalise
        bc_raw = pd.to_numeric(self.df["betweenesscentrality"], errors="coerce").fillna(0.0)
        self.df["bc_norm"] = (bc_raw - bc_raw.min()) / (bc_raw.max() - bc_raw.min() + 1e-12)
        self.df.set_index("Label", inplace=True)
        print(f"Loaded {len(self.df):,} node headers from CSV; JSON dir: {self.dir}")

    # ---------------------------------------------------------------------
    def load_embedding(self, label: str) -> np.ndarray:
        """Lazy‑load the embedding vector for a node label."""
        json_path = self.dir / f"{label}.json"
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        with open(json_path) as f:
            vec = json.load(f)["embedding"]
        return np.array(vec, dtype=np.float32)

    # ---------------------------------------------------------------------
    def node_info(self, label: str) -> dict:
        if label not in self.df.index:
            raise KeyError(label)
        row = self.df.loc[label]
        # bring only a subset of interesting columns
        return {
            "Id": int(row["Id"]),
            "Label": label,
            "Degree": row["Degree"],
            "Betweenness": row["betweenesscentrality"],
            "Clustering": row["clustering"],
        }

    # ---------------------------------------------------------------------
    def recommend(self, embed_path: str, alpha=1.0, beta=0.2, topk=10):
        with open(embed_path) as f:
            e_cur = np.array(json.load(f)["embedding"], dtype=np.float32).reshape(1, -1)
        # load all embeddings lazily once and cache
        if not hasattr(self, "_emb_mat"):
            print("Building embedding matrix ...", file=sys.stderr)
            self._emb_mat = np.vstack([self.load_embedding(lbl) for lbl in self.df.index])
            self._bc = self.df["bc_norm"].values.astype(np.float32)
            self._labels = np.array(self.df.index)
        sim = cosine_similarity(e_cur, self._emb_mat)[0]
        scores = alpha * sim + beta * self._bc
        top_idx = scores.argsort()[::-1][:topk]
        return [(self._labels[i], float(scores[i]), float(sim[i]), float(self._bc[i])) for i in top_idx]

###############################################################################
# REPL shell
###############################################################################
class RecShell(cmd.Cmd):
    intro = "Type 'help' for commands; 'exit' or Ctrl‑D to quit."
    prompt = "rec> "

    def __init__(self, nv: NodeView):
        super().__init__()
        self.nv = nv

    def do_info(self, arg):
        """info <label>  – show metrics for a node"""
        try:
            print(self.nv.node_info(arg.strip()))
        except Exception as e:
            print("error:", e)

    def do_recommend(self, arg):
        """recommend <embed.json> [topk] [alpha] [beta]"""
        try:
            parts = arg.split()
            if not parts:
                print("provide path to embedding json")
                return
            embed = parts[0]
            topk  = int(parts[1]) if len(parts) > 1 else 10
            alpha = float(parts[2]) if len(parts) > 2 else 1.0
            beta  = float(parts[3]) if len(parts) > 3 else 0.2
            res = self.nv.recommend(embed, alpha, beta, topk)
            for r in res:
                label, score, sim, bc = r
                print(f"{label:<50} score={score:.4f}  sim={sim:.3f}  bc={bc:.4f}")
        except Exception as e:
            print("error:", e)

    def default(self, line):
        if line in ("exit", "quit"):
            return True
        print("unknown command; type 'help'")

###############################################################################
# CLI entry
###############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--repl", action="store_true", help="enter interactive shell")
    ap.add_argument("--recommend", help="embedding JSON for one‑shot recommend")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=0.2)
    ap.add_argument("--topk",  type=int, default=10)
    args = ap.parse_args()

    nv = NodeView(args.csv, args.nodes_dir)

    if args.repl:
        RecShell(nv).cmdloop()
    elif args.recommend:
        out = nv.recommend(args.recommend, args.alpha, args.beta, args.topk)
        for rank, (lbl, score, sim, bc) in enumerate(out, 1):
            print(f"{rank:2d}. {lbl:<50} score={score:.4f} sim={sim:.3f} bc={bc:.4f}")
    else:
        print("Nothing to do: use --repl or --recommend", file=sys.stderr)
