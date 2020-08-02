import torch
import dgl
from model import RGAT

graph = dgl.DGLGraph()
graph.add_nodes(5)
graph.add_edges([0, 0, 0, 0], [1, 2, 3, 4])
edge_embed = torch.ones([4, 10])
graph.edata['h'] = edge_embed
gat = RGAT(10, 10, 2, 4)
input = torch.rand([5, 10])
tensor = gat(graph, input)
pass
