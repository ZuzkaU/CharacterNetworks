import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import math
import logging

MAX_EDGE_WEIGHT = 20
MAX_NODE_WEIGHT = 3000

def outputGender(G, outdir, orig_name=''):
    prefix = '' if not orig_name else orig_name + '_'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    scaleEdgeWeight(G)
    makeNewNodeWeight(G)
    
    plt.figure(1, figsize=(10,10))
    
    node_sizes = [G.nodes(data=True)[n]['count'] for n in G.nodes]
    widths = [G[u][v]['weight'] for (u, v) in G.edges]
    
    node_colors = []
    for n in G.nodes:
        if G.nodes(data=True)[n]['gender'] == 'F':
            node_colors.append((1, .3, .2, 1))
        elif G.nodes(data=True)[n]['gender'] == 'M':
            node_colors.append((.3, .4, 1, 1))
        else:
            node_colors.append((.5, .5, .5, 1))
    
    edge_colors = []
    for (u, v) in G.edges:
        if G.nodes(data=True)[u]['gender'] == 'F' and G.nodes(data=True)[v]['gender'] == 'F':
            edge_colors.append((1, .4, .3, .8))
        elif G.nodes(data=True)[u]['gender'] == 'M' and G.nodes(data=True)[v]['gender'] == 'M':
            edge_colors.append((.4, .5, 1, .8))
        else:
            edge_colors.append((.7, .7, .7, .8))
    
    pos = nx.circular_layout(G)
    
    nx.draw(G, pos, node_size=node_sizes, width=widths, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    
    labelpos = {}
    for p in pos:
        labelpos[G.nodes[p]['name']] = pos[p]
    G = relabelNodes(G, 'name')
    nx.draw_networkx_labels(G, labelpos, font_size=12, font_weight='bold')
    
    
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.3*x for x in axis.get_xlim()])
    axis.set_ylim([1.3*y for y in axis.get_ylim()])
    
    outfile = os.path.join(outdir, prefix + 'gender.pdf')
    plt.savefig(outfile)
    plt.clf()
    logging.info('Network saved to {}'.format(outfile))
    return
    


def outputThreeNetworks(cooc, conv, gold, outdir, orig_name=''):
    prefix = '' if not orig_name else orig_name + '_'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    conv, cooc, gold = scaleEdgeWeights(conv, cooc, gold)
    conv, cooc, gold = makeNewNodeWeights(conv, cooc, gold)
    
    conv = relabelNodes(conv, 'char_id')
    cooc = relabelNodes(cooc, 'char_id')
    gold = relabelNodes(gold, 'char_id')
    
    nodelist = conv.nodes
    pos = nx.circular_layout(conv)
    
    outputNetwork(cooc, nodelist, pos, os.path.join(outdir, prefix + 'cooc.pdf'), (1,.82,.35))
    outputNetwork(conv, nodelist, pos, os.path.join(outdir, prefix + 'conv.pdf'), (1,.67,.35))
    if gold:
        outputNetwork(gold, nodelist, pos, os.path.join(outdir, prefix + 'gold.pdf'), (1,.47,.35))
    return

def outputNetwork(G, nodelist, pos, outfile, color=(0.5, 0.5, 0.5)):
    plt.figure(1, figsize=(10,10))
    
    node_sizes = []
    for n in nodelist:
        node_sizes.append(G.nodes(data=True)[n]['count'])
    
    edges = G.edges()
    widths = []
    for u, v in edges:
        widths.append(G[u][v]['weight'])
    
    node_color = (max(color[0] - 0.1, 0), max(color[1] - 0.1, 0), max(color[2] - 0.1, 0))
    edge_color = [(color[0], color[1], color[2], (1-math.pow(2, -(G[u][v]['weight'])))) for u, v in edges]
    #edge_color = [(color[0], color[1], color[2], (G[u][v]['weight'] / MAX_EDGE_WEIGHT)) for u, v in edges]
    #edge_color = [(color[0], color[1], color[2], 1) for u, v in edges]

    nx.draw(G, pos, nodelist=nodelist, edgelist=edges, node_size=node_sizes, width=widths,
        node_color=[node_color]*len(nodelist), edge_color=edge_color, with_labels=False)
    
    labelpos = {}
    for p in pos:
        labelpos[G.nodes[p]['name']] = pos[p]
    G = relabelNodes(G, 'name')
    nx.draw_networkx_labels(G, labelpos, font_size=12, font_weight='bold')
    
    
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.3*x for x in axis.get_xlim()])
    axis.set_ylim([1.3*y for y in axis.get_ylim()])
    
    plt.savefig(outfile)
    plt.clf()
    logging.info('Network saved to {}'.format(outfile))
    return




def relabelNodes(G, attr):
    if not G:
        return
    new_names = [(name, data[attr]) for name, data in G.nodes(data=True)]
    mapping = dict(new_names)
    G = nx.relabel_nodes(G, mapping)
    return G


def scaleEdgeWeights(convG, coocG, goldG=None):
    scaleEdgeWeight(convG)
    scaleEdgeWeight(coocG)
    if goldG:
        scaleEdgeWeight(goldG)
    return (convG, coocG, goldG)
    
def scaleEdgeWeight(G):
    this_w = 1
    for u, v, data in G.edges(data=True):
        if data['weight'] > this_w:
            this_w = data['weight']
    
    mult = MAX_EDGE_WEIGHT / this_w
    
    for u, v, data in G.edges(data=True):
        data['weight'] *= mult
    
def makeNewNodeWeights(convG, coocG, goldG=None):
    makeNewNodeWeight(convG)
    makeNewNodeWeight(coocG)
    if goldG:
        makeNewNodeWeight(goldG)
    return (convG, coocG, goldG)
    
def makeNewNodeWeight(G):
    this_w = 1
    node_weights = {}
    for n, data in G.nodes(data=True):
        node_weights[n] = 0
        for m in G.neighbors(n):
            node_weights[n] += G[m][n]['weight']
        if node_weights[n] > this_w:
            this_w = node_weights[n]
    
    mult = MAX_NODE_WEIGHT / this_w
    
    for n, data in G.nodes(data=True):
        data['count'] = (1 + node_weights[n]) * mult

def outputSpeakers(docs, characters, outdir, orig_name=''):
    prefix = '' if not orig_name else orig_name + '_'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, prefix + 'speakers.txt')
    
    character_dict = {}
    for char_id, (variants, gender) in characters.items():
            best_name, best_count = None, 0
            for name, count in variants:
                if count > best_count:
                    best_name = name
                    best_count = count
            character_dict[char_id] = best_name
    
    with open(outfile, 'w') as f:
        for doc in docs:
            f.write(doc.text + '\n')
            if not doc._.speaker_id == None:
                f.write('***SPEAKER: ' + character_dict[doc._.speaker_id] + '\n')
            else:
                f.write('***SPEAKER: None\n')
            f.write('\n')
    logging.info('Speakers saved to {}'.format(outfile))

def outputCharacters(characters, outdir, orig_name=''):
    prefix = '' if not orig_name else orig_name + '_'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, prefix + 'characters.csv')
    
    with open(outfile, 'w') as f:
        for char_id, (variants, gender) in characters.items():
                for name, count in variants:
                    f.write("{},{},{},{}\n".format(char_id, name, gender, count))
    logging.info('Characters saved to {}'.format(outfile))


