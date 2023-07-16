from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from community import *
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from networkx.algorithms.community import *
from sklearn.metrics import *
from PIL import Image, ImageTk

# Create the main window
r = tk.Tk()
r.geometry('800x500')


nodepath = None
edgespath = None

def load_node_file():
    global nodepath
    nodepath = filedialog.askopenfilename(title="Select Node CSV File")
def load_edge_file():
    global edgespath
    edgespath = filedialog.askopenfilename(title="Select Edge CSV File")




bg_image = Image.open("social.jpeg")
bg_image = bg_image.resize((800, 500), Image.LANCZOS)  # resize image to fit window size
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(r, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

b11 =tk.Button(r, text='Select Nodes CSV file',  fg='black' ,width=20, height=1, activeforeground='black',command=load_node_file)
b22 =tk.Button(r, text='Select Edges CSV file', fg='black', width=20, height=1, activeforeground='black',command=load_edge_file)
b1 =tk.Menubutton(r, text='Direct Connection', width=50, height=5, fg='black', activeforeground='black', background='#45B0CA',)
b2 =tk.Menubutton(r, text='InDirect Connection', width=50, height=5, fg='black', activeforeground='black', background='#45B0CA')

b11.grid(row=0, column=1,pady=80,padx=50)
b22.grid(row=0, column=3)
b1.grid(row=2, column=2,pady=10)
b2.grid(row=3, column=2,pady=20)

def appScreen(x):
    print (x)
    # Create the main window
    root = tk.Tk()
    root.geometry('1250x600')
    root.configure(bg='gray')


    # Create the app bar frame
    app_bar = tk.Frame(root, bg='#45B5CA',width=50, height=10)

    # Create the buttons and add them to the app bar
    newman = tk.Menubutton(app_bar, text='Girvan Newman', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    newman.pack(side=tk.LEFT, padx=10, )

    louvain = tk.Menubutton(app_bar, text='Louvain', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    louvain.pack(side=tk.LEFT, padx=5)

    Density = tk.Menubutton(app_bar, text='Density', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    Density.pack(side=tk.LEFT, padx=5, pady=1)

    NMI = tk.Menubutton(app_bar, text='NMI', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    NMI.pack(side=tk.LEFT, padx=5, pady=1)

    modularity = tk.Menubutton(app_bar, text='Modularity', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    modularity.pack(side=tk.LEFT, padx=5, pady=1)

    Conductance = tk.Menubutton(app_bar, text='Conductance', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    Conductance.pack(side=tk.LEFT, padx=5, pady=1)

    pagerank = tk.Menubutton(app_bar, text='Page Rank', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    pagerank.pack(side=tk.LEFT, padx=5, pady=1)

    degreeCenterality = tk.Menubutton(app_bar, text='Degree Centerality', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    degreeCenterality.pack(side=tk.LEFT, padx=5, pady=1)

    betweennes = tk.Menubutton(app_bar, text='Betweenness', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    betweennes.pack(side=tk.LEFT, padx=5, pady=1)

    closeness = tk.Menubutton(app_bar, text='Closeness', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    closeness.pack(side=tk.LEFT, padx=5, pady=1)

    NodeDegree= tk.Menubutton(app_bar, text='Node Degree', width=15, height=2, fg='white', activeforeground='black', background='#45B0CA')
    NodeDegree.pack(side=tk.LEFT, padx=5, pady=1)




    # Add the app bar to the main window
    app_bar.pack(side=tk.TOP, fill=tk.X)
    # Create the plot area
    plot_area = tk.Canvas(root, width=800, height=500,)
    # Add the plot area to the main window
    plot_area.pack(side=tk.BOTTOM, padx=10, pady=10)
    # Create a figure and axes object
    fig, ax = plt.subplots()
    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=plot_area)
    canvas.draw()
    canvas.get_tk_widget().pack()


    # Load nodes from CSV file


    def GirvanNewman(G):
        # Change the graph to a scatter plot
        ax.clear()


        communities = girvan_newman(G)
        node_groups = []
        for com in next(communities):
            node_groups.append(list(com))

        print(len(node_groups))
        Colors = ['blue', 'green', 'red', 'orange', 'yellow', 'gray']
        h = len(node_groups)
        print(node_groups)

        colors = []
        for node in G:
            for i in range(0, h):
                if node in node_groups[i]:
                    #             print(Colors[i])
                    colors.append(Colors[i])

        print(plt.colormaps)

        # Draw graph using Matplotlib
        postionn = nx.spring_layout(G)
        # print(postionn)
        nx.draw_networkx_nodes(G, postionn, node_color=colors)
        nx.draw_networkx_edges(G, postionn)
        nx.draw_networkx_labels(G, postionn, font_size=12)
        canvas.draw()

    def Louvain(G):

        ax.clear()

        # Find the optimal partition using the Louvain method
        partition = community_louvain.best_partition(G.to_undirected())

        # draw the graph
        pos = nx.spring_layout(G)
        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        x = 0
        for i, community in enumerate(set(partition.values())):
            x = x + 1
            members = [node for node, partition_id in partition.items() if partition_id == community]
            print(f"Community {i}: {members}")
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                               cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.title("No. of communites is: {}".format(x))

        # nx.draw_networkx_labels(G, pos)
        canvas.draw()

    def Nmi(G):

        ax.clear()
        partition = community_louvain.best_partition(G.to_undirected())

        # Convert the partition to a dictionary with community labels as keys and node sets as values
        partition_dict = {}
        for node, community_label in partition.items():
            partition_dict.setdefault(community_label, set()).add(node)

        # Convert the partition dictionary to a list in the same order as the nodes in the graph
        partition_list = []
        for node in G.nodes():
            for community_label, nodes_in_community in partition_dict.items():
                if node in nodes_in_community:
                    partition_list.append(community_label)
                    break

        # Compute the NMI between the partition and a different set of communities
        # (in this example, we assume that the ground truth communities are not available,
        # so we randomly assign each node to a different community as a reference)
        reference_list = [i for i in range(len(G.nodes()))]

        if G.is_directed():
            if nx.is_weighted(G):
                nmi = normalized_mutual_info_score(reference_list, partition_list, weight = 'weight', reversed = True)
            else:
                nmi = normalized_mutual_info_score(reference_list, partition_list, reversed = True)

        else:
            if nx.is_weighted(G):
                nmi = normalized_mutual_info_score(reference_list, partition_list, weight = 'weight')
            else:
                nmi = normalized_mutual_info_score(reference_list, partition_list)

        # Add the community labels as node attributes
        for node, community_label in partition.items():
            G.nodes[node]['community'] = community_label

        # Draw the graph with nodes colored by community
        pos = nx.spring_layout(G)
        node_colors = [G.nodes[node]['community'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20, node_size=50)
        plt.title("ID NMI : {}".format(nmi))

        nx.draw_networkx_edges(G, pos, alpha=0.5)
        canvas.draw()

    def graph_density(G):

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        max_num_edges = num_nodes * (num_nodes - 1) // 2  # Maximum number of edges in a simple undirected graph
        density = num_edges / max_num_edges
        print(f"Density:", density)
        return density

    def Modularity(G):

        ax.clear()
        partition = community_louvain.best_partition(G)

        if G.is_directed():
            if nx.is_weighted(G):
                q = community_louvain.modularity(partition, G, weight = 'weight', reversed = True)
            else:
                q = community_louvain.modularity(partition, G, reversed = True)

        else:
            if nx.is_weighted(G):
                q = community_louvain.modularity(partition, G, weight = 'weight')
            else:
                q = community_louvain.modularity(partition, G)

        # Visualize the graph with the communities
        pos = nx.spring_layout(G)
        cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=50, cmap=cmap,
                               node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.title("Modularity: {}".format(q))

        canvas.draw()

    def Conductancee(G):
        ax.clear()
        partition = community_louvain.best_partition(G)

        for community_id in set(partition.values()):
            nodes_in_community = [n for n in partition.keys() if partition[n] == community_id]
            subgraph = G.subgraph(nodes_in_community)

            if G.is_directed():
                if nx.is_weighted(G):
                    conductance = nx.algorithms.cuts.conductance(G, nodes_in_community, weight='weight', reversed=True)
                else:
                    conductance = nx.algorithms.cuts.conductance(G, nodes_in_community,reversed=True)

            else:
                if nx.is_weighted(G):
                    conductance = nx.algorithms.cuts.conductance(G, nodes_in_community, weight='weight')
                else:
                    conductance = nx.algorithms.cuts.conductance(G, nodes_in_community)

            print(f"Community {community_id}: {conductance}")


        canvas.draw()

    def F1_Score(G):

        ax.clear()
        labels_true = [G.nodes[n]['Gender'] for n in G.nodes()]
        labels_pred = ['M' if G.nodes[n]['Gender'] == 'M' else 'F' for n in G.nodes()]

        if G.is_directed():
            if nx.is_weighted(G):
                f1 = f1_score(labels_true, labels_pred, pos_label='M', average='micro', weight='weight', reversed=True)
            else:
                f1 = f1_score(labels_true, labels_pred, pos_label='M', average='micro', reversed=True)

        else:
            if nx.is_weighted(G):
                f1 = f1_score(labels_true, labels_pred, pos_label='M', average='micro', weight='weight')
            else:
                f1 = f1_score(labels_true, labels_pred, pos_label='M', average='micro')

        # Visualize the graph
        pos = nx.spring_layout(G)
        colors = ['blue' if G.nodes[n]['Gender'] == 'M' else 'red' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=colors, alpha=0.5)
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        plt.title("F1 score: {:.2f}".format(f1))

        canvas.draw()

    def Page_Rank(G):

        ax.clear()

        if G.is_directed():
            if nx.is_weighted(G):
                pr = nx.pagerank(G,weight='weight', reversed=True)
            else:
                pr = nx.pagerank(G,reversed=True)

        else:
            if nx.is_weighted(G):
                pr = nx.pagerank(G,weight='weight')
            else:
                pr = nx.pagerank(G)

        # Create a scatter plot of PageRank scores
        plt.scatter(list(pr.keys()), list(pr.values()))
        plt.title("PageRank Scores")
        plt.xlabel("Node ID")
        plt.ylabel("Score")
        canvas.draw()

    def DegreeCent(G):

        ax.clear()
        # Calculate degree centrality for each node
        if G.is_directed():
            if nx.is_weighted(G):
                dc = nx.degree_centrality(G,weight='weight', reversed=True)
            else:
                dc = nx.degree_centrality(G,reversed=True)

        else:
            if nx.is_weighted(G):
                dc = nx.degree_centrality(G,weight='weight')
            else:
                dc = nx.degree_centrality(G)


        for node, centrality in dc.items():
            print(f" {node}: {centrality}")
        nx.draw(G, node_color=list(dc.values()), node_size=100, )

        sorted_dc = sorted(dc.items(), key=lambda x: x[1], reverse=True)

        # Extract the nodes with the highest degree centrality values
        high_centrality_nodes = sorted(nx.connected_components(G), key=len, reverse=True)[0]
        print(high_centrality_nodes)
        canvas.draw()

    def BetweenessCent(G):

        ax.clear()
        # Compute betweenness centrality for each node
        if G.is_directed():
            if nx.is_weighted(G):
                bc = nx.algorithms.centrality.edge_betweenness_centrality(G, weight = 'weight', reversed = True)
            else:
                bc = nx.algorithms.centrality.edge_betweenness_centrality(G, reversed = True)

        else:
            if nx.is_weighted(G):
                bc = nx.algorithms.centrality.edge_betweenness_centrality(G, weight = 'weight')
            else:
                bc = nx.algorithms.centrality.edge_betweenness_centrality(G)

        for node, centrality in bc.items():
            print(f" {node}: {centrality}")

        # Extract the nodes with the highest degree centrality values
        max_cc = max(bc.items(), key=lambda x: x[1])

        canvas.draw()

    def ClosenessCent(G):


        ax.clear()
        # Compute closeness centrality for each node
        if G.is_directed():
            if nx.is_weighted(G):
                cc = nx.closeness_centrality(G, weight = 'weight', reversed = True)
            else:
                cc = nx.closeness_centrality(G, reversed = True)

        else:
            if nx.is_weighted(G):
                cc = nx.closeness_centrality(G, weight = 'weight')
            else:
                cc = nx.closeness_centrality(G)

        for node, centrality in cc.items():
            print(f" {node}: {centrality}")

        # Find the node with highest closeness centrality
        max_cc = max(cc.items(), key=lambda x: x[1])

        nx.draw(G, node_color=list(cc.values()), node_size=100, )
        plt.title("Node with highest closeness centrality: {}".format(max_cc))

        canvas.draw()

    def Node_Degree(G):

        ax.clear()

        if G.is_directed():
            if nx.is_weighted(G):
                degree_dict = dict(G.degree(),weight = 'weight', reversed = True)
            else:
                degree_dict = dict(G.degree(), reversed = True)

        else:
            if nx.is_weighted(G):
                degree_dict = dict(G.degree(),weight = 'weight')
            else:
                degree_dict = dict(G.degree())

        print(degree_dict)

        node_colors = []
        for node in G.nodes:
            print(node)
            node_colors.append(degree_dict[node])

        postionn = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, postionn, node_color=node_colors)
        nx.draw_networkx_edges(G, postionn)
        nx.draw_networkx_labels(G, postionn, font_size=8)
        plt.title("Graph Based on degree ")

        # plt.axis('off')
        canvas.draw()





    # Define the function that will be called when the buttons are clicked
    def on_button_click(event):

        nodes_df = pd.read_csv(nodepath)
        nodes = []
        for index, row in nodes_df.iterrows():
            nodes.append((row['ID']))

        edges_df = pd.read_csv(edgespath)
        edges = []
        for index, row in edges_df.iterrows():
            edges.append((row['Source'], row['Target']))
        if x == "direct":
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # print (G[0])



        # point1 Girvan Newman method
        if event.widget == newman:

           GirvanNewman(G)

        # point1 Louvain method
        elif event.widget == louvain:

            Louvain(G)


        # point2 NMI
        elif event.widget == NMI:

           Nmi(G.to_undirected())


        # point2 modularity
        elif event.widget == modularity:

            Modularity(G.to_undirected())


        # point2 conductance
        elif event.widget == Conductance:

            Conductancee(G.to_undirected())


        # point2 F1 score
        elif event.widget == Density:

           graph_density(G.to_undirected())


        # point3 pagerank
        elif event.widget == pagerank:

            Page_Rank(G.to_undirected())


        # Point4 Degree Centerality
        elif event.widget == degreeCenterality:

            DegreeCent(G.to_undirected())

        # point4 betweennes
        elif event.widget == betweennes:

           BetweenessCent(G.to_undirected())


        # point4 closeness
        elif event.widget == closeness:

            ClosenessCent(G.to_undirected())


        # point5 node degree
        elif event.widget == NodeDegree:

            Node_Degree(G)



    # Bind the button clicks to the on_button_click function
    newman.bind('<Button-1>', on_button_click)
    louvain.bind('<Button-1>', on_button_click)
    Density.bind('<Button-1>', on_button_click)
    NMI.bind('<Button-1>', on_button_click)
    modularity.bind('<Button-1>', on_button_click)
    Conductance.bind('<Button-1>', on_button_click)
    pagerank.bind('<Button-1>', on_button_click)
    degreeCenterality.bind('<Button-1>', on_button_click)
    betweennes.bind('<Button-1>', on_button_click)
    closeness.bind('<Button-1>', on_button_click)
    NodeDegree.bind('<Button-1>', on_button_click)

    # Start the main event loop
    root.mainloop()


def ss(event):
    if event.widget == b1:
        x="direct"
        print(x)
        r.destroy()
        appScreen(x)
    else:
        x = "indirect"
        print(x)
        r.destroy()
        appScreen(x)



b1.bind('<Button-1>', ss)
b2.bind('<Button-1>', ss)
r.mainloop()


