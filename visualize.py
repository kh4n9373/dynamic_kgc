from flask import Flask, render_template, request, jsonify
from pyvis.network import Network
import networkx as nx
import os
import json
import ast  # For safely evaluating string representations of Python literals
import argparse  # For parsing command-line arguments

# Create argument parser
parser = argparse.ArgumentParser(description='Visualize knowledge graph from triplets file')
parser.add_argument('--triplets_file', type=str, 
                    default='/Users/khangtuan/Documents/dynamic-kg/edc/output/example_target_alignment/iter0/canon_kg.txt',
                    help='Path to the file containing triplets')
args = parser.parse_args()

app = Flask(__name__)

# Create directory for templates if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

def create_knowledge_graph(triplets):
    """
    Create an interactive knowledge graph from triplets
    """
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    # Track node types to assign colors
    subjects = set()
    objects = set()
    
    # Add nodes and edges from triplets
    for triplet in triplets:
        subject, relation, object_ = triplet
        subjects.add(subject)
        objects.add(object_)
        
        # Add edge
        G.add_edge(subject, object_, title=relation)
    
    # Create an interactive network
    net = Network(height="700px", width="100%", bgcolor="#ffffff", 
                  font_color="black", directed=True, notebook=False)
    
    # Configure physics for smoother interaction
    net.barnes_hut(spring_length=200, spring_strength=0.01, 
                   damping=0.09, central_gravity=0.1)
    
    # Add nodes with colors based on their role
    for node in G.nodes():
        if node in subjects and node in objects:
            color = "#e377c2"  # Purple for nodes that are both subject and object
            title = f"Node: {node}\nType: Both Subject & Object"
        elif node in subjects:
            color = "#7bbeff"  # Blue for subjects only
            title = f"Node: {node}\nType: Subject"
        else:
            color = "#7bed9f"  # Green for objects only
            title = f"Node: {node}\nType: Object"
        
        # Add node with hover info and styling
        net.add_node(node, label=node, title=title, color=color, 
                    size=25, borderWidth=2, borderWidthSelected=4,
                    font={'size': 14, 'face': 'arial'})
    
    # Add edges with hover info
    for source, target, attr in G.edges(data=True):
        relation = attr['title']
        net.add_edge(source, target, title=relation, label=relation, 
                    arrows='to', smooth=False,  # Changed from curved to straight
                    color={'color': '#848484', 'highlight': '#ff4500'},
                    width=1.5, physics=True)
    
    # Add configuration options similar to Obsidian
    net.set_options("""
    var options = {
        "nodes": {
            "shape": "dot",
            "borderWidth": 2,
            "shadow": true,
            "font": {
                "size": 14,
                "face": "Tahoma"
            }
        },
        "edges": {
            "color": {
                "inherit": false
            },
            "smooth": false,
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "font": {
                "size": 11,
                "align": "middle"
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0.2,
                "springLength": 100,
                "springConstant": 0.01,
                "nodeDistance": 120,
                "damping": 0.09
            },
            "minVelocity": 0.75,
            "solver": "hierarchicalRepulsion"
        },
        "interaction": {
            "navigationButtons": true,
            "keyboard": true,
            "hover": true,
            "multiselect": true,
            "tooltipDelay": 100,
            "hoverConnectedEdges": true,
            "hideEdgesOnDrag": false
        },
        "manipulation": {
            "enabled": false
        },
        "hover": {
            "enabled": true
        },
        "nodes": {
            "scaling": {
                "min": 25,
                "max": 35,
                "label": {
                    "enabled": true,
                    "min": 14,
                    "max": 18
                }
            }
        }
    }
    """)
    
    # Save to templates directory for Flask
    graph_path = os.path.join("templates", "knowledge_graph.html")
    net.save_graph(graph_path)
    
    # Add custom JavaScript for hover effects
    with open(graph_path, 'r') as file:
        html_content = file.read()
    
    # Add hover effect JavaScript before the closing body tag
    hover_js = """
    <script type="text/javascript">
      // Direct access to vis.js network instance
      var network = null;
      
      // Wait for the network to be fully loaded
      window.addEventListener('load', function() {
        // Access the network directly from the vis global variable
        setTimeout(function() {
          try {
            // Get the network instance directly
            network = document.getElementsByClassName('vis-network')[0].__vis_network__;
            
            if (!network) {
              console.error("Could not find network instance");
              return;
            }
            
            console.log("Network found, setting up hover effects");
            
            // Store original colors and sizes
            var originalColors = {};
            var originalSizes = {};
            var nodes = network.body.data.nodes.get();
            
            // Save original properties
            nodes.forEach(function(node) {
              originalColors[node.id] = JSON.parse(JSON.stringify(node.color || {
                background: node.color || "#ffffff",
                border: "#000000"
              }));
              originalSizes[node.id] = node.size || 25;
            });
            
            // On node hover
            network.on("hoverNode", function(params) {
              console.log("Node hover detected", params);
              var hoveredId = params.node;
              
              // Prepare updates for all nodes
              var nodeUpdates = [];
              
              nodes.forEach(function(node) {
                if (node.id === hoveredId) {
                  // Make hovered node bigger and more vivid
                  nodeUpdates.push({
                    id: node.id,
                    size: originalSizes[node.id] * 1.4
                  });
                } else {
                  // Make other nodes gray
                  nodeUpdates.push({
                    id: node.id,
                    color: {
                      background: "#D3D3D3",
                      border: "#A9A9A9"
                    }
                  });
                }
              });
              
              // Update all nodes at once
              network.body.data.nodes.update(nodeUpdates);
            });
            
            // Reset on blur
            network.on("blurNode", function(params) {
              console.log("Node blur detected");
              
              // Prepare updates to reset all nodes
              var nodeUpdates = [];
              
              nodes.forEach(function(node) {
                nodeUpdates.push({
                  id: node.id,
                  size: originalSizes[node.id],
                  color: originalColors[node.id]
                });
              });
              
              // Update all nodes at once
              network.body.data.nodes.update(nodeUpdates);
            });
            
            console.log("Hover effects setup complete");
          } catch (e) {
            console.error("Error setting up hover effects:", e);
          }
        }, 1000); // Wait 1 second for network to be fully initialized
      });
    </script>
    """
    
    # Insert the hover effect JavaScript before the closing body tag
    modified_html = html_content.replace('</body>', hover_js + '</body>')
    
    # Write the modified HTML back to the file
    with open(graph_path, 'w') as file:
        file.write(modified_html)
    
    return graph_path

# Create index.html template with enhanced UI
def create_index_template():
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Knowledge Graph</title>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                font-family: Arial, sans-serif;
            }
            .container {
                display: flex;
                height: 100%;
            }
            .controls {
                width: 250px;
                padding: 20px;
                background-color: #f5f5f5;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
                z-index: 100;
                overflow-y: auto;
            }
            .controls h1 {
                font-size: 1.5em;
                margin-top: 0;
            }
            .controls h2 {
                font-size: 1.2em;
                margin-top: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input[type="text"], button, select {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                margin-top: 5px;
            }
            button:hover {
                background-color: #45a049;
            }
            .graph-container {
                flex-grow: 1;
                position: relative;
                height: 100%;
            }
            iframe {
                width: 100%;
                height: 100%;
                border: none;
            }
            .checkbox-group {
                margin-bottom: 10px;
            }
            .triplet-form {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .triplets-list {
                margin-top: 15px;
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
            }
            .triplet-item {
                padding: 5px;
                margin-bottom: 5px;
                background-color: #eee;
                border-radius: 3px;
                display: flex;
                justify-content: space-between;
            }
            .remove-btn {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 2px 5px;
                cursor: pointer;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="controls">
                <h1>Knowledge Graph Explorer</h1>
                
                <h2>Search</h2>
                <div class="form-group">
                    <input type="text" id="searchNode" placeholder="Search nodes...">
                    <button onclick="searchNodes()">Search</button>
                </div>
                
                <h2>Filters</h2>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showSubjects" checked> Show Subjects</label>
                </div>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showObjects" checked> Show Objects</label>
                </div>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showBoth" checked> Show Mixed Nodes</label>
                </div>
                
                <h2>View Controls</h2>
                <div class="form-group">
                    <button onclick="resetView()">Reset View</button>
                </div>
                <div class="form-group">
                    <button onclick="expandAll()">Expand Graph</button>
                </div>
                
                <div class="triplet-form">
                    <h2>Add New Data</h2>
                    <div class="form-group">
                        <label>Subject:</label>
                        <input type="text" id="subject" placeholder="E.g., John_Doe">
                    </div>
                    <div class="form-group">
                        <label>Relation:</label>
                        <input type="text" id="relation" placeholder="E.g., student">
                    </div>
                    <div class="form-group">
                        <label>Object:</label>
                        <input type="text" id="object" placeholder="E.g., University">
                    </div>
                    <button onclick="addTriplet()">Add Triplet</button>
                    
                    <h3>Current Triplets</h3>
                    <div class="triplets-list" id="tripletsList"></div>
                    
                    <div class="form-group" style="margin-top: 15px;">
                        <button onclick="regenerateGraph()">Regenerate Graph</button>
                    </div>
                </div>
            </div>
            
            <div class="graph-container">
                <iframe id="graphFrame" src="/graph"></iframe>
            </div>
        </div>
        
        <script>
            // Store triplets in memory
            let triplets = [
                ['John_Doe', 'student', 'National_University_of_Singapore'],
                ['National_University_of_Singapore', 'located_in', 'Singapore'],
                ['John_Doe', 'age', '22']
            ];
            
            // Show initial triplets
            displayTriplets();
            
            // Functions to interact with the graph
            function getGraphDocument() {
                return document.getElementById('graphFrame').contentWindow.document;
            }
            
            function getNetwork() {
                const doc = getGraphDocument();
                const network = doc.querySelector('.vis-network');
                return network ? network.visNetwork : null;
            }
            
            function searchNodes() {
                const network = getNetwork();
                if (!network) return;
                
                const searchTerm = document.getElementById('searchNode').value.toLowerCase();
                const allNodes = network.body.data.nodes.get();
                const nodesToHighlight = allNodes.filter(node => 
                    node.label.toLowerCase().includes(searchTerm));
                
                if (nodesToHighlight.length > 0) {
                    network.focus(nodesToHighlight[0].id, {
                        scale: 1.2,
                        animation: true
                    });
                    
                    network.selectNodes([nodesToHighlight[0].id]);
                }
            }
            
            function applyFilters() {
                const network = getNetwork();
                if (!network) return;
                
                const showSubjects = document.getElementById('showSubjects').checked;
                const showObjects = document.getElementById('showObjects').checked;
                const showBoth = document.getElementById('showBoth').checked;
                
                const allNodes = network.body.data.nodes.get();
                
                allNodes.forEach(node => {
                    const nodeType = node.title.includes('Both') ? 'both' : 
                                    node.title.includes('Subject') ? 'subject' : 'object';
                    
                    const visible = (nodeType === 'both' && showBoth) || 
                                   (nodeType === 'subject' && showSubjects) || 
                                   (nodeType === 'object' && showObjects);
                    
                    network.body.data.nodes.update({id: node.id, hidden: !visible});
                });
            }
            
            // Add event listeners for filters
            document.getElementById('showSubjects').addEventListener('change', applyFilters);
            document.getElementById('showObjects').addEventListener('change', applyFilters);
            document.getElementById('showBoth').addEventListener('change', applyFilters);
            
            function resetView() {
                const network = getNetwork();
                if (network) network.fit({animation: true});
            }
            
            function expandAll() {
                const network = getNetwork();
                if (!network) return;
                
                const options = network.physics.options;
                options.hierarchicalRepulsion.nodeDistance = 200;
                network.physics.setOptions(options);
                network.startSimulation();
            }
            
            function addTriplet() {
                const subject = document.getElementById('subject').value.trim();
                const relation = document.getElementById('relation').value.trim();
                const object = document.getElementById('object').value.trim();
                
                if (subject && relation && object) {
                    triplets.push([subject, relation, object]);
                    displayTriplets();
                    
                    // Clear input fields
                    document.getElementById('subject').value = '';
                    document.getElementById('relation').value = '';
                    document.getElementById('object').value = '';
                } else {
                    alert('Please fill in all three fields');
                }
            }
            
            function removeTriplet(index) {
                triplets.splice(index, 1);
                displayTriplets();
            }
            
            function displayTriplets() {
                const list = document.getElementById('tripletsList');
                list.innerHTML = '';
                
                triplets.forEach((triplet, index) => {
                    const item = document.createElement('div');
                    item.className = 'triplet-item';
                    item.innerHTML = `
                        <div>${triplet[0]} → ${triplet[1]} → ${triplet[2]}</div>
                        <button class="remove-btn" onclick="removeTriplet(${index})">X</button>
                    `;
                    list.appendChild(item);
                });
            }
            
            function regenerateGraph() {
                // Send triplets to server to regenerate graph
                fetch('/update_graph', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ triplets: triplets }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Reload the iframe to show new graph
                        document.getElementById('graphFrame').src = '/graph?t=' + new Date().getTime();
                    }
                });
            }
            
            // Wait for iframe to load
            document.getElementById('graphFrame').onload = function() {
                // Allow some time for the graph to initialize
                setTimeout(() => {
                    const network = getNetwork();
                    if (network) {
                        // Set initial view
                        resetView();
                    }
                }, 1000);
            };
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join("templates", "index.html"), "w") as f:
        f.write(index_html)

def load_triplets_from_file(file_path):
    """
    Load triplets from a file where each line contains an array of triplets
    
    Args:
        file_path: Path to the file containing triplets
        
    Returns:
        List of merged triplets
    """
    all_triplets = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Safely evaluate the string representation of the list
                    triplets_in_line = ast.literal_eval(line)
                    all_triplets.extend(triplets_in_line)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except SyntaxError:
        print(f"Error: Could not parse triplets in {file_path}. Check the file format.")
        return []
    
    return all_triplets

# Path to the triplets file from command-line arguments
triplets_file_path = args.triplets_file

# Load triplets from file
loaded_triplets = load_triplets_from_file(triplets_file_path)

# Use loaded triplets or fallback to default if file is empty/invalid
default_triplets = [
    ['John_Doe', 'student', 'National_University_of_Singapore'],
    ['National_University_of_Singapore', 'located_in', 'Singapore'],
    ['John_Doe', 'age', '22']
]

# Use loaded triplets if available, otherwise use defaults
triplets_to_use = loaded_triplets if loaded_triplets else default_triplets

# Initial setup
create_knowledge_graph(triplets_to_use)
create_index_template()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    return render_template('knowledge_graph.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    data = request.get_json()
    triplets = data.get('triplets', triplets_to_use)
    create_knowledge_graph(triplets)
    return jsonify({"success": True})

if __name__ == '__main__':
    print(f"Loaded {len(triplets_to_use)} triplets from {triplets_file_path if loaded_triplets else 'default values'}")
    app.run(debug=True)