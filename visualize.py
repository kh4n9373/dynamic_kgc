from flask import Flask, render_template, request, jsonify
from pyvis.network import Network
import networkx as nx
import os
import json
import ast
import argparse

# Parse command line arguments
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
    """Create an interactive knowledge graph from triplets"""
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
        
        net.add_node(node, label=node, title=title, color=color, 
                    size=25, borderWidth=2, borderWidthSelected=4,
                    font={'size': 14, 'face': 'arial'})
    
    # Add edges with hover info
    for source, target, attr in G.edges(data=True):
        relation = attr['title']
        net.add_edge(source, target, title=relation, label=relation, 
                    arrows='to', smooth=False,
                    color={'color': '#848484', 'highlight': '#ff4500'},
                    width=1.5, physics=True)
    
    # Add network visualization options
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
    
    # Add hover effect JavaScript
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
                  // Make all other nodes slightly faded
                  var fadedColor = Object.assign({}, originalColors[node.id]);
                  
                  if (typeof fadedColor === 'string') {
                    // If color is a string, convert to object
                    fadedColor = {
                      background: fadedColor,
                      border: "#000000"
                    };
                  }
                  
                  // Add opacity to the color
                  fadedColor.opacity = 0.5;
                  
                  nodeUpdates.push({
                    id: node.id,
                    color: fadedColor,
                    size: originalSizes[node.id] * 0.9
                  });
                }
              });
              
              // Apply all updates at once for better performance
              network.body.data.nodes.update(nodeUpdates);
              
              // Get connected edges
              var connectedEdges = network.getConnectedEdges(hoveredId);
              
              // Highlight connected edges
              var edgeUpdates = [];
              network.body.data.edges.get().forEach(function(edge) {
                if (connectedEdges.includes(edge.id)) {
                  edgeUpdates.push({
                    id: edge.id,
                    width: 3,
                    color: { color: '#ff4500', opacity: 1 }
                  });
                } else {
                  edgeUpdates.push({
                    id: edge.id,
                    width: 1,
                    color: { color: '#848484', opacity: 0.4 }
                  });
                }
              });
              
              // Apply edge updates
              network.body.data.edges.update(edgeUpdates);
            });
            
            // On hover end
            network.on("blurNode", function(params) {
              console.log("Node blur detected");
              
              // Reset all nodes to original state
              var nodeUpdates = [];
              
              nodes.forEach(function(node) {
                nodeUpdates.push({
                  id: node.id,
                  color: originalColors[node.id],
                  size: originalSizes[node.id]
                });
              });
              
              // Apply all updates at once
              network.body.data.nodes.update(nodeUpdates);
              
              // Reset all edges
              var edgeUpdates = [];
              network.body.data.edges.get().forEach(function(edge) {
                edgeUpdates.push({
                  id: edge.id,
                  width: 1.5,
                  color: { color: '#848484', opacity: 1 }
                });
              });
              
              // Apply edge updates
              network.body.data.edges.update(edgeUpdates);
            });
            
          } catch (e) {
            console.error("Error setting up hover effects", e);
          }
        }, 1000); // Wait for the network to be fully initialized
      });
    </script>
    """
    
    # Insert hover effect JavaScript before closing body tag
    enhanced_html = html_content.replace('</body>', hover_js + '</body>')
    
    with open(graph_path, 'w') as file:
        file.write(enhanced_html)
    
    # Create a standalone version for easier sharing
    with open('knowledge_graph.html', 'w') as file:
        file.write(enhanced_html)
    
    return graph_path


def create_index_template():
    """Create HTML template for the index page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Visualizer</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            #container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: #4257b2;
                color: white;
                padding: 20px;
                border-radius: 5px 5px 0 0;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                margin: 0;
                font-weight: 500;
            }
            .card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
            }
            textarea {
                width: 100%;
                min-height: 100px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: monospace;
                margin-bottom: 10px;
                resize: vertical;
            }
            #formatInfo {
                font-size: 14px;
                color: #666;
                margin-bottom: 20px;
            }
            .button {
                background-color: #4257b2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            .button:hover {
                background-color: #374499;
            }
            #errorMessage {
                color: #d32f2f;
                margin-top: 10px;
                display: none;
            }
            #loading {
                display: none;
                margin-top: 10px;
                color: #666;
            }
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(66, 87, 178, 0.3);
                border-radius: 50%;
                border-top-color: #4257b2;
                animation: spin 1s ease-in-out infinite;
                margin-right: 10px;
                vertical-align: middle;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            iframe {
                width: 100%;
                height: 700px;
                border: none;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 14px;
                margin-top: 40px;
            }
            .examples {
                margin-top: 10px;
                margin-bottom: 20px;
            }
            .example-button {
                background-color: #f1f1f1;
                border: 1px solid #ddd;
                padding: 5px 10px;
                margin-right: 5px;
                margin-bottom: 5px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                display: inline-block;
                transition: background-color 0.3s;
            }
            .example-button:hover {
                background-color: #e0e0e0;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <div class="header">
                <h1>Knowledge Graph Visualizer</h1>
            </div>
            
            <div class="card">
                <label for="triplets">Enter Knowledge Graph Triplets:</label>
                <div id="formatInfo">
                    Format: Each line should contain a triplet in the form of a Python list: ["Subject", "Relation", "Object"]<br>
                    Example: [["Person", "works_at", "Company"], ["Person", "lives_in", "City"]]
                </div>
                
                <div class="examples">
                    <span class="example-button" onclick="loadExample('basic')">Basic Example</span>
                    <span class="example-button" onclick="loadExample('academic')">Academic Example</span>
                    <span class="example-button" onclick="loadExample('company')">Corporate Example</span>
                </div>
                
                <textarea id="triplets" placeholder='[["Person", "works_at", "Company"], ["Person", "lives_in", "City"]]'></textarea>
                
                <button class="button" onclick="updateGraph()">Visualize Graph</button>
                
                <div id="loading">
                    <div class="spinner"></div> Generating knowledge graph...
                </div>
                <div id="errorMessage"></div>
            </div>
            
            <div class="card">
                <iframe id="graphFrame" src="/graph"></iframe>
            </div>
            
            <div class="footer">
                Knowledge Graph Visualizer | Based on NetworkX, Pyvis, and vis.js
            </div>
        </div>
        
        <script>
            function updateGraph() {
                const tripletsText = document.getElementById('triplets').value.trim();
                const errorMessage = document.getElementById('errorMessage');
                const loading = document.getElementById('loading');
                
                if (!tripletsText) {
                    errorMessage.style.display = 'block';
                    errorMessage.textContent = 'Please enter triplets data.';
                    return;
                }
                
                try {
                    // Try to parse as JSON to validate
                    JSON.parse(tripletsText.replace(/'/g, '"'));
                    
                    errorMessage.style.display = 'none';
                    loading.style.display = 'block';
                    
                    fetch('/update_graph', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ triplets: tripletsText }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        loading.style.display = 'none';
                        if (data.success) {
                            document.getElementById('graphFrame').src = '/graph?' + new Date().getTime();
                        } else {
                            errorMessage.style.display = 'block';
                            errorMessage.textContent = data.error || 'An error occurred.';
                        }
                    })
                    .catch(error => {
                        loading.style.display = 'none';
                        errorMessage.style.display = 'block';
                        errorMessage.textContent = 'An error occurred while updating the graph.';
                        console.error('Error:', error);
                    });
                } catch (e) {
                    errorMessage.style.display = 'block';
                    errorMessage.textContent = 'Invalid triplets format. Please check your input.';
                }
            }
            
            function loadExample(type) {
                let example = '';
                
                if (type === 'basic') {
                    example = '[["Person", "works_at", "Company"], ["Person", "lives_in", "City"], ["Company", "located_in", "City"], ["Person", "friends_with", "Another Person"]]';
                } else if (type === 'academic') {
                    example = '[["Student", "enrolled_in", "Course"], ["Professor", "teaches", "Course"], ["Course", "part_of", "Department"], ["Student", "advised_by", "Professor"], ["Department", "part_of", "University"], ["Professor", "works_at", "University"], ["Student", "studies_at", "University"], ["Professor", "published", "Research Paper"], ["Student", "contributed_to", "Research Paper"], ["Research Paper", "cited_in", "Journal"]]';
                } else if (type === 'company') {
                    example = '[["CEO", "leads", "Company"], ["Employee", "works_for", "Department"], ["Department", "part_of", "Company"], ["Company", "sells", "Product"], ["Customer", "buys", "Product"], ["Product", "has_feature", "Feature"], ["Employee", "reports_to", "Manager"], ["Manager", "reports_to", "CEO"], ["Company", "competes_with", "Competitor"], ["Competitor", "sells", "Alternative Product"]]';
                }
                
                document.getElementById('triplets').value = example;
            }
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join("templates", "index.html"), 'w') as file:
        file.write(html)


def load_triplets_from_file(file_path):
    """Load triplets from a file"""
    triplets = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # First try to parse as JSON
                    triplet_list = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        # Then try to evaluate as Python literal
                        triplet_list = ast.literal_eval(line)
                    except (SyntaxError, ValueError):
                        print(f"Warning: Could not parse line: {line}")
                        continue
                
                # Handle single triplet or list of triplets
                if isinstance(triplet_list, list):
                    if len(triplet_list) == 3 and all(isinstance(x, str) for x in triplet_list):
                        # Single triplet as a list of three strings
                        triplets.append(triplet_list)
                    elif all(isinstance(x, list) for x in triplet_list):
                        # List of triplets
                        for triplet in triplet_list:
                            if len(triplet) == 3 and all(isinstance(x, str) for x in triplet):
                                triplets.append(triplet)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error loading triplets: {e}")
    
    return triplets


# Create an index page
create_index_template()

# Load triplets from file
if args.triplets_file:
    triplets = load_triplets_from_file(args.triplets_file)
    if triplets:
        print(f"Loaded {len(triplets)} triplets from {args.triplets_file}")
        create_knowledge_graph(triplets)
    else:
        print(f"No valid triplets found in {args.triplets_file}")
        # Create an empty graph
        create_knowledge_graph([["Example", "is_a", "Triplet"]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    return render_template('knowledge_graph.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    try:
        data = request.get_json()
        triplets_text = data.get('triplets', '[]')
        
        # Convert string representation to Python object
        triplets_text = triplets_text.replace("'", '"')
        triplets = json.loads(triplets_text)
        
        # Create knowledge graph
        create_knowledge_graph(triplets)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating graph: {e}")
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting Knowledge Graph Visualizer...")
    print("Access the visualizer at http://127.0.0.1:5000/")
    app.run(debug=True)