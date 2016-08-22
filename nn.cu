#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;
struct Edge;
struct Vertex;
struct Layer;
int random(int min, int max);

template<typename T>
inline void removeFromVector(vector<T> & v, const T & item);

struct Edge {
    double weight;
    Vertex* from;
    Vertex* to;
};

struct Vertex {
    vector<Edge*> inputs;
    vector<Edge*> outputs;
    double output;
    double bias;
    int layer;
};

struct Layer {
    vector<Vertex*> nodes;
    vector<Edge*> edges;
    vector<int> edgeNodeMapping;
};

class NN {
public:
    NN(int input, int hidden, int output);

    vector<Layer> layers;
    vector<Vertex*> nodes;
    int inputSize;
    int outputSize;
    void addVertex(int inputs, int outputs);
    void removeVertex();
    Edge* addEdge(Vertex* from, Vertex* to);
    void removeEdge(Edge* edge);
    int findLayer(Vertex* vertex);
    vector<double> runGPULauncher(vector<double> inputs);
    vector<double> runCpu(vector<double>& inputs);
    void print();
};

int main() {
    NN nn(1, 3, 1);
    nn.print();
    vector<double> input;
    input.push_back(0.5);
    vector<double> output = nn.runGPULauncher(input);
    //vector<double> output = nn.runCpu(input);//nn.runGPULauncher(input);
    cout << "Output: " << output[0] << "\n";
}

// layers: [inputSize, hiddenSize, outputSize]
NN::NN(int input, int hidden, int output) {
    // comment in for release
    //srand(time(NULL));
    inputSize = 0;
    outputSize = 0;
    int nodes = 0;
    for (int i=0; i<input; ++i) {
        addVertex(0, 0);
        nodes++;
        inputSize++;
    }
    
    for (int i=0; i<hidden; ++i) {
        addVertex(random(1, nodes), 0);
        nodes++;
    }


    for (int i=0; i<output; ++i) {
        addVertex(random(1, nodes), 0);
        outputSize++;
    }
}


// inputs: input on each edge
// weights: weights for each edge
// edgeNodeMapping: index is an edge, value is a node index
// nodeRunCount: Starts at the number of inputs not node.
// outputs: output of each node
__global__ void runGPU(double* inputs, double* weights, size_t* edgeNodeMapping, size_t* nodeRunCount, double* outputs) {
    size_t id = blockIdx.x*blockDim.x+threadIdx.x;
    size_t node = edgeNodeMapping[id];
    
    outputs[node] += inputs[id];
    nodeRunCount[node]--;
    
    if (nodeRunCount[node] == 0) {
        outputs[node] = 1/(1+exp(-1*outputs[node])); 
        printf("Output: %f\n", outputs[node]);
    }
    outputs[node] = 0.5;
    printf("Output2: %f\n", outputs[node]);
}

vector<double> NN::runGPULauncher(vector<double> inputs) {
    while (inputs.size() < layers[0].nodes.size()) {
        inputs.push_back(0);
    }
    
    // Stage input vertices
    for (int i=0; i<inputs.size(); ++i) {
        layers[0].nodes[i]->output = inputs[i];
    }
    vector<double> result(outputSize);
    for (int i=1; i<layers.size(); ++i) {
        size_t edgeCount = layers[i].edges.size();
        size_t nodeCount = layers[i].nodes.size();

        size_t floatEdge = sizeof(double)*edgeCount;
        size_t sizeEdge = sizeof(size_t)*edgeCount;
        size_t floatNode = sizeof(double)*nodeCount;
        size_t sizeNode = sizeof(size_t)*nodeCount;
        

        double* inputs = (double*) malloc(floatEdge);
        double* weights = (double*) malloc(floatEdge);
        size_t* edgeNodeMapping = (size_t*) malloc(sizeEdge);
        size_t* nodeRunCount = (size_t*) malloc(sizeNode);
        double* outputs = (double*) malloc(floatNode);

        for (int node = 0; node<nodeCount; ++node) {
            nodeRunCount[node] = layers[i].nodes[node]->inputs.size(); 
        }

        int nodeMappingIdx = 0;
        Vertex* currentVertex = layers[i].edges[0]->to;
        cout << "EdgeNodeMapping: ";
        for (int j=0; j<edgeCount; ++j) {
            Edge* edge = layers[i].edges[j];
            inputs[j] = edge->from->output;
            weights[j] = edge->weight;
            edgeNodeMapping[j] = nodeMappingIdx;
            if (edge->to != currentVertex) {
                nodeMappingIdx++;
                currentVertex = edge->to;
            }
            cout << edgeNodeMapping[j] << ", "; 
        }
        cout << "\n";

        // Malloc device memory
        double* d_inputs;
        double* d_weights;
        double* d_outputs;
        size_t* d_edgeNodeMapping;
        size_t* d_nodeRunCount;
        cudaMalloc(&d_inputs, floatEdge);
        cudaMalloc(&d_weights, floatEdge);
        cudaMalloc(&d_outputs, floatNode);
        cudaMalloc(&d_edgeNodeMapping, sizeEdge);
        cudaMalloc(&d_nodeRunCount, sizeNode);
        
        // Copy from host to device
        cudaMemcpy(d_inputs, inputs, floatEdge, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights, floatEdge, cudaMemcpyHostToDevice);
        cudaMemcpy(d_outputs, outputs, floatNode, cudaMemcpyHostToDevice);
        cudaMemcpy(d_edgeNodeMapping, edgeNodeMapping, sizeEdge, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nodeRunCount, nodeRunCount, sizeNode, cudaMemcpyHostToDevice);
        
        // launch kernal
        int blockSize, gridSize; 
        blockSize = 1024;
        gridSize = (int)ceil((float)edgeCount/blockSize);
        runGPU<<<gridSize, blockSize>>>(d_inputs, d_weights, d_edgeNodeMapping, d_nodeRunCount, d_outputs);       

        // copy result back to host
        cudaMemcpy(outputs, d_outputs, floatNode, cudaMemcpyDeviceToHost);

        // Process result
        for (int j=0; j<nodeCount; ++j) {
            layers[i].nodes[j]->output = outputs[j];
            cout << "Out: " << outputs[j] << "\n";
            if (i==layers.size()-1) {
                result.push_back(outputs[j]);
            }
        }

        // Release device memory
        cudaFree(d_inputs);
        cudaFree(d_weights);
        cudaFree(d_outputs);
        cudaFree(d_edgeNodeMapping);
        cudaFree(d_nodeRunCount);
 
        // Release host memory
        free(inputs);
        free(weights);
        free(outputs); 
        free(edgeNodeMapping); 
        free(nodeRunCount);
    }
    return result;
}

vector<double> NN::runCpu(vector<double>& inputs) {
    while (inputs.size() < layers[0].nodes.size()) {
        inputs.push_back(0);
    }
    
    // Stage input vertices
    for (int i=0; i<inputs.size(); ++i) {
        layers[0].nodes[i]->output = inputs[i];
    }
    
    vector<double> result;
    int layerSize = layers.size();

    // Forward propegate each layer
    for (int i=1; i<layerSize; ++i) {
        vector<Vertex*> layerNodes = layers[i].nodes;
        for (int j=0; j<layerNodes.size(); ++j) {
            Vertex* node = layerNodes[j];
            double sum = node->bias;
            for (int inputIdx=0; inputIdx<node->inputs.size(); ++inputIdx) {
                Edge* inputEdge = node->inputs[inputIdx];
                sum += inputEdge->weight * inputEdge->from->output;
            }
            double outputValue = 1/(1+exp(-1*sum)); 
            node->output = sum;

            if (i==layerSize-1) {
                result.push_back(sum);
            }
        }
    }
    return result;
}
void NN::print() {
    cout << "View at: http://www.webgraphviz.com/\n";
    cout << "digraph G {\n";
    for (int i=0; i<nodes.size(); ++i) {
        for (int j=0; j<nodes[i]->outputs.size(); j++) {
            cout << "  \"" << nodes[i] << "\" -> \"" << nodes[i]->outputs[j]->to << "\"\n";
        }
    }
    cout << "}\n";
}

void NN::addVertex(int inputs, int outputs) {
    Vertex* vertex = new Vertex;
    vector<Edge*> edges;
    vector<int> edgeNodeMapping;
    for (int i=0; i<inputs; ++i) {
        int inputVertex = random(0, nodes.size()-outputSize-1);
        edges.push_back(addEdge(nodes[inputVertex], vertex));
        edgeNodeMapping.push_back(nodes.size());
    }
    for (int i=0; i<outputs; ++i) {
        int outputVertex = random(inputSize, nodes.size());
        //edges.push_back(addEdge(vertex, nodes[outputVertex]));
    }

    nodes.push_back(vertex);
    int layerIdx = findLayer(vertex);
    cout << "Layer : " << layerIdx << " < " << layers.size() << "\n";
    if (layerIdx < ((int)layers.size())-1) {
        layers[layerIdx].nodes.push_back(vertex);
    } else {
        Layer layer;
        layer.nodes.push_back(vertex);
        layers.insert(layers.begin()+layerIdx, layer);
    }
    layers[layerIdx] = edgeNodeMapping // TODO save as int*?
    layers[layerIdx].edges.insert(layers[layerIdx].edges.end(), edges.begin(), edges.end());
}

// TODO: remove edge from layer
void NN::removeVertex() {
    int vertexIdx = random(0, nodes.size());
    Vertex* vertex = nodes[vertexIdx];
    for (int i=0; i<vertex->inputs.size(); ++i) {
        removeEdge(vertex->inputs[i]);
    }
    for (int i=0; i<vertex->outputs.size(); ++i) {
        removeEdge(vertex->outputs[i]);
    }
    nodes.erase(nodes.begin() + vertexIdx);
}

Edge* NN::addEdge(Vertex* from, Vertex* to) {
    Edge *edge = new Edge;
    edge->to = to;
    edge->from = from;
    edge->weight = 0.25;
    to->inputs.push_back(edge);
    from->outputs.push_back(edge);
    
    return edge;
}

void NN::removeEdge(Edge* edge) {
    removeFromVector(edge->from->outputs, edge);
    removeFromVector(edge->to->inputs, edge);
    delete edge;
}

template<typename T>
inline void removeFromVector(vector<T> & v, const T & item) {
    for(typename vector<T>::iterator iter = v.begin(); iter != v.end(); ++iter) {
        if(*iter == item) {
            v.erase(iter);
            break;
        }
    } 
}

int NN::findLayer(Vertex* vertex) {
    int maxDepth = -1;
    for (int i=0; i<vertex->inputs.size(); ++i) {
        maxDepth = max(maxDepth, vertex->inputs[i]->from->layer);
    }
    vertex->layer = maxDepth+1;
    return maxDepth+1;
}

int random(int min, int max) {
    return rand()%(max-min + 1) + min;    
}

