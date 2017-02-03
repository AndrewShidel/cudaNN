#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <unordered_set>

#include <curand.h>
#include <curand_kernel.h>

using namespace std;
struct Edge;
struct Vertex;
struct Layer;
int random(int min, int max);
float average(float average, float dataPoint);
void setupListeners();

template<typename T>
inline void removeFromVector(vector<T> & v, const T & item);

struct Edge {
    float weight;
    float change;
    Vertex* from;
    Vertex* to;
};

struct Vertex {
    vector<Edge*> inputs;
    vector<Edge*> outputs;
    float output;
    float bias;
    int layer;
    bool isOutput;
    int index;
    float delta;
    float error;
};

struct Layer {
    vector<Vertex*> nodes;
    vector<Edge*> edges;
    int* edgeNodeMapping;
};

class NN {
public:
    NN(bool useGPU, int input, vector<int> hidden, int output);

    vector<Layer> layers;
    vector<Vertex*> nodes;
    vector<Edge*> edges;
    int inputSize;
    int outputSize;
    int outputIdx;
    bool useGPU;
    int nodeIdx;

    Vertex* addVertex(std::vector<Vertex*> inputs, std::vector<Vertex*> outputs);
    Vertex* addVertex(int inputCount, int outputCount, int layer, bool useBasicLayering);
    Vertex* addVertex(int inputs, int outputs, bool isInput, bool isOutput, int layer, bool useStrictLayers);
    void removeVertex();
    Edge* addEdge(Vertex* from, Vertex* to, bool addLayer);
    Edge* addEdge(Vertex* from, Vertex* to);
    void removeEdge(Edge* edge);
    int findLayer(Vertex* vertex);
    void initRNGCPU();
    float trainGPU(vector<float> inputs, vector<float> target);
    float trainGPU(vector<float> inputs, vector<float> target, float learningRate, float momentum);
    vector<float> runGPULauncher(vector<float>& inputs);
    vector<float> runCpu(vector<float>& inputs);
    vector<float> run(vector<float>& inputs);
    void updateHostMemory();
    void updateDeviceMemory();
    void print(ostream& output);
    double layerDist(double x, int mean);

    // CUDA pointers
    float* d_weights;
    float* d_outputs;
    float* d_bias;
    int* d_edgeNodeMappingTo;
    int* d_edgeNodeMappingFrom;
    float* d_errors;
    float* d_deltas;
    float* d_changes;
    float* d_target;
    bool* d_managementThreads;
    curandState_t* RNGStates;
};

bool userEntryMode = false;
bool running = true;
void my_handler(int s){
    printf("Caught signal %d. Entering user input mode.\n",s);
    if (userEntryMode) {
        running = false;
    }
    userEntryMode = true;
    setupListeners();
}
void setupListeners() {
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
}

/*int test(bool useGPU, vector<vector<float>> inputs, vector<vector<float>> targets, vector<int> hidden) {
    int inputSize = inputs[0].size();
    int outputSize = targets[0].size();
    srand(0);
    setupListeners();
    NN nn(useGPU, inputSize, hidden, outputSize);

    //ofstream outputFile(string(useGPU?"gpu":"cpu") + ".graph", ofstream::out);
    //nn.print(outputFile);

    clock_t begin = clock();
    vector<float> target(outputSize);
    vector<float> input(inputSize);

    float error = 1.0;

    do {
        int index = random(0, inputSize);
        if (userEntryMode) {
            std::cout << "\n";
            cin >> iInput;
            std::cout << "\n";
        }
    } while(error > 0 && running);
}*/

int test(bool useGPU, int inputSize, vector<int> hidden, int outputSize) {
    srand(0);
    setupListeners();
    NN nn(useGPU, inputSize, hidden, outputSize);
    ofstream outputFile(string(useGPU?"gpu":"cpu") + ".graph", ofstream::out);
    nn.print(outputFile);

    clock_t begin = clock();

    vector<float> target(outputSize);
    vector<float> input(inputSize);
    float error = 1.0;
    do {
        int iInput = random(100, 1000);
        if (userEntryMode) {
            std::cout << "\n";
            cin >> iInput;
            std::cout << "\n" << iInput << "\n";
        }
        int n = iInput;
        int i = 0;
        while (n) {
            input[i++] = (n % 10)/10.0;
            n /= 10;
        }

        //input[0] = iInput/1000.0;
        target[0] = iInput%2==0?1.0:0.0;
        //target[0] = (iInput>500?1.0:0.0);

        std::cout << "Expected: " << target[0] << ", Input: " << iInput << ", ";
        if (userEntryMode) {
            float result = nn.runGPULauncher(input)[0];
            cout << "Output: " << result << "\n";
        }else{
            error = average(error, nn.trainGPU(input, target, error*2, error));
            std::cout << "\rError: " << error;
        }
    } while(error > 0.005 && running);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time ms: " << elapsed_secs*1000 << "\n";
    return 0;
}

int main(int argc, char** argv) {
    vector<int> hiddenSizes;
    hiddenSizes.push_back(5);
    hiddenSizes.push_back(10);
    hiddenSizes.push_back(50);
    hiddenSizes.push_back(200);
    hiddenSizes.push_back(50);
    hiddenSizes.push_back(10);
    //hiddenSizes.push_back(1);

    for (int i=0; i<1; i++) {
      test(true, 3, hiddenSizes, 1);
    }
    cout << "--------------------------\n";
    /*for (int i=0; i<1; i++) {
      test(false, 1, hiddenSizes, 1);
    }*/
}

// layers: [inputSize, hiddenSize, outputSize]
NN::NN(bool useGPU, int input, vector<int> hidden, int output) {
    // comment in for release
    //srand(time(NULL));
    inputSize = 0;
    outputSize = 0;
    outputIdx = 0;
    nodeIdx = 0;
    this->useGPU = useGPU;

    int nodes = 0;
    for (int i=0; i<input; ++i) {
        addVertex(vector<Vertex*>(), vector<Vertex*>());
        //addVertex(0, 0, true, false, 0, false);
        nodes++;
        inputSize++;
    }
    for (int i=0; i<hidden.size(); ++i) {
        int prevLayerNodes = nodes;
        for (int j=0; j<hidden[i]; ++j) {
            addVertex(random(1, prevLayerNodes), 0, i+1, false);
            //addVertex(random(1, prevLayerNodes), 0, false, false, i+1, false);
            nodes++;
        }
    }

    for (int i=0; i<output; ++i) {
        addVertex(random(1, nodes), 0, hidden.size()+1, false);
        //addVertex(random(1, nodes), 0, false, true, 2, false);
        outputSize++;
    }

    /*for (int i=0; i<layers.size()-1; ++i) {
        for (int j=0; j<layers[i].nodes.size(); ++j) {
            Vertex* vertex = layers[i].nodes[j];
            if (vertex->outputs.size() == 0) {
                Layer& outputLayer = layers[i+1];
                Vertex* outputVertex = outputLayer.nodes[
                    random(0,
                            outputLayer.nodes.size()-1
                    )
                ];
                addEdge(vertex, outputVertex, true);
            }
        }
    }*/


    if (useGPU) {
        updateDeviceMemory();
    }
}

double NN::layerDist(double x, int mean) {
    return -1*pow(2*x-mean,2)+mean;
}

void NN::updateDeviceMemory() {
    initRNGCPU();
    int edgeCount = edges.size();
    int nodeCount = nodes.size();

    size_t floatEdge = sizeof(float)*edgeCount;
    size_t floatNode = sizeof(float)*nodeCount;
    size_t intEdge = sizeof(int)*edgeCount;

    // TODO Free previous device memory

    // Malloc device memory
    cudaMalloc(&d_weights, floatEdge);
    cudaMalloc(&d_outputs, floatNode);
    cudaMalloc(&d_bias, floatNode);
    cudaMalloc(&d_edgeNodeMappingTo, intEdge);
    cudaMalloc(&d_edgeNodeMappingFrom, intEdge);
    cudaMalloc(&d_errors, floatNode);
    cudaMalloc(&d_deltas, floatNode);
    cudaMalloc(&d_changes, floatEdge);
    cudaMalloc(&d_target, sizeof(float)*outputSize);
    cudaMalloc(&d_managementThreads, sizeof(bool)*edgeCount);

    cudaMemset(d_outputs, 0, floatNode);
    cudaMemset(d_changes, 0, floatEdge);

    float* weights = (float*) malloc(floatEdge);
    float* bias = (float*) malloc(floatNode);
    int* edgeNodeMappingTo = (int*) malloc(intEdge);
    int* edgeNodeMappingFrom = (int*) malloc(intEdge);
    float* errors = (float*) malloc(floatNode);
    float* deltas = (float*) malloc(floatNode);
    float* changes = (float*) malloc(floatEdge);
    bool* managementThreads = (bool*) malloc(sizeof(bool)*edgeCount);

    int currEdge = 0;
    int currNode = 0;
    for (int i=0; i<layers.size(); ++i) {
        int edgesInLayer = 0;
        for (int j=0; j<layers[i].nodes.size(); ++j) {
            Vertex* node = layers[i].nodes[j];

            bias[currNode] = node->bias;
            errors[currNode] = node->error;
            deltas[currNode] = node->delta;

            for (int k=0; k<node->inputs.size(); ++k) {
                edgesInLayer++;
                Edge* edge = node->inputs[k];
                weights[currEdge] = edge->weight;
                changes[currEdge] = edge->change;
                managementThreads[currEdge] = (k==0);

                edgeNodeMappingTo[currEdge] = currNode;
                edgeNodeMappingFrom[currEdge] = edge->from->index;
                currEdge++;
            }
            int inputCount = node->inputs.size();
            node->index = currNode;
            currNode++;
        }
        cout << "Edges In Layer " << i << " = " << edgesInLayer << "\n";
    }

    for (int i=0; i<edgeCount; ++i) {
        if (edgeNodeMappingTo[i] < 0 || edgeNodeMappingTo[i]>nodeCount)
            cout << "edgeNodeMappingTo: [" << i << "]: " << edgeNodeMappingTo[i] << "\n";
        if (edgeNodeMappingFrom[i] < 0 || edgeNodeMappingFrom[i]>nodeCount)
            cout << "edgeNodeMappingFrom: [" << i << "]: " << edgeNodeMappingFrom[i] << "\n";
        if (weights[i] < 0 || weights[i]>1)
            cout << "weights: [" << i << "]: " << weights[i] << "\n";
    }
    for (int i=0; i<edgeCount; ++i) {
        if (edgeNodeMappingTo[i] < 0 || edgeNodeMappingTo[i]>nodeCount)
            cout << "edgeNodeMappingTo: [" << i << "]: " << edgeNodeMappingTo[i] << "\n";
    }

    cudaMemcpy(d_weights, weights, floatEdge, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, floatNode, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeNodeMappingTo, edgeNodeMappingTo, intEdge, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeNodeMappingFrom, edgeNodeMappingFrom, intEdge, cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, bias, floatNode, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deltas, bias, floatNode, cudaMemcpyHostToDevice);
    cudaMemcpy(d_changes, bias, floatNode, cudaMemcpyHostToDevice);
    cudaMemcpy(d_managementThreads, managementThreads, sizeof(bool)*edgeCount, cudaMemcpyHostToDevice);

    free(weights);
    free(bias);
    free(edgeNodeMappingTo);
    free(edgeNodeMappingFrom);
    free(errors);
    free(deltas);
    free(changes);
    free(managementThreads);
}

void NN::updateHostMemory() {
    int edgeCount = edges.size();
    int nodeCount = nodes.size();

    size_t floatEdge = sizeof(float)*edgeCount;
    size_t floatNode = sizeof(float)*nodeCount;
    size_t intEdge = sizeof(int)*edgeCount;
    //size_t intNode = sizeof(int)*nodeCount;

    float* weights = (float*) malloc(floatEdge);
    float* bias = (float*) malloc(floatNode);
    int* edgeNodeMappingTo = (int*) malloc(intEdge);
    int* edgeNodeMappingFrom = (int*) malloc(intEdge);
    float* errors = (float*) malloc(floatNode);
    float* deltas = (float*) malloc(floatNode);
    float* changes = (float*) malloc(floatEdge);

    cudaMemcpy(weights, d_weights, floatEdge, cudaMemcpyDeviceToHost);
    cudaMemcpy(bias, d_bias, floatNode, cudaMemcpyDeviceToHost);
    cudaMemcpy(edgeNodeMappingTo, d_edgeNodeMappingTo, intEdge, cudaMemcpyDeviceToHost);
    cudaMemcpy(edgeNodeMappingFrom, d_edgeNodeMappingFrom, intEdge, cudaMemcpyDeviceToHost);
    cudaMemcpy(errors, d_errors, floatNode, cudaMemcpyDeviceToHost);
    cudaMemcpy(deltas, d_deltas, floatNode, cudaMemcpyDeviceToHost);
    cudaMemcpy(changes, d_changes, floatNode, cudaMemcpyDeviceToHost);

    unordered_set<int> nodesCreated;
    vector<Vertex*> emptyEdgeList;
    for (int i=0; i<edgeCount; ++i) {
        int nodeTo = edgeNodeMappingTo[i];
        int nodeFrom = edgeNodeMappingFrom[i];

        int nodeToCreate = -1;
        if (!nodesCreated.count(nodeTo)) {
            nodeToCreate = nodeTo;
        }else if (!nodesCreated.count(nodeFrom)) {
            nodeToCreate = nodeFrom;
        }

        if (nodeToCreate != -1) {
            nodesCreated.insert(nodeToCreate);
            Vertex* vertex = addVertex(emptyEdgeList, emptyEdgeList);
            vertex->delta = deltas[nodeToCreate];
            vertex->error = errors[nodeToCreate];
            //vertex->change = changes[nodeToCreate];
            vertex->bias = bias[nodeToCreate];
        }
    }

    free(weights);
    free(bias);
    free(edgeNodeMappingTo);
    free(edgeNodeMappingFrom);
    free(errors);
    free(deltas);
    free(changes);
}

vector<float> NN::run(vector<float>& inputs) {
    if (useGPU) {
        return runGPULauncher(inputs);
    }else{
        return runCpu(inputs);
    }
}

__global__ void initRNG(unsigned int seed, curandState_t* states) {
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                         cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}

void NN::initRNGCPU() {
    int N = edges.size()/512+1;
    cudaMalloc((void**) &RNGStates, N * sizeof(curandState_t));
    initRNG<<<N, 1>>>(/*time(0)*/0, RNGStates);
}

__global__ void runGPU(float* weights, int* edgeNodeMappingTo, int* edgeNodeMappingFrom, float* outputs, float* bias, bool* managementThreads, int offset, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x + offset;
    if (id < n) {
        int nodeTo = edgeNodeMappingTo[id];
        int nodeFrom = edgeNodeMappingFrom[id];
        atomicAdd(&outputs[nodeTo], outputs[nodeFrom] * weights[id]);
        /*if (managementThreads[id]) {
            outputs[nodeTo] = 1/(1+exp(-1*(outputs[nodeTo]+bias[nodeTo])));
        }*/
    }
}

__global__ void runGPUPost(float* outputs, float* bias, int offset, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x + offset;
    if (id < n) {
        outputs[id] = 1/(1+exp(-1*(outputs[id]+bias[id])));
    }
}

__global__ void runGPUProb(float* weights, int* edgeNodeMappingTo, int* edgeNodeMappingFrom, float* outputs, float* bias, int offset, int n, curandState_t* states) {
    int id = blockIdx.x*blockDim.x+threadIdx.x + offset;
    if (id < n) {
        int nodeTo = edgeNodeMappingTo[id];
        int nodeFrom = edgeNodeMappingFrom[id];
        if (curand_uniform(&states[blockIdx.x]) >= weights[id]) {
            atomicAdd(&outputs[nodeTo], outputs[nodeFrom]);
        }
        outputs[nodeTo] = 1/(1+exp(-1*(outputs[nodeTo]+bias[nodeTo])));
    }
}

__global__ void runGPUMultiplex(bool* activeEdges, float* weights, int* edgeNodeMappingTo, int* edgeNodeMappingFrom, float* outputs, float* bias, int offset, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x + offset;
    if (id < n) {
        int nodeTo = edgeNodeMappingTo[id];
        int nodeFrom = edgeNodeMappingFrom[id];
        if (activeEdges[id]) {
            atomicAdd(&outputs[nodeTo], outputs[nodeFrom] * weights[id]);
        }
        outputs[nodeTo] = 1/(1+exp(-1*(outputs[nodeTo]+bias[nodeTo])));

        //outputs[nodeTo]
    }
}


/*
learningRate
momentum
target: Node-wise
weights: Edge-wise
outputs: Node-wise
edgeNodeMappingFrom: Edge-wise
edgeNodeMappingTo: Edge-wise
nodeRunCount: Node-wise
initialNodeRunCount: Node-wise
errors: Node-wise
deltas: Node-wise
bias: Node-wise
changes: Edge-wise
n
*/
__global__ void learnGPU(float learningRate,
                         float momentum,
                         float* weights,
                         float* outputs,
                         int* edgeNodeMappingFrom,
                         int* edgeNodeMappingTo,
                         float* errors,
                         float* deltas,
                         float* bias,
                         float* changes,
                         int offset,
                         int n,
                          float* buffer) {
    int id = blockIdx.x*blockDim.x+threadIdx.x+offset;
    if (id < offset+n && id>=0) {
        int nodeTo = edgeNodeMappingTo[id];
        int nodeFrom = edgeNodeMappingFrom[id];
        float output = outputs[nodeFrom];

        float& weight = weights[id];
        float delta = deltas[nodeTo];

        atomicAdd(&errors[nodeFrom], delta * weight);
        deltas[nodeFrom] = errors[nodeFrom] * output * (1-output);

        //atomicAdd(&nodeRunCount[nodeFrom], -1);
        //if (nodeRunCount[nodeFrom] == 0) {
        //bias[nodeTo] += learningRate * delta;
        //    nodeRunCount[nodeFrom] = initialNodeRunCount[nodeFrom];
        //}

        float& change = changes[id];
        change = (learningRate * delta * output)
                    + (momentum * change);
        weight += change;
    }
}

__global__ void learnGPUPost(float* deltas, float* bias, float learningRate, int offset, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x+offset;
    if (id < offset+n && id>=0) {
        bias[id] = learningRate * deltas[id];
    }
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

float NN::trainGPU(vector<float> inputs, vector<float> target) {
    return trainGPU(inputs, target, 0.3, 0.1);
}

float NN::trainGPU(vector<float> inputs, vector<float> target, float learningRate, float momentum) {
    vector<float> results = runGPULauncher(inputs);

    int nodeSize = nodes.size();
    int outputSize = target.size();
    float errors[nodeSize];
    float deltas[nodeSize];

    memset(errors, 0, sizeof errors);
    memset(deltas, 0, sizeof deltas);

    float errorSum = 0;
    for (int i=1; i<=outputSize; ++i) {
        //cout << "output[1] = " << results[results.size()-i] << ", output[2] = " << std::round(results[results.size()-i]) << "\n";
        float output = results[results.size()-i];
        std::cout << "Output: " << output << "\n";
        errors[nodeSize-i] = target[outputSize-i] - output;
        deltas[nodeSize-i] = errors[nodeSize-i] * output * (1-output);
        errorSum += errors[nodeSize-i];
    }
    float error = abs(errorSum/outputSize);
    //std::cout << "Error: " << error << "\n";

    cudaMemcpy(d_errors, &errors[0], nodeSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deltas, &deltas[0], nodeSize*sizeof(float), cudaMemcpyHostToDevice);

    int offset = edges.size();
    int nodesOffset = nodes.size();

    float* d_buffer;
    //cudaMalloc(&d_buffer, offset*sizeof(float));
    for (int i=layers.size()-1; i>0; --i) {
        int N = layers[i].edges.size();
        int nodesN = layers[i].nodes.size();
        offset -= N;
        nodesOffset -= nodesN;
        //std::cout << "N["<<i<<"]: "<<N<<", offset["<<i<<"]: " << offset << "\n";
        int edgeCount = layers[i].edges.size();
        int nodeCount = layers[i].nodes.size();
        int blockSize = 512;// or 64?
        int gridSize = edgeCount/blockSize + 1;

        learnGPU<<<gridSize, blockSize>>>(learningRate,
                momentum,
                d_weights,
                d_outputs,
                d_edgeNodeMappingFrom,
                d_edgeNodeMappingTo,
                d_errors,
                d_deltas,
                d_bias,
                d_changes,
                offset,
                N,
                d_buffer);

        blockSize = 512;// or 64?
        gridSize = nodeCount/blockSize + 1;
        learnGPUPost<<<gridSize, blockSize>>>(d_deltas, d_bias, learningRate, nodesOffset, nodesN);
        //cudaCheckErrors("kernel");
    }
    /*float* buffer = (float*) (sizeof(float)*edges.size());
    cudaMemcpy(buffer, d_buffer, edges.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<edges.size(); ++i) {
        std::cout << "buffer["<<i<<"] = " << buffer[i] << "\n";
    }*/
    return error;
}
vector<float> NN::runGPULauncher(vector<float>& inputs) {
    clock_t begin = clock();
    //cout << "Time ms: " << ((clock() - begin)/(double)CLOCKS_PER_SEC)*1000 << "\n";

    for (int i=inputs.size(); i<nodes.size(); ++i) {
        inputs.push_back(0);
    }

    cudaMemcpy(d_outputs, &inputs[0], inputs.size()*sizeof(float), cudaMemcpyHostToDevice);
    //cudaCheckErrors("copy");
    //cout << "Time Post Copy: " << ((clock() - begin)/(double)CLOCKS_PER_SEC)*1000 << "\n";

    int offset = 0;
    int nodeOffset = layers[0].nodes.size();
    //cout << "\nnodeCount: " << layers[0].nodes.size() << "\n";
    for (int i=1; i<layers.size(); ++i) {
        int edgeCount = layers[i].edges.size();
        int nodeCount = layers[i].nodes.size();
        //cout << "nodeCount: " << nodeCount << "\n";
        int gridSize, blockSize;
        blockSize = 512;// or 64?
        gridSize = edgeCount/blockSize + 1;
        runGPU<<<gridSize, blockSize>>>(d_weights, d_edgeNodeMappingTo, d_edgeNodeMappingFrom, d_outputs, d_bias, d_managementThreads, offset, offset+edgeCount);
        //runGPUProb<<<gridSize, blockSize>>>(d_weights, d_edgeNodeMappingTo, d_edgeNodeMappingFrom, d_outputs, d_bias, offset, offset+edgeCount, RNGStates);

        blockSize = 512;// or 64?
        gridSize = nodeCount/blockSize + 1;
        runGPUPost<<<gridSize, blockSize>>>(d_outputs, d_bias, nodeOffset, nodeOffset+nodeCount);

        //cout << "Time Post Kernel " << i << ": " << ((clock() - begin)/(double)CLOCKS_PER_SEC)*1000 << "\n";
        //cudaCheckErrors("kernel");
        offset += edgeCount;
        nodeOffset += nodeCount;
    }
    int outputLayerSize = layers[layers.size()-1].nodes.size();
    //cout << "\nOutput Size = " << outputLayerSize << "\n";
    float* outputs = (float*) malloc( sizeof(float)*outputLayerSize );
    //float* weights = (float*) malloc( sizeof(float)*edges.size() );

    cudaMemcpy(outputs, d_outputs+(nodes.size()-outputLayerSize), outputLayerSize*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(weights, d_weights, edges.size()*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<edges.size(); ++i) {
    //    std::cout << "Weight["<<i<<"] = " << weights[i] << "\n";
    }

    //cout << "Time post copy output: " << ((clock() - begin)/(double)CLOCKS_PER_SEC)*1000 << "\n";
    vector<float> result(outputs, outputs + outputLayerSize);
    free(outputs);
    return result;
}

vector<float> NN::runCpu(vector<float>& inputs) {
    vector<float> result;
    while (inputs.size() < layers[0].nodes.size()) {
        inputs.push_back(0);
    }

    // Stage input vertices
    for (int i=0; i<inputs.size(); ++i) {
        layers[0].nodes[i]->output = inputs[i];
        result.push_back(inputs[i]);
    }

    int layerSize = layers.size();

    // Forward propegate each layer
    for (int i=1; i<layerSize; ++i) {
        vector<Vertex*> layerNodes = layers[i].nodes;
        for (int j=0; j<layerNodes.size(); ++j) {
            Vertex* node = layerNodes[j];
            float sum = node->bias;
            for (int inputIdx=0; inputIdx < node->inputs.size(); ++inputIdx) {
                Edge* inputEdge = node->inputs[inputIdx];
                sum += inputEdge->weight * inputEdge->from->output;
            }
            float outputValue = 1/(1+exp(-1*sum));
            node->output = outputValue;

            if (i==outputIdx) {
                result.push_back(outputValue);
            }
        }
    }
    return result;
}
void NN::print(ostream& output) {
//    output << "View at: http://www.webgraphviz.com/\n";
    output << "digraph G {\n";
    stringstream edges;
    for (int i=0; i<layers.size(); ++i) {
        output << "\tsubgraph cluster_" << i << " {\n"
             << "\t\tstyle=filled;\n"
             << "\t\tcolor=lightgrey;\n"
             << "\t\tnode [style=filled,color=white];\n";

        for (int j=0; j<layers[i].nodes.size(); ++j) {
            output << "\t\t\"" << layers[i].nodes[j]->index << "\"\n";
            for (int k=0; k<layers[i].nodes[j]->outputs.size(); k++) {
                edges << "\t\"" << layers[i].nodes[j]->index << "\" -> \"" << layers[i].nodes[j]->outputs[k]->to->index << "\";\n";
            }
        }

        output << "\t\tlabel = \"layer #" << i << "\";\n";
        output << "\t}\n";
    }
    output << edges.str();

    output << "}\n";
}

Vertex* NN::addVertex(std::vector<Vertex*> inputs, std::vector<Vertex*> outputs) {
    Vertex* vertex = new Vertex;
    vertex->index = nodeIdx++;
    vertex->bias = 0.0;
    vertex->error = 0.0;
    vertex->delta = 0.0;
    vertex->output = 0.0;

    vector<Edge*> edges;
    for (int i=0; i<inputs.size(); ++i) {
        edges.push_back(addEdge(inputs[i], vertex));
    }

    for (int i=0; i<outputs.size(); ++i) {
        edges.push_back(addEdge(vertex, outputs[i]));
    }

    nodes.push_back(vertex);
    int layerIdx = findLayer(vertex);

    if (layerIdx < (int)layers.size()) {
        layers[layerIdx].nodes.push_back(vertex);
    } else {
        Layer layer;
        layer.nodes.push_back(vertex);
        layers.insert(layers.begin()+layerIdx, layer);
    }
    vertex->layer = layerIdx;
    layers[layerIdx].edges.insert(layers[layerIdx].edges.end(), edges.begin(), edges.end());
    return vertex;
}

Vertex* NN::addVertex(int inputCount, int outputCount, int layer, bool useBasicLayering) {
    vector<Vertex*> inputs;
    vector<Vertex*> outputs;
    bool useBasicLayeringSave = useBasicLayering;
    useBasicLayering = true;
    for (int i=0; i<inputCount; ++i) {
        Layer* inputLayer = useBasicLayering ? &layers[layer-1] : &layers[random(0,layer-1)];
        int inputVertex = random(0, inputLayer->nodes.size()-1);
        inputs.push_back(inputLayer->nodes[inputVertex]);

        if (i==0) useBasicLayering=useBasicLayeringSave;
    }
    for (int i=0; i<outputCount; ++i) {
        Layer* outputLayer;
        if (useBasicLayering) {
            outputLayer = &layers[layer+1];
        }else{
            int outputLayerId = random(0,layer+1);
            if (outputLayerId == layer) {
                outputLayerId++;
            }
            outputLayer = &layers[outputLayerId];
        }
        int outputVertex = random(0, outputLayer->nodes.size()-1);
        outputs.push_back(outputLayer->nodes[outputVertex]);
    }
    return addVertex(inputs, outputs);
}

Vertex* NN::addVertex(int inputs, int outputs, bool isInput, bool isOutput, int layer, bool useStrictLayers) {
    Vertex* vertex = new Vertex;
    vertex->index = nodeIdx++;
    vertex->isOutput = isOutput;
    vertex->bias = 0.0;
    vertex->error = 0.0;
    vertex->delta = 0.0;
    vertex->output = 0.0;
    vector<Edge*> edges;

    inputs = inputs>0?inputs:inputs+1;
    int* edgeNodeMapping = (int*) malloc(sizeof(int)*inputs);

    if (isInput) {
        inputs--;
    }

    if (layer > 0) {
        Layer* inputLayer = &layers[layer-1];
        int inputVertex = random(0, inputLayer->nodes.size()-1);
        edges.push_back(addEdge(inputLayer->nodes[inputVertex], vertex));
        inputs--;
    }

    for (int i=0; i<inputs; ++i) {
        Layer* inputLayer = useStrictLayers ? &layers[layer-1] : &layers[random(0,layer-1)];
        int inputVertex = random(0, inputLayer->nodes.size()-1);
        edges.push_back(addEdge(inputLayer->nodes[inputVertex], vertex));
    }
    for (int i=0; i<outputs; ++i) {
        int outputVertex = random(inputSize, nodes.size());
    }

    nodes.push_back(vertex);
    int layerIdx = findLayer(vertex);

    if (isOutput && outputIdx != 0) {
        isOutput = false;
        layerIdx = outputIdx;
    }

    if (layerIdx < (int)layers.size() && !isOutput) {
        layers[layerIdx].nodes.push_back(vertex);
    } else {
        if (isOutput) {
            outputIdx = layers.size();
            layerIdx = outputIdx;
        }
        Layer layer;
        layer.nodes.push_back(vertex);
        layers.insert(layers.begin()+layerIdx, layer);
    }
    vertex->layer = layerIdx;
    layers[layerIdx].edges.insert(layers[layerIdx].edges.end(), edges.begin(), edges.end());
    return vertex;
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

void NN::removeEdge(Edge* edge) {
    removeFromVector(edge->from->outputs, edge);
    removeFromVector(edge->to->inputs, edge);
    delete edge;
}

Edge* NN::addEdge(Vertex* from, Vertex* to) {
    return addEdge(from, to, false);
}

Edge* NN::addEdge(Vertex* from, Vertex* to, bool addLayer) {
    Edge *edge = new Edge;
    edge->to = to;
    edge->from = from;
    edge->weight = 0.1;
    edge->change = 0.0;
    to->inputs.push_back(edge);
    from->outputs.push_back(edge);
    edges.push_back(edge);
    if (addLayer) {
        layers[to->layer].edges.push_back(edge);
    }
    return edge;
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

float average(float average, float dataPoint) {
    static int N = 20;
    average -= average / N;
    average += dataPoint / N;
    return average;
}
