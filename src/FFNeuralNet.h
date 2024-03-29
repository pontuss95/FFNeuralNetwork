
#ifndef FF_NEURAL_NET
#define FF_NEURAL_NET
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <signal.h>
#include <string.h>
#include "floatCsvReader.h"
typedef struct DMatrix
{
    double *p;
    int NRows;
    int NCols;
} DMatrix;

typedef struct Neuron
{
    double *Weights;           // weights, first weight is bias.
    double *derivWeights;      // Derivative of weights, used to take gradient descent step.
    double *movAvgSquareDer;  // deriv used for RMSProp
    double *momentumGrad;  //deriv used for momentum training
    double (*actFunc)(double); // ActvFunc
    double derivActFunVal;     //Activation function derivative.
    double (*derivActFunc)(double); // derivative of actvFunc w.r.t to input. Used for back-prop
    double derivWrtCostFun;         // Derivative from before non-linear activation fun to cost.
} Neuron;

typedef struct Layer
{
    Neuron *Neurn;
    double *output; // Output of layer
    int LSize;
    // double *derivWrtCostFun;
} Layer;

typedef struct Network
{
    int NLayers;
    int InpSize;
    int OutSize;
    Layer *Layers;
    double *DCostDLastLay;
    void (*CostFun)(double *, double *, int, double *, double *);
} Network;

typedef struct dataSet
{
    // Original dataset can be yielded by
    // yAugmented = y*yGain+yOffs;
    unsigned int nInps;
    unsigned int nOuts;
    unsigned int nDataPoints;

    double **yAugmented;
    double **xAugmented;
    double *yGain;
    double *yOffs;
    double *xGain;
    double *xOffs;
} dataSet;

void Layer_forwardProp(Layer *, double *, int);
void forwardProp(Network *, double *);
double back_prop(Network *, double *, double *);
void rms(double *, double *, int, double *, double *);
double Sigmoid(double);
double DerivSigmoid(double);
double LinearFun(double val);
double derivLinearFun(double val);
void InitializeWeightsAndBiases(Network *);
void InitializeLayers(Network *);
Network *InitializeNetwork(int, int *, int, int outSize);
dataSet *GenerateSineInputData(int);
void gradientDescent(Network *, int, double, double, double);
void printAllVals(Network *);
dataSet *formatCsvData(Array *inpArray, Array *outArray);
void printData(dataSet *datSet);
dataSet *generateStep(int);
double calculateGradientNorm(Network *Net);
#endif