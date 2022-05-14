#include <math.h>
#include <stdio.h>

typedef struct DMatrix
{
    double *p;
    int NRows;
    int NCols;
} DMatrix;


typedef struct Neuron
{
    double *Weights;                 // weights, first weight is bias.
    double *derivWeights;           // Derivative of weights, used to take gradient descent step.
    double (*actFunc)(double);      // ActvFunc
    double (*derivActFunc)(double); // derivative of actvFunc w.r.t to input. Used for back-prop
} Neuron;

typedef struct Layer
{
    Neuron *Neurn;
    double *output; // Output of layer
    int LSize;
    int inpSize;
    double *derivWrtCostFun;
} Layer;

typedef struct Network
{
    int NLayers;
    int *LSize;
    int InpSize;
    int OutSize;
    Layer *Layers;
    void (*CostFun)(double *, double *, int, double *, double *)
} Network;

double Layer_forwardProp(Layer, double *, int);
void forwardProp(Network, double *);
double back_prop(Network, double *, int, double);
void rms(double *, double *, int, double *, double *);
double Sigmoid(double);
double DerivSigmoid(double);