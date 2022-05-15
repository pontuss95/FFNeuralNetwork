#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    double derivActFunVal;
    double (*derivActFunc)(double); // derivative of actvFunc w.r.t to input. Used for back-prop
    double derivWrtCostFun; //Derivative from before non-linear activation fun to cost.
} Neuron;

typedef struct Layer
{
    Neuron *Neurn;
    double *output; // Output of layer
    int LSize;
    int inpSize;
    //double *derivWrtCostFun;
} Layer;

typedef struct Network
{
    int NLayers;
    int *LSize;
    int InpSize;
    int OutSize;
    Layer *Layers;
    double *DCostDLastLay;
    void (*CostFun)(double *, double *, int, double *, double *)
} Network;

typedef struct dataSet
{
    //Original dataset can be yielded by 
    //yAugmented = y*yGain-yOffs;    

    double **yAugmented;
    double **xAugmented;
    double yGain;
    double yOffs;
    double xGain;
    double yOffs;
} dataSet;

void Layer_forwardProp(Layer*, double*, int);
void forwardProp(Network*, double*);
double back_prop(Network*, double*, double*);
void rms(double *, double *, int, double *, double *);
double Sigmoid(double);
double DerivSigmoid(double);